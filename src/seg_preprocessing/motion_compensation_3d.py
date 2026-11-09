"""
3D Motion Compensation for CEUS Analysis
Based on ILSA tracking
Axis convention: (X, Y, Z, T) = (lateral, depth, elevational, time)
"""

import numpy as np
from scipy.ndimage import shift
from scipy.fft import next_fast_len
from typing import Tuple, List, Optional
from dataclasses import dataclass
from collections import Counter
from tqdm import tqdm

from .motion_compensation_3d_gpu import GPU_AVAILABLE, cp

# Smallest fraction of the template's valid voxels that must overlap valid image
# voxels for a shift to be scored. Below this, the NCC rests on too few voxels to
# be trusted and tends to reward shifting the template out of the image.
MIN_OVERLAP = 0.5


def masked_ncc(search, search_mask, template, template_mask, xp=np):
    """Valid-mode zero-mean NCC of template over search, using only voxels where
    both masks are True at that shift (Padfield, "Masked object registration in
    the Fourier domain", 2012).

    Voxels outside the volume or the imaged sector carry no image, so a template
    partly shifted out of the image is scored on the part still inside rather
    than against padding. Shifts overlapping less than MIN_OVERLAP of the
    template's valid voxels score -1.

    Args:
        search, search_mask: 3D search region and its validity mask
        template, template_mask: 3D template and its validity mask
        xp: array module, numpy or cupy

    Returns:
        Host array of shape search.shape - template.shape + 1, in [-1, 1].
    """
    corr_shape = tuple(s - t + 1 for s, t in zip(search.shape, template.shape))
    ms = xp.asarray(search_mask, dtype=xp.float64)
    mt = xp.asarray(template_mask, dtype=xp.float64)
    n_template = float(mt.sum())
    if n_template == 0 or float(ms.sum()) == 0:
        return np.full(corr_shape, -1.0)

    def standardize(v, m):
        # Zero-mean, unit-std over the valid voxels keeps the FFT sums below
        # well conditioned; NCC itself is invariant to this.
        v = xp.asarray(v, dtype=xp.float64)
        n = m.sum()
        mean = (v * m).sum() / n
        std = xp.sqrt((((v - mean) * m) ** 2).sum() / n)
        return (v - mean) / xp.where(std == 0, 1, std) * m

    s = standardize(search, ms)
    t = standardize(template, mt)

    # Circular correlation at an FFT size >= search shape has no wraparound
    # over the valid region.
    fft_shape = tuple(next_fast_len(n, real=True) for n in search.shape)
    valid = tuple(slice(0, c) for c in corr_shape)
    f_m, f_s, f_s2 = (xp.fft.rfftn(a, fft_shape) for a in (ms, s, s * s))
    k_m, k_t, k_t2 = (xp.conj(xp.fft.rfftn(a, fft_shape)) for a in (mt, t, t * t))

    def corr(f, k):
        return xp.fft.irfftn(f * k, fft_shape)[valid]

    n = xp.round(corr(f_m, k_m))      # overlapping valid voxels per shift
    sum_s, sum_s2 = corr(f_s, k_m), corr(f_s2, k_m)
    sum_t, sum_t2 = corr(f_m, k_t), corr(f_m, k_t2)
    sum_st = corr(f_s, k_t)

    n_safe = xp.maximum(n, 1)
    num = sum_st - sum_s * sum_t / n_safe
    var_s = xp.maximum(sum_s2 - sum_s ** 2 / n_safe, 0)
    var_t = xp.maximum(sum_t2 - sum_t ** 2 / n_safe, 0)
    den = xp.sqrt(var_s * var_t)

    ok = (n >= MIN_OVERLAP * n_template) & (den > 1e-6 * n_safe)
    ncc = xp.clip(xp.where(ok, num / xp.where(ok, den, 1), -1.0), -1.0, 1.0)
    return cp.asnumpy(ncc) if xp is not np else ncc


def compute_sector_mask(bmode: np.ndarray, frame_idx: int = 0) -> np.ndarray:
    """Boolean mask of the imaged sector, True where the scan has data.

    A sector scan fills only part of its array; the rest is padding. Derive this
    from the B-mode, never from the contrast volume -- unenhanced tissue is
    legitimately zero in CEUS, so thresholding CEUS would throw away exactly the
    voxels perfusion analysis needs. The sector is fixed by probe geometry, so
    one frame describes the whole cine.
    """
    vol = bmode[..., frame_idx] if bmode.ndim == 4 else bmode
    return np.asarray(vol) > 0


@dataclass
class MotionCompensationResult:
    """Store motion compensation results efficiently"""
    translation_vectors: np.ndarray
    reference_frame: int
    correlations: np.ndarray
    reference_bbox: 'BoundingBox3D'
    tracked_bboxes: List['BoundingBox3D']
    # Imaged sector (see compute_sector_mask). VOI shifted outside it is dropped
    # by apply_to_mask. None means no sector clipping, which is only right when
    # the scan fills its whole array.
    sector_mask: Optional[np.ndarray] = None
    
    def get_translation(self, frame_idx: int) -> Tuple[float, float, float]:
        return tuple(self.translation_vectors[frame_idx])
    
    def apply_to_mask(self, mask_3d: np.ndarray, frame_idx: int, order: int = 0,
                      valid: Optional[np.ndarray] = None) -> np.ndarray:
        """Translate mask_3d onto frame_idx.

        Shifting alone only drops VOI that leaves the *array*. In a sector scan
        the imaged region is much smaller than the array, so a VOI can travel a
        long way out of the imaged sector while every voxel stays in bounds and
        nothing is dropped. `valid` -- a boolean array, True where this frame has
        image data -- drops that part too; it defaults to self.sector_mask.
        """
        dx, dy, dz = self.get_translation(frame_idx)
        shifted = shift(mask_3d, shift=[dx, dy, dz], order=order, cval=0,
                    prefilter=True if order > 0 else False)
        if valid is None:
            valid = self.sector_mask
        if valid is not None:
            shifted = np.where(valid, shifted, 0).astype(mask_3d.dtype)
        return shifted
    
    def apply_to_all_frames(self, mask_3d: np.ndarray, order: int = 0,
                            valid_4d: Optional[np.ndarray] = None) -> np.ndarray:
        """As apply_to_mask, over every frame. valid_4d is indexed [..., frame]."""
        n_frames = len(self.translation_vectors)
        mc_mask_4d = np.zeros((*mask_3d.shape, n_frames), dtype=mask_3d.dtype)
        for frame_idx in range(n_frames):
            valid = None if valid_4d is None else valid_4d[..., frame_idx]
            mc_mask_4d[..., frame_idx] = self.apply_to_mask(mask_3d, frame_idx, order, valid)
        return mc_mask_4d

@dataclass
class BoundingBox3D:
    """3D Bounding box definition"""
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min)
    
    @property
    def center(self) -> Tuple[float, float, float]:
        return ((self.x_min + self.x_max) / 2,
                (self.y_min + self.y_max) / 2,
                (self.z_min + self.z_max) / 2)
    
    def expand(self, margin: Tuple[int, int, int],
               bounds: Optional[Tuple[int, int, int]] = None) -> 'BoundingBox3D':
        """Expand by margin, clamping to [0, bounds) on each axis if bounds is given.

        bounds is the image/volume shape (X, Y, Z); without it, the box can
        expand past the far edge of the image, letting the search region
        drift outside the frame.
        """
        x_min = max(0, self.x_min - margin[0])
        y_min = max(0, self.y_min - margin[1])
        z_min = max(0, self.z_min - margin[2])
        x_max = self.x_max + margin[0]
        y_max = self.y_max + margin[1]
        z_max = self.z_max + margin[2]
        if bounds is not None:
            x_max = min(bounds[0], x_max)
            y_max = min(bounds[1], y_max)
            z_max = min(bounds[2], z_max)
        return BoundingBox3D(x_min, x_max, y_min, y_max, z_min, z_max)

    def grow(self, margin: Tuple[int, int, int]) -> 'BoundingBox3D':
        """Expand by margin on each axis with no clamping; may extend past the
        image. Pair with extract_padded."""
        return BoundingBox3D(
            self.x_min - margin[0], self.x_max + margin[0],
            self.y_min - margin[1], self.y_max + margin[1],
            self.z_min - margin[2], self.z_max + margin[2])

    def extract_from_volume(self, volume: np.ndarray) -> np.ndarray:
        return volume[self.x_min:self.x_max, self.y_min:self.y_max, self.z_min:self.z_max]
    
    def extract_padded(self, volume: np.ndarray, valid: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract this box even where it runs past the volume.

        Returns (region, mask): out-of-volume voxels are 0 in region and False
        in mask; in-volume voxels take their value from `valid`.
        """
        region = np.zeros(self.shape, dtype=volume.dtype)
        mask = np.zeros(self.shape, dtype=bool)
        lo = (self.x_min, self.y_min, self.z_min)
        hi = (self.x_max, self.y_max, self.z_max)
        src = tuple(slice(max(l, 0), min(h, n)) for l, h, n in zip(lo, hi, volume.shape))
        if all(sl.stop > sl.start for sl in src):
            dst = tuple(slice(sl.start - l, sl.stop - l) for sl, l in zip(src, lo))
            region[dst] = volume[src]
            mask[dst] = valid[src]
        return region, mask

    def translate(self, dx: int, dy: int, dz: int) -> 'BoundingBox3D':
        return BoundingBox3D(
            self.x_min + dx, self.x_max + dx,
            self.y_min + dy, self.y_max + dy,
            self.z_min + dz, self.z_max + dz)
    
    @classmethod
    def from_mask(cls, mask: np.ndarray, padding: int = 0) -> 'BoundingBox3D':
        nonzero = np.argwhere(mask > 0)
        if len(nonzero) == 0:
            raise ValueError("Mask contains no non-zero voxels")
        
        x_min = max(0, nonzero[:, 0].min() - padding)
        x_max = min(mask.shape[0], nonzero[:, 0].max() + 1 + padding)
        y_min = max(0, nonzero[:, 1].min() - padding)
        y_max = min(mask.shape[1], nonzero[:, 1].max() + 1 + padding)
        z_min = max(0, nonzero[:, 2].min() - padding)
        z_max = min(mask.shape[2], nonzero[:, 2].max() + 1 + padding)
        
        return cls(x_min, x_max, y_min, y_max, z_min, z_max)


class MotionCompensation3D:
    """
    3D Motion Compensation using ILSA tracking or Reference-only tracking
    """
    
    def __init__(
        self,
        search_margin: Tuple[int, int, int] = (5, 5, 5),
        use_reference_only: bool = False,  # ← NEW PARAMETER
        use_gpu: bool = True
    ):
        """
        Args:
            search_margin: Per-axis (X, Y, Z) search margin in voxels. The search
                      window is centered on the previous frame's tracked position,
                      so this is the largest per-frame displacement that can be
                      found on each axis; motion beyond it pins the correlation
                      peak against the window edge and the track stalls. Size the
                      elevational (Z) entry from expected out-of-plane motion --
                      it is independent of the other two.
            use_reference_only: If True, always track from reference frame only
                               If False, use ILSA (compare reference vs previous)
            use_gpu: Run the 3D correlation on the GPU via CuPy. Ignored (with
                     a warning) when CuPy or a CUDA device isn't available.
        """
        self.search_margin = tuple(int(m) for m in search_margin)
        self.use_reference_only = use_reference_only
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if use_gpu and not GPU_AVAILABLE:
            print("Warning: use_gpu requested but CuPy/CUDA unavailable, "
                  "falling back to CPU correlation")

    def compute_masked_ncc(
        self,
        search_region: np.ndarray,
        search_mask: np.ndarray,
        template: np.ndarray,
        template_mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Masked NCC map of template over search_region and its peak value."""
        xp = cp if self.use_gpu else np
        correlation_map = masked_ncc(search_region, search_mask, template, template_mask, xp)
        return correlation_map, float(correlation_map.max())

    def find_optimal_translation(
        self,
        correlation_map: np.ndarray,
        search_bbox: BoundingBox3D,
        reference_bbox: BoundingBox3D
    ) -> Tuple[int, int, int]:
        """Find optimal translation from correlation map"""
        max_idx = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
        
        dx = search_bbox.x_min + max_idx[0] - reference_bbox.x_min
        dy = search_bbox.y_min + max_idx[1] - reference_bbox.y_min
        dz = search_bbox.z_min + max_idx[2] - reference_bbox.z_min
        
        return dx, dy, dz
    
    def track_motion_ilsa_3d(
        self,
        volumes: np.ndarray,
        reference_frame_idx: int,
        reference_bbox: BoundingBox3D,
        verbose: bool = True
    ) -> Tuple[List[BoundingBox3D], List[float]]:
        """
        ILSA tracking with optional reference-only mode

        Args:
            volumes: All volume frames (X, Y, Z, T), lateral, depth, elevational, time
            reference_frame_idx: Index of reference frame
            reference_bbox: Bounding box around lesion in reference frame
            verbose: Print progress/summary output

        Returns:
            tracked_bboxes: List of bounding boxes for each frame
            correlations: List of correlation values
        """
        n_frames = volumes.shape[-1]
        ref_img = volumes[..., reference_frame_idx]

        # Imaged sector; voxels outside it (or outside the volume) never count
        # toward the NCC
        sector = compute_sector_mask(volumes, reference_frame_idx)
        ref_voi, ref_mask = reference_bbox.extract_padded(ref_img, sector)

        search_margin = self.search_margin

        tracked_bboxes = [None] * n_frames
        correlations = [0.0] * n_frames
        tracking_sources = [''] * n_frames

        # Set reference frame
        tracked_bboxes[reference_frame_idx] = reference_bbox
        correlations[reference_frame_idx] = 1.0
        tracking_sources[reference_frame_idx] = 'reference'

        # Print tracking mode
        if verbose:
            if self.use_reference_only:
                print("\n=== Reference-Only Tracking Mode ===")
                print("All frames will be tracked from reference frame only")
            else:
                print("\n=== ILSA Tracking Mode ===")
                print("Comparing reference frame vs previous frame")

        # === FORWARD TRACKING ===
        forward_frames = tqdm(range(reference_frame_idx + 1, n_frames), desc="Tracking forward",
                               unit="frame", disable=not verbose)

        for frame_idx in forward_frames:
            prev_bbox = tracked_bboxes[frame_idx - 1]
            # Not clamped to the volume: masked NCC scores a template partly
            # outside the image on the part still inside
            search_bbox = prev_bbox.grow(search_margin)
            search_region, search_mask = search_bbox.extract_padded(
                volumes[..., frame_idx], sector)

            corr_map_ref, max_corr_ref = self.compute_masked_ncc(
                search_region, search_mask, ref_voi, ref_mask
            )

            if self.use_reference_only:
                dx, dy, dz = self.find_optimal_translation(
                    corr_map_ref, search_bbox, reference_bbox
                )
                tracked_bboxes[frame_idx] = reference_bbox.translate(dx, dy, dz)
                correlations[frame_idx] = max_corr_ref
                tracking_sources[frame_idx] = 'reference_only'
            else:
                prev_voi, prev_mask = prev_bbox.extract_padded(
                    volumes[..., frame_idx - 1], sector)
                corr_map_prev, max_corr_prev = self.compute_masked_ncc(
                    search_region, search_mask, prev_voi, prev_mask
                )

                if max_corr_ref >= max_corr_prev:
                    dx, dy, dz = self.find_optimal_translation(
                        corr_map_ref, search_bbox, reference_bbox
                    )
                    tracked_bboxes[frame_idx] = reference_bbox.translate(dx, dy, dz)
                    correlations[frame_idx] = max_corr_ref
                    tracking_sources[frame_idx] = 'reference'
                else:
                    dx, dy, dz = self.find_optimal_translation(
                        corr_map_prev, search_bbox, prev_bbox
                    )
                    tracked_bboxes[frame_idx] = prev_bbox.translate(dx, dy, dz)
                    correlations[frame_idx] = max_corr_prev
                    tracking_sources[frame_idx] = 'previous'

        # === BACKWARD TRACKING ===
        backward_frames = tqdm(range(reference_frame_idx - 1, -1, -1), desc="Tracking backward",
                                unit="frame", disable=not verbose)

        for frame_idx in backward_frames:
            next_bbox = tracked_bboxes[frame_idx + 1]
            # Not clamped to the volume: masked NCC scores a template partly
            # outside the image on the part still inside
            search_bbox = next_bbox.grow(search_margin)
            search_region, search_mask = search_bbox.extract_padded(
                volumes[..., frame_idx], sector)

            corr_map_ref, max_corr_ref = self.compute_masked_ncc(
                search_region, search_mask, ref_voi, ref_mask
            )

            if self.use_reference_only:
                dx, dy, dz = self.find_optimal_translation(
                    corr_map_ref, search_bbox, reference_bbox
                )
                tracked_bboxes[frame_idx] = reference_bbox.translate(dx, dy, dz)
                correlations[frame_idx] = max_corr_ref
                tracking_sources[frame_idx] = 'reference_only'
            else:
                next_voi, next_mask = next_bbox.extract_padded(
                    volumes[..., frame_idx + 1], sector)
                corr_map_next, max_corr_next = self.compute_masked_ncc(
                    search_region, search_mask, next_voi, next_mask
                )

                if max_corr_ref >= max_corr_next:
                    dx, dy, dz = self.find_optimal_translation(
                        corr_map_ref, search_bbox, reference_bbox
                    )
                    tracked_bboxes[frame_idx] = reference_bbox.translate(dx, dy, dz)
                    correlations[frame_idx] = max_corr_ref
                    tracking_sources[frame_idx] = 'reference'
                else:
                    dx, dy, dz = self.find_optimal_translation(
                        corr_map_next, search_bbox, next_bbox
                    )
                    tracked_bboxes[frame_idx] = next_bbox.translate(dx, dy, dz)
                    correlations[frame_idx] = max_corr_next
                    tracking_sources[frame_idx] = 'previous'

        # === PRINT STATS ===
        if verbose:
            source_counts = Counter(tracking_sources)
            print(f"\n=== Tracking Complete ===")
            print(f"Sources: {dict(source_counts)}")
            print(f"Mean correlation: {np.mean(correlations):.3f}")
            print(f"Min correlation: {np.min(correlations):.3f}")

        return tracked_bboxes, correlations
