"""
3D Motion Compensation for CEUS Analysis
Based on ILSA tracking
Axis convention: (X, Y, Z, T) = (lateral, depth, elevational, time)
"""

import numpy as np
from scipy.ndimage import shift
from scipy.signal import correlate
from typing import Tuple, List, Optional
from dataclasses import dataclass
from collections import Counter
import cv2
from tqdm import tqdm

@dataclass
class MotionCompensationResult:
    """Store motion compensation results efficiently"""
    translation_vectors: np.ndarray
    reference_frame: int
    correlations: np.ndarray
    reference_bbox: 'BoundingBox3D'
    tracked_bboxes: List['BoundingBox3D']
    
    def get_translation(self, frame_idx: int) -> Tuple[float, float, float]:
        return tuple(self.translation_vectors[frame_idx])
    
    def apply_to_mask(self, mask_3d: np.ndarray, frame_idx: int, order: int = 0) -> np.ndarray:
        dx, dy, dz = self.get_translation(frame_idx)
        return shift(mask_3d, shift=[dx, dy, dz], order=order, cval=0,
                    prefilter=True if order > 0 else False)
    
    def apply_to_all_frames(self, mask_3d: np.ndarray, order: int = 0) -> np.ndarray:
        n_frames = len(self.translation_vectors)
        mc_mask_4d = np.zeros((*mask_3d.shape, n_frames), dtype=mask_3d.dtype)
        for frame_idx in range(n_frames):
            mc_mask_4d[..., frame_idx] = self.apply_to_mask(mask_3d, frame_idx, order)
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

    def touches_bounds(self, bounds: Tuple[int, int, int]) -> bool:
        """True if this box is at or beyond the image edge on any axis."""
        return (self.x_min <= 0 or self.x_max >= bounds[0] or
                self.y_min <= 0 or self.y_max >= bounds[1] or
                self.z_min <= 0 or self.z_max >= bounds[2])

    def clip(self, bounds: Tuple[int, int, int]) -> 'BoundingBox3D':
        """Clamp this box to stay within [0, bounds) on every axis."""
        return BoundingBox3D(
            max(0, min(self.x_min, bounds[0])), max(0, min(self.x_max, bounds[0])),
            max(0, min(self.y_min, bounds[1])), max(0, min(self.y_max, bounds[1])),
            max(0, min(self.z_min, bounds[2])), max(0, min(self.z_max, bounds[2])))
    
    def extract_from_volume(self, volume: np.ndarray) -> np.ndarray:
        return volume[self.x_min:self.x_max, self.y_min:self.y_max, self.z_min:self.z_max]
    
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
        search_margin_ratio: float = 0.5 / 30,
        use_reference_only: bool = False,  # ← NEW PARAMETER
        n_splits: Tuple[int, int, int] = (3, 3, 2)
    ):
        """
        Args:
            search_margin_ratio: Search margin as ratio of volume dimensions
            use_reference_only: If True, always track from reference frame only
                               If False, use ILSA (compare reference vs previous)
            n_splits: Default (X, Y, Z) grid used by track_motion_blockwise_3d
                      to split a large ROI into smaller, independently
                      trackable sub-blocks
        """
        self.search_margin_ratio = search_margin_ratio
        self.use_reference_only = use_reference_only
        self.n_splits = n_splits

    def compute_search_margin(self, volume_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(int(self.search_margin_ratio * x) for x in volume_shape)

    def split_into_blocks(
        self,
        bbox: BoundingBox3D,
        n_splits: Tuple[int, int, int]
    ) -> List[BoundingBox3D]:
        """Tile bbox into an n_splits[0] x n_splits[1] x n_splits[2] grid of
        smaller, non-overlapping sub-boxes covering it exactly."""
        edges = [
            np.linspace(mn, mx, n + 1)
            for mn, mx, n in zip(
                (bbox.x_min, bbox.y_min, bbox.z_min),
                (bbox.x_max, bbox.y_max, bbox.z_max),
                n_splits
            )
        ]
        blocks = []
        for i in range(n_splits[0]):
            x0, x1 = int(round(edges[0][i])), int(round(edges[0][i + 1]))
            for j in range(n_splits[1]):
                y0, y1 = int(round(edges[1][j])), int(round(edges[1][j + 1]))
                for k in range(n_splits[2]):
                    z0, z1 = int(round(edges[2][k])), int(round(edges[2][k + 1]))
                    if x1 > x0 and y1 > y0 and z1 > z0:
                        blocks.append(BoundingBox3D(x0, x1, y0, y1, z0, z1))
        return blocks
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        mean = np.mean(volume)
        std = np.std(volume)
        if std == 0:
            return volume - mean
        return (volume - mean) / std
       
    def compute_3d_correlation_vectorized(
        self,
        volumes: np.ndarray,
        reference_voi: np.ndarray,
        search_bbox: BoundingBox3D
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized 3D correlation computation"""
        n_frames = volumes.shape[-1]
        ref_shape = reference_voi.shape
        ref_normalized = self.normalize_volume(reference_voi)
        
        search_shape = search_bbox.shape
        corr_shape = tuple(s - r + 1 for s, r in zip(search_shape, ref_shape))
        correlation_map = np.zeros((n_frames, *corr_shape))
        
        search_region = search_bbox.extract_from_volume(volumes[..., 0])
        search_normalized = self.normalize_volume(search_region)
        
        correlation = correlate(search_normalized, ref_normalized, mode='valid', method='fft')
        
        ref_sum_sq = np.sum(ref_normalized ** 2)
        search_sq = search_normalized ** 2
        kernel = np.ones(ref_shape)
        local_sum_sq = correlate(search_sq, kernel, mode='valid', method='fft')
        
        denominator = np.sqrt(ref_sum_sq * local_sum_sq)
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        correlation_map = correlation / denominator
        max_correlations = np.max(correlation_map.reshape(n_frames, -1), axis=1)
        
        return correlation_map, max_correlations
    
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
            verbose: Print progress/summary output (set False for per-block
                     sub-calls from track_motion_blockwise_3d)

        Returns:
            tracked_bboxes: List of bounding boxes for each frame
            correlations: List of correlation values
        """
        n_frames = volumes.shape[-1]
        ref_img = volumes[..., reference_frame_idx]
        volume_shape = volumes.shape[:-1]

        # Apply image enhancement
        ref_voi = reference_bbox.extract_from_volume(ref_img)

        search_margin = self.compute_search_margin(volume_shape)

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
            search_bbox = prev_bbox.expand(search_margin, volume_shape)

            image_analysis = volumes[..., frame_idx]
            final_image = image_analysis[..., np.newaxis]

            corr_map_ref, max_corr_ref = self.compute_3d_correlation_vectorized(
                final_image, ref_voi, search_bbox
            )

            if self.use_reference_only:
                dx, dy, dz = self.find_optimal_translation(
                    corr_map_ref, search_bbox, reference_bbox
                )
                tracked_bboxes[frame_idx] = reference_bbox.translate(dx, dy, dz)
                correlations[frame_idx] = max_corr_ref[0]
                tracking_sources[frame_idx] = 'reference_only'
            else:
                prev_voi = prev_bbox.extract_from_volume(volumes[..., frame_idx - 1])
                corr_map_prev, max_corr_prev = self.compute_3d_correlation_vectorized(
                    final_image, prev_voi, search_bbox
                )

                if max_corr_ref[0] >= max_corr_prev[0]:
                    dx, dy, dz = self.find_optimal_translation(
                        corr_map_ref, search_bbox, reference_bbox
                    )
                    tracked_bboxes[frame_idx] = reference_bbox.translate(dx, dy, dz)
                    correlations[frame_idx] = max_corr_ref[0]
                    tracking_sources[frame_idx] = 'reference'
                else:
                    dx, dy, dz = self.find_optimal_translation(
                        corr_map_prev, search_bbox, prev_bbox
                    )
                    tracked_bboxes[frame_idx] = prev_bbox.translate(dx, dy, dz)
                    correlations[frame_idx] = max_corr_prev[0]
                    tracking_sources[frame_idx] = 'previous'

        # === BACKWARD TRACKING ===
        backward_frames = tqdm(range(reference_frame_idx - 1, -1, -1), desc="Tracking backward",
                                unit="frame", disable=not verbose)

        for frame_idx in backward_frames:
            next_bbox = tracked_bboxes[frame_idx + 1]
            search_bbox = next_bbox.expand(search_margin, volume_shape)

            image_analysis = volumes[..., frame_idx]
            final_image = image_analysis[..., np.newaxis]

            corr_map_ref, max_corr_ref = self.compute_3d_correlation_vectorized(
                final_image, ref_voi, search_bbox
            )

            if self.use_reference_only:
                dx, dy, dz = self.find_optimal_translation(
                    corr_map_ref, search_bbox, reference_bbox
                )
                tracked_bboxes[frame_idx] = reference_bbox.translate(dx, dy, dz)
                correlations[frame_idx] = max_corr_ref[0]
                tracking_sources[frame_idx] = 'reference_only'
            else:
                next_voi = next_bbox.extract_from_volume(volumes[..., frame_idx + 1])
                corr_map_next, max_corr_next = self.compute_3d_correlation_vectorized(
                    final_image, next_voi, search_bbox
                )

                if max_corr_ref[0] >= max_corr_next[0]:
                    dx, dy, dz = self.find_optimal_translation(
                        corr_map_ref, search_bbox, reference_bbox
                    )
                    tracked_bboxes[frame_idx] = reference_bbox.translate(dx, dy, dz)
                    correlations[frame_idx] = max_corr_ref[0]
                    tracking_sources[frame_idx] = 'reference'
                else:
                    dx, dy, dz = self.find_optimal_translation(
                        corr_map_next, search_bbox, next_bbox
                    )
                    tracked_bboxes[frame_idx] = next_bbox.translate(dx, dy, dz)
                    correlations[frame_idx] = max_corr_next[0]
                    tracking_sources[frame_idx] = 'previous'

        # === PRINT STATS ===
        if verbose:
            source_counts = Counter(tracking_sources)
            print(f"\n=== Tracking Complete ===")
            print(f"Sources: {dict(source_counts)}")
            print(f"Mean correlation: {np.mean(correlations):.3f}")
            print(f"Min correlation: {np.min(correlations):.3f}")

        return tracked_bboxes, correlations

    def track_motion_blockwise_3d(
        self,
        volumes: np.ndarray,
        reference_frame_idx: int,
        reference_bbox: BoundingBox3D,
        n_splits: Optional[Tuple[int, int, int]] = None,
        min_correlation: float = 0.5
    ) -> Tuple[List[BoundingBox3D], List[float]]:
        """
        Track motion by tiling the ROI into a grid of smaller sub-blocks,
        tracking each independently, and averaging the reliable blocks'
        translations into a single rigid shift for the whole ROI.

        A single large ROI can leave no room to search once the search
        window is kept inside the image (see BoundingBox3D.expand), and
        out-of-plane motion can make part of the ROI's content vanish from
        the frame entirely. Smaller sub-blocks are individually small enough
        to have search room.

        Reliability is judged per frame, not per block: at each frame, a
        block only votes if its track hasn't reached the image edge *at
        that frame* and its correlation *at that frame* clears
        min_correlation. A block that drifts out of bounds or loses its
        feature for part of the sequence still contributes wherever it's
        reliable, instead of being thrown out for the whole run over one
        bad frame. If a frame ends up with no reliable block at all, its
        shift is carried over from the nearest already-resolved frame,
        working outward from the reference frame (same direction as the
        ILSA forward/backward passes).

        Returns:
            tracked_bboxes: reference_bbox translated by the per-frame
                            averaged shift, clipped to the image bounds
            correlations: mean correlation of the contributing blocks per
                          frame (0.0 for a frame that had to carry over)
        """
        volume_shape = volumes.shape[:-1]
        n_frames = volumes.shape[-1]
        n_splits = n_splits or self.n_splits

        blocks = self.split_into_blocks(reference_bbox, n_splits)
        if not blocks:
            blocks = [reference_bbox]

        print(f"\n=== Block-wise Tracking ({len(blocks)} blocks, n_splits={n_splits}) ===")

        block_results = []
        for block in tqdm(blocks, desc="Tracking blocks", unit="block"):
            tracked, corrs = self.track_motion_ilsa_3d(
                volumes, reference_frame_idx, block, verbose=False
            )
            block_results.append((block, tracked, corrs))

        deltas_per_frame: List[Optional[Tuple[int, int, int]]] = [None] * n_frames
        corr_per_frame = [0.0] * n_frames
        n_valid_per_frame = [0] * n_frames

        for frame_idx in range(n_frames):
            deltas, weights = [], []
            for block, tracked, corrs in block_results:
                bbox = tracked[frame_idx]
                if bbox.touches_bounds(volume_shape) or corrs[frame_idx] < min_correlation:
                    continue
                deltas.append((
                    bbox.x_min - block.x_min,
                    bbox.y_min - block.y_min,
                    bbox.z_min - block.z_min
                ))
                weights.append(max(corrs[frame_idx], 1e-6))

            n_valid_per_frame[frame_idx] = len(deltas)
            if deltas:
                avg = np.average(np.array(deltas, dtype=float), axis=0, weights=np.array(weights))
                deltas_per_frame[frame_idx] = tuple(int(round(v)) for v in avg)
                corr_per_frame[frame_idx] = float(np.mean(weights))

        # The reference frame is always a 0-shift by definition even if every
        # block happened to be excluded there (e.g. a block sitting on the
        # image edge from the start).
        if deltas_per_frame[reference_frame_idx] is None:
            deltas_per_frame[reference_frame_idx] = (0, 0, 0)

        # Carry frames with no reliable block over from the nearest resolved
        # neighbor, walking outward from the reference frame.
        n_fallback = 0
        last = deltas_per_frame[reference_frame_idx]
        for frame_idx in range(reference_frame_idx + 1, n_frames):
            if deltas_per_frame[frame_idx] is None:
                deltas_per_frame[frame_idx] = last
                n_fallback += 1
            else:
                last = deltas_per_frame[frame_idx]
        last = deltas_per_frame[reference_frame_idx]
        for frame_idx in range(reference_frame_idx - 1, -1, -1):
            if deltas_per_frame[frame_idx] is None:
                deltas_per_frame[frame_idx] = last
                n_fallback += 1
            else:
                last = deltas_per_frame[frame_idx]

        print(f"Average {np.mean(n_valid_per_frame):.1f}/{len(blocks)} blocks contributed per "
              f"frame (edge-touching or low-correlation blocks excluded per-frame, not globally)")
        if n_fallback:
            print(f"{n_fallback}/{n_frames} frames had no reliable block and used the nearest "
                  f"resolved frame's shift instead")

        tracked_bboxes = [None] * n_frames
        correlations = [0.0] * n_frames
        for frame_idx in range(n_frames):
            dx, dy, dz = deltas_per_frame[frame_idx]
            tracked_bboxes[frame_idx] = reference_bbox.translate(dx, dy, dz).clip(volume_shape)
            correlations[frame_idx] = corr_per_frame[frame_idx]

        print(f"\n=== Tracking Complete ===")
        print(f"Mean correlation: {np.mean(correlations):.3f}")
        print(f"Min correlation: {np.min(correlations):.3f}")

        return tracked_bboxes, correlations