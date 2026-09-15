"""
Coronal/sagittal-guided adaptive-bbox 3D mask reconstruction.

Used wherever a full (X, Y, Z) MedSAM2 mask is needed for one CEUS frame —
the napari 4D viewer and the TIC analysis both need this, so the
reconstruction logic lives here once instead of being copy-pasted across
notebook cells. Also usable directly as an interactive Frame/Slice viewer
(Medsam2SliceViewerBase, medsam2_viewer_base.py), same as the other two
MedSAM2 approaches.

Pipeline per frame:
  1) Segment the coronal slice at y_mid and the sagittal slice at x_mid using
     SAM2's 2D image predictor with the tracked 3D bbox as the box prompt.
  2) For every axial slice z in [bbox.z_min, bbox.z_max), derive a tight
     "adaptive" bbox from those two masks' active extent at that z, then run
     the 2D image predictor on that axial slice with the adaptive bbox. A
     round tumor's cross-section shrinks near the top/bottom of the tracked
     volume, so this adaptive box shrinks there too, rather than staying one
     fixed size through the whole stack.
  3) Stack the per-slice axial predictions into a 3D mask.

Like Medsam2IndependentPlaneViewer (medsam2_video_viewer.py), every one of
those calls -- coronal, sagittal, and every axial slice -- is still an
independent 2D image-predictor call with no memory linking one slice to the
next; only Medsam2AxialPropagationViewer (medsam2_axial_propagation.py) has
real cross-slice memory.

The interactive viewer here supports all three planes, but only Axial is
computed directly (step 2/3 above) -- Coronal/Sagittal are a different-axis
re-slice of the already-built (X, Y, Z) mask_3d, not a new independent SAM2
call on those planes. The only real inference on coronal/sagittal is the
single guide slice at y_mid/x_mid in step 1, which isn't itself what's
displayed when you pick those planes here.
"""

import time

import numpy as np

from medsam2_viewer_base import (
    Medsam2SliceViewerBase, run_medsam2_2d, dice_score, get_plane_slice, smooth_3d_mask,
)  # reuse, don't redefine


def synthetic_coronal_mask(axial_mask_yx, sagittal_mask_yz, volume_shape, inflate=1.2):
    """
    An (X, Z) stand-in for the coronal guide, built without segmenting any
    coronal slice.

    The coronal guide's only job in `get_adaptive_bbox_at_z` is to supply the
    X extent at each z. That function does not have to come from a
    segmentation -- and in this data it arguably should not, because the beam
    transmits only ~20-25 real elevation planes and the ~219 z voxels are
    mostly interpolation. A coronal slice (X, Z) is therefore largely
    interpolated along one of its two axes, while the axial plane (X, Y) is
    sampled in both.

    So X(z) is modelled instead, from two measurements taken in planes that
    are actually sampled:

      - the X extent of the **centre axial** segmentation, setting the
        half-width at the lesion's widest point;
      - the z extent of the **sagittal** segmentation, setting where the
        lesion begins and ends along z.

    Those become the semi-axes of an ellipse in (X, Z), so the X extent tapers
    smoothly to the poles instead of collapsing into a few noisy pixels the
    way a real coronal guide does at the ends of the lesion -- which is the
    regime where the reconstruction was observed to break down.

    `inflate` scales the X semi-axis, and the asymmetry behind the 1.2 default
    matters: this is a *prompt*, not a hard constraint, so a box that is too
    wide costs almost nothing (SAM2 segments the smaller object inside it)
    while a box that is too narrow can clip the lesion irrecoverably.

    Measured against this cohort's ten reference VOIs -- asking whether the
    synthetic box, padding included, actually *contains* the true X extent at
    each z -- the failure rate is 15.6% of slices un-inflated, 6.4% at 1.2x,
    3.2% at 1.4x and 2.1% at 1.6x.

    Those residual failures are mostly not a width problem. Decomposed at
    1.2x, only 1.4% of slices miss on both sides (genuinely too narrow) while
    5.1% miss on one side only -- the ellipse is centred at the centre-axial
    slice's x-centre, and lesions that drift in X as z changes leave it
    offset. Raising `inflate` treats that by brute force; the targeted fix
    would be to segment two or three axial slices instead of one and let
    x_centre(z) follow a fitted line, which also stays entirely within the
    axial plane. Worth doing only if the residual turns out to matter, since
    SAM2's own per-slice error may well dominate it.

    Returns an (X, Z) bool array, or None if either input mask is empty.
    """
    x_active = np.where(axial_mask_yx.any(axis=0))[0]       # columns index X
    z_active = np.where(sagittal_mask_yz.any(axis=0))[0]    # columns index Z
    if len(x_active) == 0 or len(z_active) == 0:
        return None

    x_lo, x_hi = int(x_active.min()), int(x_active.max())
    z_lo, z_hi = int(z_active.min()), int(z_active.max())
    x_centre, z_centre = (x_lo + x_hi) / 2.0, (z_lo + z_hi) / 2.0
    a_x = inflate * (x_hi - x_lo + 1) / 2.0
    a_z = max((z_hi - z_lo + 1) / 2.0, 1e-6)

    n_x, _, n_z = volume_shape
    coronal = np.zeros((n_x, n_z), dtype=bool)
    for z in range(max(0, z_lo), min(n_z, z_hi + 1)):
        t = 1.0 - ((z - z_centre) / a_z) ** 2
        # Never let a column go empty: get_adaptive_bbox_at_z returns None for
        # a column with no active pixels and skips that axial slice entirely.
        half = a_x * np.sqrt(t) if t > 0 else 0.5
        lo = max(0, int(np.floor(x_centre - half)))
        hi = min(n_x, int(np.ceil(x_centre + half)) + 1)
        coronal[lo:hi, z] = True
    return coronal


def get_adaptive_bbox_at_z(coronal_mask, sagittal_mask, z_abs, volume_shape, padding=4):
    """
    Derive a tight 2D bbox for the axial slice at z_abs from the coronal and
    sagittal masks' active extent at that z.

    coronal_mask  : (X, Z) bool — from SAM2 on volume[:, y_mid, :]
    sagittal_mask : (Y, Z) bool — from SAM2 on volume[x_mid, :, :]
    volume_shape  : (X, Y, Z) — used to clamp the bbox to volume bounds

    Returns: [x_min, y_min, x_max, y_max] or None if no active pixels at this z.
    """
    coronal_col = coronal_mask[:, z_abs]
    x_active = np.where(coronal_col)[0]

    sagittal_col = sagittal_mask[:, z_abs]
    y_active = np.where(sagittal_col)[0]

    if len(x_active) == 0 or len(y_active) == 0:
        return None

    x_min = max(0, x_active.min() - padding)
    x_max = min(volume_shape[0], x_active.max() + padding)
    y_min = max(0, y_active.min() - padding)
    y_max = min(volume_shape[1], y_active.max() + padding)

    return [x_min, y_min, x_max, y_max]


def compute_3d_mask(image_predictor, volume, bbox, y_mid, x_mid, padding=4,
                    z_mid=None, synthetic_coronal=False, inflate=1.2):
    """
    Reconstruct the full 3D mask by:
      1) segmenting two guide slices with the tracked 3D bbox as the prompt,
      2) deriving a per-z adaptive bbox from them,
      3) segmenting each axial slice in [bbox.z_min, bbox.z_max) with its adaptive bbox.

    Only step 1 differs between the two modes; steps 2 and 3 -- and therefore
    the fact that the 3D volume is built purely out of axial segmentations --
    are identical either way:

      synthetic_coronal=False (default)
          coronal at y_mid  -> X extent per z
          sagittal at x_mid -> Y extent per z

      synthetic_coronal=True
          axial at z_mid    -> X extent at the lesion's widest point
          sagittal at x_mid -> Y extent per z, and the lesion's z extent
          the two are combined into an elliptical stand-in coronal mask by
          `synthetic_coronal_mask`, so no coronal slice is ever segmented

    Exactly two guide segmentations run per frame in both modes.

    Returns: mask_3d (X, Y, Z) uint8, same shape as `volume`.
    """
    z_min, z_max = int(bbox.z_min), int(bbox.z_max)
    mask_3d = np.zeros(volume.shape, dtype=np.uint8)           # (X, Y, Z)

    # Sagittal first: the synthetic path needs its z extent to build the ellipse.
    sagittal_slice = volume[x_mid, :, :]                       # (Y, Z)
    sagittal_bbox  = [bbox.z_min, bbox.y_min, bbox.z_max, bbox.y_max]
    sagittal_mask  = run_medsam2_2d(image_predictor, sagittal_slice, sagittal_bbox)
    if sagittal_mask is None:
        return mask_3d

    if synthetic_coronal:
        z_centre = int(z_mid if z_mid is not None else (z_min + z_max) // 2)
        z_centre = int(np.clip(z_centre, z_min, max(z_min, z_max - 1)))
        axial_bbox = [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
        axial_mask = run_medsam2_2d(image_predictor, volume[:, :, z_centre].T, axial_bbox)
        if axial_mask is None:
            return mask_3d
        coronal_mask = synthetic_coronal_mask(
            axial_mask, sagittal_mask, volume.shape, inflate=inflate
        )
    else:
        coronal_slice = volume[:, y_mid, :]                    # (X, Z)
        coronal_bbox  = [bbox.z_min, bbox.x_min, bbox.z_max, bbox.x_max]
        coronal_mask  = run_medsam2_2d(image_predictor, coronal_slice, coronal_bbox)

    if coronal_mask is None:
        return mask_3d

    for z in range(z_min, z_max):
        adaptive_bbox = get_adaptive_bbox_at_z(coronal_mask, sagittal_mask, z, volume.shape, padding=padding)
        if adaptive_bbox is None:
            continue

        axial_slice = volume[:, :, z].T                        # (Y, X)
        pred_yx = run_medsam2_2d(image_predictor, axial_slice, adaptive_bbox)
        if pred_yx is not None:
            mask_3d[:, :, z] = pred_yx.T                        # (Y,X) -> (X,Y)

    return mask_3d


class Medsam2AdaptiveBboxMasker(Medsam2SliceViewerBase):
    """
    Builds one SAM2 image predictor and lazily computes + caches the full
    3D adaptive-bbox mask per CEUS frame via compute_frame() -- the API the
    napari 4D viewer and TIC analysis consume directly, unchanged. Also
    usable as its own interactive Frame/Plane/Slice viewer via show(), same
    as the other two approaches, since it's now a Medsam2SliceViewerBase
    subclass -- Coronal/Sagittal here just re-slice the already-computed
    mask_3d along a different axis, not a new SAM2 call on those planes.
    """

    SUPPORTED_PLANES = ("Axial", "Coronal", "Sagittal")

    def __init__(self, seg_data, bmode_image_data, model_cfg, checkpoint, device="cuda",
                 smooth_sigma_mm=1.0, synthetic_coronal=False, inflate=1.2):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        self.image_predictor = SAM2ImagePredictor(sam2_model)
        # 3D Gaussian-smooth the reconstructed mask to remove the axial
        # per-slice staircase seam (see smooth_3d_mask's docstring) --
        # 0/None disables it, for an exact comparison against the raw output.
        self.smooth_sigma_mm = smooth_sigma_mm
        # Skip the coronal segmentation entirely and model X(z) instead --
        # see synthetic_coronal_mask for why that is the better guide here.
        self.synthetic_coronal = synthetic_coronal
        self.inflate = inflate
        super().__init__(seg_data, bmode_image_data, device=device)

    def compute_frame(self, frame_idx):
        """Full (X,Y,Z) adaptive-bbox mask + reference mask + whole-volume
        Dice for one CEUS frame. Unchanged from before this class also
        became a viewer -- napari and TIC call this directly."""
        if frame_idx in self.cache:
            return self.cache[frame_idx]

        geom = self._frame_geometry(frame_idx)
        volume = self.bmode_image_data.pixel_data[:, :, :, frame_idx]        # (X, Y, Z)
        start = time.time()
        mask_3d = compute_3d_mask(
            self.image_predictor, volume, geom["bbox"], geom["y_mid"], geom["x_mid"],
            z_mid=geom["z_mid"], synthetic_coronal=self.synthetic_coronal,
            inflate=self.inflate,
        )
        if self.smooth_sigma_mm:
            mask_3d = smooth_3d_mask(
                mask_3d, spacing_xyz=self.bmode_image_data.pixdim, sigma_mm=self.smooth_sigma_mm
            ).astype(np.uint8)
        elapsed = time.time() - start

        result = {
            "volume": volume,
            "bbox": geom["bbox"],
            "mc_mask": geom["mc_mask"],
            "mask_3d": mask_3d,
            "dice": dice_score(mask_3d, geom["mc_mask"]),
            "elapsed": elapsed,
        }
        self.cache[frame_idx] = result
        return result

    def _slice_range(self, frame_idx, plane):
        """
        Axial is clamped to [bbox.z_min, bbox.z_max) -- the loop range
        compute_3d_mask actually populates; outside it mask_3d is
        definitionally all zero. Coronal/Sagittal are clamped to the bbox's
        Y/X extent for the same reason a wider range wouldn't show anything
        meaningful: every axial slice's adaptive box is itself derived from
        (and padded only slightly past) the bbox-prompted coronal/sagittal
        guide masks, so real mask content isn't expected far outside it.
        """
        geom = self._frame_geometry(frame_idx)
        bbox = geom["bbox"]
        if plane == "Axial":
            return geom["z_min"], geom["z_max"] - 1, geom["z_mid"]
        elif plane == "Coronal":
            return int(bbox.y_min), int(bbox.y_max) - 1, geom["y_mid"]
        else:
            return int(bbox.x_min), int(bbox.x_max) - 1, geom["x_mid"]

    def _get_prediction(self, frame_idx, plane, idx):
        result = self.compute_frame(frame_idx)
        pred = get_plane_slice(result["mask_3d"], plane, idx).astype(bool)
        return {"pred": pred if pred.any() else None, "elapsed": result["elapsed"]}
