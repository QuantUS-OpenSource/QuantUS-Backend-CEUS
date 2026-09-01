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


def compute_3d_mask(image_predictor, volume, bbox, y_mid, x_mid, padding=4):
    """
    Reconstruct the full 3D mask by:
      1) segmenting coronal (at y_mid) and sagittal (at x_mid) with the 3D bbox,
      2) deriving a per-z adaptive bbox from those two masks,
      3) segmenting each axial slice in [bbox.z_min, bbox.z_max) with its adaptive bbox.

    Returns: mask_3d (X, Y, Z) uint8, same shape as `volume`.
    """
    z_min, z_max = int(bbox.z_min), int(bbox.z_max)
    mask_3d = np.zeros(volume.shape, dtype=np.uint8)           # (X, Y, Z)

    coronal_slice = volume[:, y_mid, :]                        # (X, Z)
    coronal_bbox  = [bbox.z_min, bbox.x_min, bbox.z_max, bbox.x_max]
    coronal_mask  = run_medsam2_2d(image_predictor, coronal_slice, coronal_bbox)

    sagittal_slice = volume[x_mid, :, :]                       # (Y, Z)
    sagittal_bbox  = [bbox.z_min, bbox.y_min, bbox.z_max, bbox.y_max]
    sagittal_mask  = run_medsam2_2d(image_predictor, sagittal_slice, sagittal_bbox)

    if coronal_mask is None or sagittal_mask is None:
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

    def __init__(self, seg_data, bmode_image_data, model_cfg, checkpoint, device="cuda", smooth_sigma_mm=1.0):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        self.image_predictor = SAM2ImagePredictor(sam2_model)
        # 3D Gaussian-smooth the reconstructed mask to remove the axial
        # per-slice staircase seam (see smooth_3d_mask's docstring) --
        # 0/None disables it, for an exact comparison against the raw output.
        self.smooth_sigma_mm = smooth_sigma_mm
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
        mask_3d = compute_3d_mask(self.image_predictor, volume, geom["bbox"], geom["y_mid"], geom["x_mid"])
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
