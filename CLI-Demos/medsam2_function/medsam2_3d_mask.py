"""
Coronal/sagittal-guided adaptive-bbox 3D mask reconstruction.

Used wherever a full (X, Y, Z) MedSAM2 mask is needed for one CEUS frame —
the napari 4D viewer and the TIC analysis both need this, so the
reconstruction logic lives here once instead of being copy-pasted across
notebook cells.

Pipeline per frame:
  1) Segment the coronal slice at y_mid and the sagittal slice at x_mid using
     SAM2's 2D image predictor with the tracked 3D bbox as the box prompt.
  2) For every axial slice z in [bbox.z_min, bbox.z_max), derive a tight
     "adaptive" bbox from those two masks' active extent at that z, then run
     the 2D image predictor on that axial slice with the adaptive bbox.
  3) Stack the per-slice axial predictions into a 3D mask.

(This is deliberately separate from medsam2_video_viewer.py, which mirrors
the independent per-plane 2D inference cell exactly and has no notion of a
full 3D mask — a volumetric mask is only needed here, for napari display and
TIC computation.)
"""

import numpy as np

from medsam2_video_viewer import run_medsam2_2d, dice_score  # reuse, don't redefine


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


class Medsam2AdaptiveBboxMasker:
    """Builds one SAM2 image predictor and lazily computes + caches the full
    3D adaptive-bbox mask per CEUS frame. Shared by the napari 4D viewer and
    the TIC analysis so the model is only loaded once per notebook section."""

    def __init__(self, seg_data, bmode_image_data, model_cfg, checkpoint, device="cuda"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.seg_data = seg_data
        self.bmode_image_data = bmode_image_data
        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        self.image_predictor = SAM2ImagePredictor(sam2_model)
        self.cache = {}   # frame_idx -> result dict

    def _frame_geometry(self, frame_idx):
        bbox = self.seg_data.motion_compensation.tracked_bboxes[frame_idx]
        mc_mask = self.seg_data.motion_compensation.apply_to_mask(
            self.seg_data.seg_mask, frame_idx, 0
        )
        return {
            "bbox": bbox,
            "mc_mask": mc_mask,
            "z_mid": int(np.argmax(mc_mask.sum(axis=(0, 1)))),
            "y_mid": int(np.argmax(mc_mask.sum(axis=(0, 2)))),
            "x_mid": int(np.argmax(mc_mask.sum(axis=(1, 2)))),
        }

    def compute_frame(self, frame_idx):
        if frame_idx in self.cache:
            return self.cache[frame_idx]

        geom = self._frame_geometry(frame_idx)
        volume = self.bmode_image_data.pixel_data[:, :, :, frame_idx]        # (X, Y, Z)
        mask_3d = compute_3d_mask(self.image_predictor, volume, geom["bbox"], geom["y_mid"], geom["x_mid"])

        result = {
            "volume": volume,
            "bbox": geom["bbox"],
            "mc_mask": geom["mc_mask"],
            "mask_3d": mask_3d,
            "dice": dice_score(mask_3d, geom["mc_mask"]),
        }
        self.cache[frame_idx] = result
        return result
