"""
Genuine SAM2 video-object propagation through the axial (Z) slice stack —
as opposed to Medsam2IndependentPlaneViewer's independent per-slice 2D calls
(medsam2_video_viewer.py) or Medsam2AdaptiveBboxMasker's per-z adaptive-bbox
independent calls (medsam2_3d_mask.py).

The axial Z-stack of one CEUS frame's B-mode volume is treated as a video:
a single box prompt is placed on the slice with the most tumor overlap
(the tracked bbox's "key slice"), and SAM2's memory-attention mechanism
propagates the mask forward and backward from there through the rest of the
stack (mirroring MedSAM2's own reference recipe for 3D CT lesion
segmentation in medsam2_infer_3D_CT.py: init_state -> add_new_points_or_box
-> propagate_in_video, then reset_state -> add_new_points_or_box ->
propagate_in_video(reverse=True), unioning both passes).

This is a meaningfully different algorithm from the other two approaches,
not just a variant of the same idea: because propagation carries an
object-presence/occlusion score across frames, it can correctly predict
"no object here" on slices where the tumor isn't visible, rather than
either being forced to produce some mask inside a fixed box regardless of
whether the object is actually present (independent-plane viewer) or
depending entirely on two single 2D guide-slice segmentations to derive a
shrinking box (adaptive-bbox masker).

Only the axial direction is propagated -- coronal/sagittal don't have a
well-defined "video order" the model was ever trained to propagate along in
that orientation. But the propagated result covers the whole (Z, Y, X)
volume, so the viewer can still display Coronal/Sagittal cross-sections of
it (a different-axis re-slice of the same result, not new inference).

The propagated mask is also hard-clipped to the tracked 3D bbox
(clip_mask_to_bbox) before display: SAM2's box prompt only conditions
propagation, it doesn't hard-constrain the output pixel-for-pixel, so a
small amount of leakage past the tracked VOI is possible even though
propagation is far better-behaved than independent per-slice calls (that's
what made the coronal/sagittal views unreliable in the other two
approaches). Clipping guarantees the prediction never exceeds the VOI's
actual tracked region, matching what the tracked bbox is supposed to mean.

Everything but the propagation mechanism itself (Medsam2SliceViewerBase,
medsam2_viewer_base.py) is shared with the other two viewers: the Frame
slider, Plane toggle, Slice slider, Enhance checkbox, contour rendering,
and Dice/status line.

Usage (inside the notebook, after seg_data/bmode_image_data are defined):

    from medsam2_axial_propagation import Medsam2AxialPropagationViewer

    viewer = Medsam2AxialPropagationViewer(
        seg_data, bmode_image_data, model_cfg,
        checkpoint="/path/to/exp_log/3DMPUS_liver_tumor/checkpoints/checkpoint.pt",
    )
    viewer.show()
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from medsam2_viewer_base import (
    Medsam2SliceViewerBase, project_bbox_to_plane, get_plane_slice, smooth_3d_mask,
)

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def build_axial_video_tensor(volume_xyz, image_size, device):
    """
    (X, Y, Z) volume -> a (Z, 3, image_size, image_size) normalized tensor,
    one axial (Y, X) slice per Z index — same orientation as
    get_plane_slice(volume, "Axial", z). Resizing is done batched on the
    GPU (torch.nn.functional.interpolate) rather than per-slice PIL calls,
    since a full volume can be several hundred slices.
    """
    # (Z, Y, X), matching get_plane_slice's per-slice convention
    axial_stack = np.transpose(volume_xyz, (2, 1, 0)).astype(np.float32)
    stack = torch.from_numpy(axial_stack).to(device).unsqueeze(1)  # (Z, 1, Y, X)
    stack = stack.repeat(1, 3, 1, 1)  # (Z, 3, Y, X)
    resized = F.interpolate(
        stack, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    resized = resized / 255.0
    mean = torch.tensor(IMG_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMG_STD, device=device).view(1, 3, 1, 1)
    resized = (resized - mean) / std
    return resized


def propagate_axial(predictor, video_tensor, video_height, video_width, key_slice, box, device):
    """
    Prompt with `box` (original-pixel-space [x_min,y_min,x_max,y_max]) on
    `key_slice`, then propagate forward and backward through the whole
    Z-stack. Returns a (Z, video_height, video_width) bool mask volume.
    """
    z_dim = video_tensor.shape[0]
    mask_zyx = np.zeros((z_dim, video_height, video_width), dtype=bool)
    box_arr = np.array(box, dtype=np.float32)

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        inference_state = predictor.init_state(video_tensor, video_height, video_width)

        predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=key_slice, obj_id=1, box=box_arr,
        )
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask_zyx[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
        predictor.reset_state(inference_state)

        predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=key_slice, obj_id=1, box=box_arr,
        )
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
            inference_state, reverse=True
        ):
            mask_zyx[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
        predictor.reset_state(inference_state)

    return mask_zyx


def clip_mask_to_bbox(mask_zyx, bbox, z_dim, y_dim, x_dim):
    """
    Zero out every predicted voxel outside the tracked 3D bbox.

    SAM2's box prompt only *conditions* propagation -- it isn't a hard
    pixel-level constraint on the output, so a small amount of leakage past
    the tracked VOI is possible even though propagation is far
    better-behaved than independent per-slice calls (near-zero leakage was
    measured in practice, but not exactly zero -- see
    medsam2_axial_propagation's evaluation notes). Since the tracked bbox is
    the actual region of interest being followed for this frame, the
    prediction should never extend past it.
    """
    z_min = max(0, int(bbox.z_min)); z_max = min(z_dim, int(bbox.z_max))
    y_min = max(0, int(bbox.y_min)); y_max = min(y_dim, int(bbox.y_max))
    x_min = max(0, int(bbox.x_min)); x_max = min(x_dim, int(bbox.x_max))

    clipped = np.zeros_like(mask_zyx)
    clipped[z_min:z_max, y_min:y_max, x_min:x_max] = mask_zyx[z_min:z_max, y_min:y_max, x_min:x_max]
    return clipped


class Medsam2AxialPropagationViewer(Medsam2SliceViewerBase):
    """
    Frame slider + Plane toggle + Slice slider over one true SAM2 video
    propagation per CEUS frame. Only Axial is actually propagated -- the
    (Z,Y,X) result covers the whole volume, though, so Coronal/Sagittal are
    a different-axis re-slice of that same result, not new inference along
    those planes. The prompt-slice cyan-box overlay only makes sense on
    Axial (that's the only axis with a "seed slice" concept), so it's
    skipped for the other two planes.
    """

    SUPPORTED_PLANES = ("Axial", "Coronal", "Sagittal")

    def __init__(self, seg_data, bmode_image_data, model_cfg, checkpoint, device="cuda", smooth_sigma_mm=1.0):
        from sam2.build_sam import build_sam2_video_predictor_npz

        self.predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint, device=device)
        self.image_size = self.predictor.image_size
        # 3D Gaussian-smooth the propagated volume the same way
        # Medsam2AdaptiveBboxMasker does -- 0/None disables it.
        self.smooth_sigma_mm = smooth_sigma_mm
        super().__init__(seg_data, bmode_image_data, device=device)

    def _propagate_frame(self, frame_idx):
        """One full-volume propagation per CEUS frame, cached -- not once
        per (frame, slice) like the other two approaches' caches."""
        if frame_idx in self.cache:
            return self.cache[frame_idx]

        geom = self._frame_geometry(frame_idx)
        bbox = geom["bbox"]
        volume = self.bmode_image_data.pixel_data[:, :, :, frame_idx]  # (X, Y, Z)
        x_dim, y_dim, z_dim = volume.shape
        box = project_bbox_to_plane(bbox, "Axial")  # [x_min, y_min, x_max, y_max]
        key_slice = geom["z_mid"]

        start = time.time()
        video_tensor = build_axial_video_tensor(volume, self.image_size, self.device)
        mask_zyx = propagate_axial(
            self.predictor, video_tensor, y_dim, x_dim, key_slice, box, self.device
        )
        mask_zyx = clip_mask_to_bbox(mask_zyx, bbox, z_dim, y_dim, x_dim)
        if self.smooth_sigma_mm:
            # mask_zyx is (Z,Y,X); reorder pixdim (X,Y,Z) to match.
            sx, sy, sz = self.bmode_image_data.pixdim
            mask_zyx = smooth_3d_mask(mask_zyx, spacing_xyz=(sz, sy, sx), sigma_mm=self.smooth_sigma_mm)
            # Smoothing can bleed a voxel or two past the hard edge clip_mask_to_bbox
            # just applied (Gaussian blur mixes across the boundary) -- re-clip so the
            # "never exceeds the tracked bbox" guarantee still holds after smoothing.
            mask_zyx = clip_mask_to_bbox(mask_zyx, bbox, z_dim, y_dim, x_dim)
        elapsed = time.time() - start

        # Canonical (X, Y, Z) layout, matching the raw volume/mc_mask, so
        # get_plane_slice works the same way for any plane here as it does
        # for those -- this is what makes Coronal/Sagittal display "free".
        pred_xyz = np.transpose(mask_zyx, (2, 1, 0))

        result = {"pred_xyz": pred_xyz, "key_slice": key_slice, "box": box,
                  "mc_mask": geom["mc_mask"], "elapsed": elapsed}
        self.cache[frame_idx] = result
        return result

    def _slice_range(self, frame_idx, plane):
        """
        Axial stays the full stack, unclamped -- propagation is what
        correctly predicts "absent" outside the true Z extent, so there's
        nothing to gain by hiding those slices. Coronal/Sagittal are
        clamped to the bbox's Y/X extent, same convention as the other two
        viewers -- and now doubly justified, since clip_mask_to_bbox
        guarantees the prediction actually is empty outside that range.
        """
        if plane == "Axial":
            result = self._propagate_frame(frame_idx)
            z_dim = result["pred_xyz"].shape[2]
            return 0, z_dim - 1, result["key_slice"]
        geom = self._frame_geometry(frame_idx)
        bbox = geom["bbox"]
        if plane == "Coronal":
            return int(bbox.y_min), int(bbox.y_max) - 1, geom["y_mid"]
        else:
            return int(bbox.x_min), int(bbox.x_max) - 1, geom["x_mid"]

    def _get_prediction(self, frame_idx, plane, idx):
        result = self._propagate_frame(frame_idx)
        pred = get_plane_slice(result["pred_xyz"], plane, idx)
        return {
            "pred": pred if pred.any() else None,
            "elapsed": result["elapsed"],
            "key_slice": result["key_slice"],
            "box": result["box"],
        }

    def _status_note(self, frame_idx, plane, idx, result):
        if plane == "Axial" and idx == result.get("key_slice"):
            return "  <- prompt slice (cyan box)"
        return ""

    def _draw_extra(self, ax, frame_idx, plane, idx, result):
        if plane == "Axial" and idx == result.get("key_slice"):
            x0, y0, x1, y1 = result["box"]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                        edgecolor="cyan", facecolor="none", lw=1.5, linestyle=":"))
