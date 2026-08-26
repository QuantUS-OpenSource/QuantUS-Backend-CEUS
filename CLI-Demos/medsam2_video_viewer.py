"""
Interactive frame/plane viewer for MedSAM2 axial video-propagation results.

Wraps `propagate_axial_stack` (prompt one axial slice near z_mid with the 3D
bbox, then let SAM2's video predictor propagate the mask forward/backward
through the Z-stack) behind an ipywidgets UI:
  - a Frame slider (which CEUS frame),
  - a Plane toggle (Axial / Coronal / Sagittal),
  - a Slice slider (which index within that plane — not just the mid-slice),
  - an "Enhance" checkbox (denoise + percentile contrast stretch for display
    only; the underlying SAM2 input/prediction is unaffected, since plain
    min-max normalization was found to segment better than the denoised
    version).

Scrubbing the frame slider re-runs (and caches) the propagation for that
frame; changing plane or slice index just re-slices the already-computed 3D
mask, no re-inference needed.

Usage (inside the notebook, after seg_data/bmode_image_data/model_cfg/checkpoint
are already defined):

    from medsam2_video_viewer import Medsam2VideoPropViewer

    viewer = Medsam2VideoPropViewer(seg_data, bmode_image_data, model_cfg, checkpoint)
    viewer.show()
"""

import os
import shutil
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.restoration import denoise_nl_means, estimate_sigma
from ipywidgets import IntSlider, ToggleButtons, Checkbox, VBox, HTML, Output
from IPython.display import display


def slice_to_rgb(slice_2d):
    """Normalize a 2D slice to uint8 RGB."""
    s_min, s_max = slice_2d.min(), slice_2d.max()
    uint8 = ((slice_2d - s_min) / (s_max - s_min + 1e-8) * 255).astype(np.uint8)
    return np.stack([uint8] * 3, axis=-1)


def enhance_bmode_noise(image_slice, p_low_percentile=15.0, p_high_percentile=98.0):
    """Denoise (non-local means) + percentile-clip contrast stretch, for display only."""
    bmode_frame = image_slice.astype(np.float32)
    sigma_bmode_est = float(np.mean(estimate_sigma(bmode_frame)))
    bmode_denoised = denoise_nl_means(
        image_slice.astype(np.float32),
        h=1.15 * sigma_bmode_est, sigma=sigma_bmode_est,
        patch_size=5, patch_distance=6, fast_mode=True,
    )

    non_zero = bmode_denoised[bmode_denoised != 0]
    if non_zero.size == 0:
        return np.zeros_like(image_slice, dtype=np.uint8)

    p_low = np.percentile(non_zero, p_low_percentile)
    p_high = np.percentile(non_zero, p_high_percentile)
    clipped = np.clip(bmode_denoised, p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low + 1e-8) * 255).astype(np.uint8)


def get_mask_boundary(mask_slice):
    """Extract boundary of a binary mask using erosion."""
    if mask_slice.max() == 0:
        return np.zeros_like(mask_slice, dtype=bool)
    eroded = binary_erosion(mask_slice)
    return mask_slice.astype(bool) & ~eroded


def dice_score(a, b):
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / denom if denom > 0 else float("nan")


def propagate_axial_stack(video_predictor, volume, bbox, z_mid, frame_dir):
    """
    Segment axial slices z in [bbox.z_min, bbox.z_max) by prompting the z_mid
    slice with the 3D bbox's (x, y) extent, then propagating the mask forward
    and backward through the Z-stack with SAM2's video predictor.

    volume    : (X, Y, Z) — single B-mode frame
    bbox      : tracked 3D bbox for this frame
    z_mid     : seed slice index (absolute, into `volume`)
    frame_dir : empty directory to write the JPEG "video" frames into

    Returns: mask_3d (X, Y, Z) uint8, same shape as `volume`.
    """
    z_min, z_max = int(bbox.z_min), int(bbox.z_max)
    z_indices = list(range(z_min, z_max))    # video frame index i -> absolute z
    ann_frame_idx = z_mid - z_min
    if not (0 <= ann_frame_idx < len(z_indices)):
        raise ValueError(f"z_mid={z_mid} is outside bbox Z range [{z_min}, {z_max})")

    # SAM2's video loader only reads from a directory of "<frame_idx>.jpg" files,
    # so each axial slice becomes one frame.
    for i, z in enumerate(z_indices):
        axial_slice = volume[:, :, z].T                       # (Y, X)
        rgb = slice_to_rgb(axial_slice)
        Image.fromarray(rgb).save(os.path.join(frame_dir, f"{i:05d}.jpg"), quality=95)

    state = video_predictor.init_state(video_path=frame_dir)
    box = np.array([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max], dtype=np.float32)

    mask_3d = np.zeros(volume.shape, dtype=np.uint8)           # (X, Y, Z)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        video_predictor.add_new_points_or_box(
            state, frame_idx=ann_frame_idx, obj_id=1, box=box,
        )

        # Forward: z_mid -> z_max, then backward: z_mid -> z_min.
        for out_frame_idx, _, out_mask_logits in video_predictor.propagate_in_video(
            state, start_frame_idx=ann_frame_idx, reverse=False
        ):
            pred_yx = (out_mask_logits[0, 0] > 0.0).cpu().numpy()   # (Y, X)
            mask_3d[:, :, z_indices[out_frame_idx]] = pred_yx.T     # (Y,X) -> (X,Y)

        for out_frame_idx, _, out_mask_logits in video_predictor.propagate_in_video(
            state, start_frame_idx=ann_frame_idx, reverse=True
        ):
            pred_yx = (out_mask_logits[0, 0] > 0.0).cpu().numpy()
            mask_3d[:, :, z_indices[out_frame_idx]] = pred_yx.T

    return mask_3d


class Medsam2VideoPropViewer:
    """Frame slider + plane toggle + slice slider over cached video-propagation results."""

    PLANES = ("Axial", "Coronal", "Sagittal")

    def __init__(self, seg_data, bmode_image_data, model_cfg, checkpoint, device="cuda"):
        from sam2.build_sam import build_sam2_video_predictor

        self.seg_data = seg_data
        self.bmode_image_data = bmode_image_data
        self.video_predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
        self.cache = {}   # frame_idx -> computed result dict
        self.n_frames = bmode_image_data.pixel_data.shape[-1]
        self.volume_shape = bmode_image_data.pixel_data.shape[:3]   # (X, Y, Z)

        self.status = HTML(value="")
        self.out = Output()

        self.frame_slider = IntSlider(
            value=0, min=0, max=self.n_frames - 1, step=1,
            description="Frame:", continuous_update=False,
            layout={"width": "500px"},
        )
        self.plane_toggle = ToggleButtons(options=list(self.PLANES), description="Plane:")
        self.slice_slider = IntSlider(
            description="Slice:", continuous_update=False,
            layout={"width": "500px"},
        )
        self.enhance_checkbox = Checkbox(value=False, description="Enhance display (denoise + contrast)")

        self.frame_slider.observe(self._on_context_change, names="value")
        self.plane_toggle.observe(self._on_context_change, names="value")
        self.slice_slider.observe(self._on_slice_change, names="value")
        self.enhance_checkbox.observe(self._on_slice_change, names="value")

        self.ui = VBox([
            self.frame_slider,
            self.plane_toggle,
            self.slice_slider,
            self.enhance_checkbox,
            self.status,
            self.out,
        ])

    def _frame_geometry(self, frame_idx):
        """Cheap per-frame info (bbox + mid-slice indices) — no SAM2 inference."""
        bbox = self.seg_data.motion_compensation.tracked_bboxes[frame_idx]
        mc_mask = self.seg_data.motion_compensation.apply_to_mask(
            self.seg_data.seg_mask, frame_idx, 0
        )
        return {
            "bbox": bbox,
            "mc_mask": mc_mask,
            "z_min": int(bbox.z_min), "z_max": int(bbox.z_max),
            "z_mid": int(np.argmax(mc_mask.sum(axis=(0, 1)))),
            "y_mid": int(np.argmax(mc_mask.sum(axis=(0, 2)))),
            "x_mid": int(np.argmax(mc_mask.sum(axis=(1, 2)))),
        }

    def _compute_frame(self, frame_idx):
        geom = self._frame_geometry(frame_idx)
        volume = self.bmode_image_data.pixel_data[:, :, :, frame_idx]        # (X, Y, Z)

        tmp_dir = tempfile.mkdtemp(prefix="medsam2_video_")
        try:
            start = time.time()
            mask_3d = propagate_axial_stack(
                self.video_predictor, volume, geom["bbox"], geom["z_mid"], tmp_dir
            )
            elapsed = time.time() - start
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return {
            "volume": volume, "mc_mask": geom["mc_mask"], "mask_3d": mask_3d,
            "z_min": geom["z_min"], "z_max": geom["z_max"],
            "z_mid": geom["z_mid"], "y_mid": geom["y_mid"], "x_mid": geom["x_mid"],
            "dice": dice_score(mask_3d, geom["mc_mask"]), "elapsed": elapsed,
        }

    def _get_frame(self, frame_idx):
        if frame_idx not in self.cache:
            self.status.value = f"<i>Running video-propagation on frame {frame_idx}…</i>"
            self.cache[frame_idx] = self._compute_frame(frame_idx)
        return self.cache[frame_idx]

    def _sync_slice_slider(self):
        """Reset the slice slider's range/default to match the current frame + plane."""
        frame_idx = self.frame_slider.value
        plane = self.plane_toggle.value
        geom = self._frame_geometry(frame_idx)
        x_dim, y_dim, _ = self.volume_shape

        if plane == "Axial":
            lo, hi, default = geom["z_min"], geom["z_max"] - 1, geom["z_mid"]
        elif plane == "Coronal":
            lo, hi, default = 0, y_dim - 1, geom["y_mid"]
        else:
            lo, hi, default = 0, x_dim - 1, geom["x_mid"]

        # Set in an order that never puts min > max transiently.
        self.slice_slider.min = min(lo, self.slice_slider.max, self.slice_slider.min)
        self.slice_slider.max = hi
        self.slice_slider.min = lo
        self.slice_slider.value = default

    def _on_context_change(self, _change):
        self._sync_slice_slider()
        self._render()

    def _on_slice_change(self, _change):
        self._render()

    def _render(self):
        frame_idx = self.frame_slider.value
        plane = self.plane_toggle.value
        idx = self.slice_slider.value
        r = self._get_frame(frame_idx)

        if plane == "Axial":
            img  = r["volume"][:, :, idx].T
            pred = r["mask_3d"][:, :, idx].T
            mc   = r["mc_mask"][:, :, idx].T
            title, xlabel, ylabel = f"Axial (Z={idx})", "Lateral (X)", "Depth (Y)"
        elif plane == "Coronal":
            img  = r["volume"][:, idx, :]
            pred = r["mask_3d"][:, idx, :]
            mc   = r["mc_mask"][:, idx, :]
            title, xlabel, ylabel = f"Coronal (Y={idx})", "Elevation (Z)", "Lateral (X)"
        else:
            img  = r["volume"][idx, :, :]
            pred = r["mask_3d"][idx, :, :]
            mc   = r["mc_mask"][idx, :, :]
            title, xlabel, ylabel = f"Sagittal (X={idx})", "Elevation (Z)", "Depth (Y)"

        disp = enhance_bmode_noise(img) if self.enhance_checkbox.value else slice_to_rgb(img)[..., 0]

        with self.out:
            self.out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(disp, cmap="gray")
            ax.contour(get_mask_boundary(mc.astype(bool)), colors="red", linewidths=2)
            ax.contour(get_mask_boundary(pred.astype(bool)), colors="lime",
                       linewidths=2, linestyles="dashed")
            ax.set_title(f"Frame {frame_idx} — {title}", fontsize=12)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.tight_layout()
            plt.show()

        self.status.value = (
            f"Frame {frame_idx} | Dice vs MC mask: {r['dice']:.3f} | "
            f"propagation time: {r['elapsed']:.1f}s "
            f"(red = MC reference, lime dashed = video-prop mask)"
        )

    def show(self):
        display(self.ui)
        self._sync_slice_slider()
        self._render()
