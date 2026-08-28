"""
Interactive frame/plane viewer for independent per-plane 2D MedSAM2 segmentation.

Mirrors the standalone "2D inference on all 3 planes (axial, coronal,
sagittal)" cell exactly: for whichever plane is selected, the tracked 3D bbox
is projected onto that plane's two axes and a single SAM2 2D image-predictor
call segments that one slice. There is no cross-plane guidance (coronal/
sagittal do not derive an axial bbox), no per-z "adaptive" bbox, and no
Z-stack reconstruction/propagation — each (frame, plane, slice index) is
computed independently and on demand.

The UI:
  - a Frame slider (which CEUS frame),
  - a Plane toggle (Axial / Coronal / Sagittal),
  - a Slice slider (which index within that plane — the fixed 3D bbox is
    projected the same way regardless of index, so this is a faithful
    generalization of the reference cell's single mid-slice, not a different
    algorithm),
  - an "Enhance" checkbox (denoise + percentile contrast stretch for display
    only; the SAM2 input/prediction always uses plain min-max normalization,
    since the denoised version was found to segment worse).

Each (frame, plane, slice) prediction is cached so revisiting one is instant.

Usage (inside the notebook, after seg_data/bmode_image_data/model_cfg/checkpoint
are already defined):

    from medsam2_video_viewer import Medsam2VideoPropViewer

    viewer = Medsam2VideoPropViewer(seg_data, bmode_image_data, model_cfg, checkpoint)
    viewer.show()
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import binary_erosion
from skimage.restoration import denoise_nl_means, denoise_wavelet, estimate_sigma
from ipywidgets import IntSlider, ToggleButtons, Checkbox, VBox, HTML, Output
from IPython.display import display


def slice_to_rgb(slice_2d):
    """Normalize a 2D slice to uint8 RGB."""
    s_min, s_max = slice_2d.min(), slice_2d.max()
    uint8 = ((slice_2d - s_min) / (s_max - s_min + 1e-8) * 255).astype(np.uint8)
    return np.stack([uint8] * 3, axis=-1)


def enhance_bmode_noise(
    image_slice,
    p_low_percentile=5.0,
    p_high_percentile=98.0,
    method="nlm",
    patch_size=5,
    patch_distance=9,
    h_multiplier=1.15,
    wavelet="db1",
    wavelet_mode="soft",
):
    """
    Denoise + percentile-clip contrast stretch, for display only. Prints the
    wall-clock time taken (denoise + clip/stretch combined) so different
    methods/parameters can be compared while tuning.

    method:
        "none"              - skip denoising, contrast-stretch only
        "nlm"               - non-local means only (patch_size/patch_distance/h_multiplier apply)
        "wavelet"           - wavelet-coefficient thresholding only (BayesShrink via
                              skimage.restoration.denoise_wavelet; wavelet/wavelet_mode apply)
        "nlm_then_wavelet"  - non-local means, then wavelet thresholding applied to the NLM
                              result (default) -- sequential, not a parallel choice between them
    """
    start = time.time()
    valid_methods = ("none", "nlm", "wavelet", "nlm_then_wavelet")
    if method not in valid_methods:
        raise ValueError(f"Unknown method {method!r}; expected one of {valid_methods}")

    denoised = image_slice.astype(np.float32)

    if method in ("nlm", "nlm_then_wavelet"):
        sigma_est = float(np.mean(estimate_sigma(denoised)))
        denoised = denoise_nl_means(
            denoised, h=h_multiplier * sigma_est, sigma=sigma_est,
            patch_size=patch_size, patch_distance=patch_distance, fast_mode=True,
        )

    if method in ("wavelet", "nlm_then_wavelet"):
        sigma_est = estimate_sigma(denoised)
        denoised = denoise_wavelet(
            denoised, sigma=sigma_est, wavelet=wavelet, mode=wavelet_mode,
            rescale_sigma=True,
        )

    non_zero = denoised[denoised != 0]
    if non_zero.size == 0:
        result = np.zeros_like(image_slice, dtype=np.uint8)
    else:
        p_low = np.percentile(non_zero, p_low_percentile)
        p_high = np.percentile(non_zero, p_high_percentile)
        clipped = np.clip(denoised, p_low, p_high)
        result = ((clipped - p_low) / (p_high - p_low + 1e-8) * 255).astype(np.uint8)

    elapsed_ms = (time.time() - start) * 1000
    print(f"enhance_bmode_noise[method={method}]: {elapsed_ms:.1f} ms")
    return result


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


def run_medsam2_2d(image_predictor, slice_2d, bbox_2d, device="cuda"):
    """
    Run SAM2ImagePredictor on one 2D slice.
    slice_2d : (H, W) float
    bbox_2d  : [x_min, y_min, x_max, y_max]
    Returns  : (H, W) bool mask, or None if bbox is degenerate.
    """
    x_min, y_min, x_max, y_max = bbox_2d
    if (x_max - x_min) < 2 or (y_max - y_min) < 2:
        return None

    rgb = slice_to_rgb(slice_2d)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        image_predictor.set_image(rgb)
        masks, _, _ = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(bbox_2d, dtype=np.float32)[None, :],
            multimask_output=False,
        )
    return masks[0].astype(bool)


def project_bbox_to_plane(bbox, plane):
    """Project the tracked 3D bbox onto one plane — same formulas as the
    reference "2D inference on all 3 planes" cell for each orientation."""
    if plane == "Axial":
        return [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
    elif plane == "Coronal":
        return [bbox.z_min, bbox.x_min, bbox.z_max, bbox.x_max]
    else:  # Sagittal
        return [bbox.z_min, bbox.y_min, bbox.z_max, bbox.y_max]


def get_plane_slice(volume_xyz, plane, idx):
    """
    Extract one 2D slice from an (X, Y, Z) array, in the same orientation
    convention used throughout (axial is transposed to (Y, X); coronal/
    sagittal are left as (X, Z) / (Y, Z)).
    """
    if plane == "Axial":
        return volume_xyz[:, :, idx].T      # (Y, X)
    elif plane == "Coronal":
        return volume_xyz[:, idx, :]        # (X, Z)
    else:
        return volume_xyz[idx, :, :]        # (Y, Z)


class Medsam2VideoPropViewer:
    """Frame slider + plane toggle + slice slider over independent per-plane 2D SAM2 segmentation."""

    PLANES = ("Axial", "Coronal", "Sagittal")

    def __init__(self, seg_data, bmode_image_data, model_cfg, checkpoint, device="cuda"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.seg_data = seg_data
        self.bmode_image_data = bmode_image_data
        self.device = device
        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        self.image_predictor = SAM2ImagePredictor(sam2_model)
        self.cache = {}   # (frame_idx, plane, idx) -> {"pred": mask|None, "elapsed": float}
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

    def _get_prediction(self, frame_idx, volume, bbox, plane, idx):
        key = (frame_idx, plane, idx)
        if key not in self.cache:
            slice_2d = get_plane_slice(volume, plane, idx)
            box = project_bbox_to_plane(bbox, plane)
            start = time.time()
            pred = run_medsam2_2d(self.image_predictor, slice_2d, box, device=self.device)
            elapsed = time.time() - start
            self.cache[key] = {"pred": pred, "elapsed": elapsed}
        return self.cache[key]

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

        geom = self._frame_geometry(frame_idx)
        volume = self.bmode_image_data.pixel_data[:, :, :, frame_idx]        # (X, Y, Z)

        img = get_plane_slice(volume, plane, idx)
        mc = get_plane_slice(geom["mc_mask"], plane, idx)
        result = self._get_prediction(frame_idx, volume, geom["bbox"], plane, idx)
        pred = result["pred"]

        if plane == "Axial":
            title, xlabel, ylabel = f"Axial (Z={idx})", "Lateral (X)", "Depth (Y)"
        elif plane == "Coronal":
            title, xlabel, ylabel = f"Coronal (Y={idx})", "Elevation (Z)", "Lateral (X)"
        else:
            title, xlabel, ylabel = f"Sagittal (X={idx})", "Elevation (Z)", "Depth (Y)"

        disp = enhance_bmode_noise(img) if self.enhance_checkbox.value else slice_to_rgb(img)[..., 0]

        with self.out:
            self.out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(disp, cmap="gray")
            ax.contour(get_mask_boundary(mc.astype(bool)), colors="red", linewidths=2)
            if pred is not None:
                ax.contour(get_mask_boundary(pred), colors="lime", linewidths=2, linestyles="dashed")
            ax.set_title(f"Frame {frame_idx} — {title}", fontsize=12)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.tight_layout()
            plt.show()

        dice = dice_score(pred, mc) if pred is not None else float("nan")
        note = "" if pred is not None else " (bbox degenerate on this plane)"
        self.status.value = (
            f"Frame {frame_idx} | {plane} slice {idx} | Dice vs MC (this slice): {dice:.3f}{note} | "
            f"segmentation time: {result['elapsed']:.2f}s "
            f"(red = MC reference, lime dashed = MedSAM2 mask)"
        )

    def show(self):
        display(self.ui)
        self._sync_slice_slider()
        self._render()
