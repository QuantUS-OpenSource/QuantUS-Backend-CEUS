"""
Shared machinery for the three MedSAM2 slice-viewer approaches:

  - Medsam2IndependentPlaneViewer (medsam2_video_viewer.py)  -- independent
    per-(frame, plane, slice) 2D SAM2 calls, fixed 3D bbox projected onto
    whichever plane is selected.
  - Medsam2AdaptiveBboxMasker (medsam2_3d_mask.py)  -- axial mask stack
    reconstructed from a per-z bbox derived from one coronal + one sagittal
    2D call, still independent 2D calls per axial slice.
  - Medsam2AxialPropagationViewer (medsam2_axial_propagation.py)  -- one
    real SAM2 video-object propagation per CEUS frame through the axial
    Z-stack, memory-linked across slices.

All three need the same interactive shell: a Frame slider (which CEUS
frame), optionally a Plane toggle (only the approaches that support more
than one plane get one), a Slice slider, an Enhance-display checkbox, and a
matplotlib render of the B-mode slice with the motion-compensated reference
contour (red) and the MedSAM2 prediction contour (lime dashed), plus a Dice
status line. What differs between them is only *how a prediction for one
(frame, plane, slice) is obtained* and *what slice range is valid* --
independent per-slice inference vs. indexing into an already-computed
volume, and bbox-clamped vs. full-stack slice ranges. Those two differences
are the hook methods subclasses must implement; everything else lives here
once instead of being copy-pasted three times.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import binary_erosion, gaussian_filter
from skimage.restoration import denoise_nl_means, denoise_wavelet, estimate_sigma
from ipywidgets import IntSlider, ToggleButtons, Checkbox, VBox, HTML, Output
from IPython.display import display


# ── Generic plane/mask/prediction utilities, shared by all three approaches ──

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


def smooth_3d_mask(mask_xyz, spacing_xyz=None, sigma_mm=1.0):
    """
    Smooth a binary 3D mask by Gaussian-blurring its float-cast field and
    re-thresholding at 0.5.

    Both Medsam2AdaptiveBboxMasker and Medsam2AxialPropagationViewer build
    their 3D mask out of a stack of per-axial-slice 2D predictions (fully
    independent 2D calls for the former; the video predictor's per-slice
    output for the latter). Nothing enforces that adjacent slices' boundary
    contours agree with each other, so the mask can show a visible
    staircase/"jigsaw" seam when viewed from Coronal/Sagittal -- planes that
    cut directly across many independently-decided slices. Smoothing the
    per-slice 2D output on its own wouldn't touch this, since the artifact
    is *between* slices, not within one; blurring the whole 3D field is
    what actually blends across the slice axis.

    spacing_xyz, if given, is the physical (mm) voxel spacing per axis, in
    the same axis order as mask_xyz -- sigma_mm is then converted to a
    different number of voxels per axis so the smoothing is isotropic in
    physical space rather than voxel space. Axial in-plane resolution is
    typically much finer than the through-plane spacing (e.g. ~0.3mm vs
    ~0.4-0.6mm here), so an equal voxel-count sigma would over-smooth
    in-plane while barely touching the actual staircase seam.
    """
    if mask_xyz.sum() == 0:
        return mask_xyz.astype(bool)

    if spacing_xyz is None:
        sigma_voxels = (sigma_mm, sigma_mm, sigma_mm)
    else:
        sigma_voxels = tuple(sigma_mm / max(float(s), 1e-6) for s in spacing_xyz)

    prob = gaussian_filter(mask_xyz.astype(np.float32), sigma=sigma_voxels)
    return prob > 0.5


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


# ── Shared interactive-viewer shell ──

class Medsam2SliceViewerBase:
    """
    Frame slider (+ Plane toggle, if SUPPORTED_PLANES has more than one) +
    Slice slider, rendering the B-mode slice with the MC reference contour
    (red) and the MedSAM2 prediction contour (lime dashed).

    Subclasses must implement:
      _get_prediction(frame_idx, plane, idx) -> dict with at least
          {"pred": (H,W) bool mask or None, "elapsed": float}
          (extra keys are fine -- passed through to _status_note/_draw_extra)
      _slice_range(frame_idx, plane) -> (lo, hi, default)

    Subclasses may optionally override:
      _status_note(frame_idx, plane, idx, result) -> str appended to the status line
      _draw_extra(ax, frame_idx, plane, idx, result) -> extra matplotlib annotations
    """

    SUPPORTED_PLANES = ("Axial",)

    def __init__(self, seg_data, bmode_image_data, device="cuda"):
        self.seg_data = seg_data
        self.bmode_image_data = bmode_image_data
        self.device = device
        self.n_frames = bmode_image_data.pixel_data.shape[-1]
        self.cache = {}

        self.status = HTML(value="")
        self.out = Output()

        self.frame_slider = IntSlider(
            value=0, min=0, max=self.n_frames - 1, step=1,
            description="Frame:", continuous_update=False,
            layout={"width": "500px"},
        )
        self.frame_slider.observe(self._on_context_change, names="value")

        if len(self.SUPPORTED_PLANES) > 1:
            self.plane_toggle = ToggleButtons(options=list(self.SUPPORTED_PLANES), description="Plane:")
            self.plane_toggle.observe(self._on_context_change, names="value")
        else:
            self.plane_toggle = None

        self.slice_slider = IntSlider(
            description="Slice:", continuous_update=False,
            layout={"width": "500px"},
        )
        self.slice_slider.observe(self._on_slice_change, names="value")

        self.enhance_checkbox = Checkbox(value=False, description="Enhance display (denoise + contrast)")
        self.enhance_checkbox.observe(self._on_slice_change, names="value")

        widgets = [self.frame_slider]
        if self.plane_toggle is not None:
            widgets.append(self.plane_toggle)
        widgets += [self.slice_slider, self.enhance_checkbox, self.status, self.out]
        self.ui = VBox(widgets)

    def _current_plane(self):
        return self.plane_toggle.value if self.plane_toggle is not None else self.SUPPORTED_PLANES[0]

    def _frame_geometry(self, frame_idx):
        """Cheap per-frame info (bbox + mc mask + mid-slice indices) shared
        by every subclass's prompt/range logic -- no SAM2 inference."""
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

    # ── hooks subclasses must implement ──

    def _get_prediction(self, frame_idx, plane, idx):
        raise NotImplementedError

    def _slice_range(self, frame_idx, plane):
        raise NotImplementedError

    # ── hooks subclasses may optionally override ──

    def _status_note(self, frame_idx, plane, idx, result):
        return ""

    def _draw_extra(self, ax, frame_idx, plane, idx, result):
        pass

    # ── shared UI wiring ──

    @staticmethod
    def _plane_axis_labels(plane, idx):
        if plane == "Axial":
            return f"Axial (Z={idx})", "Lateral (X)", "Depth (Y)"
        elif plane == "Coronal":
            return f"Coronal (Y={idx})", "Elevation (Z)", "Lateral (X)"
        else:
            return f"Sagittal (X={idx})", "Elevation (Z)", "Depth (Y)"

    def _sync_slice_slider(self):
        """Reset the slice slider's range/default to match the current frame + plane."""
        frame_idx = self.frame_slider.value
        plane = self._current_plane()
        lo, hi, default = self._slice_range(frame_idx, plane)

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
        plane = self._current_plane()
        idx = self.slice_slider.value

        geom = self._frame_geometry(frame_idx)
        volume = self.bmode_image_data.pixel_data[:, :, :, frame_idx]        # (X, Y, Z)

        img = get_plane_slice(volume, plane, idx)
        mc = get_plane_slice(geom["mc_mask"], plane, idx)
        result = self._get_prediction(frame_idx, plane, idx)
        pred = result.get("pred")

        title, xlabel, ylabel = self._plane_axis_labels(plane, idx)
        disp = enhance_bmode_noise(img) if self.enhance_checkbox.value else slice_to_rgb(img)[..., 0]

        with self.out:
            self.out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(disp, cmap="gray")
            ax.contour(get_mask_boundary(mc.astype(bool)), colors="red", linewidths=2)
            if pred is not None and np.any(pred):
                ax.contour(get_mask_boundary(pred), colors="lime", linewidths=2, linestyles="dashed")
            self._draw_extra(ax, frame_idx, plane, idx, result)
            ax.set_title(f"Frame {frame_idx} — {title}", fontsize=12)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.tight_layout()
            plt.show()

        dice = dice_score(pred, mc) if pred is not None else float("nan")
        note = self._status_note(frame_idx, plane, idx, result)
        elapsed = result.get("elapsed", float("nan"))
        self.status.value = (
            f"Frame {frame_idx} | {plane} slice {idx} | Dice vs MC (this slice): {dice:.3f}{note} | "
            f"prediction time: {elapsed:.2f}s "
            f"(red = MC reference, lime dashed = MedSAM2 mask)"
        )

    def show(self):
        display(self.ui)
        self._sync_slice_slider()
        self._render()
