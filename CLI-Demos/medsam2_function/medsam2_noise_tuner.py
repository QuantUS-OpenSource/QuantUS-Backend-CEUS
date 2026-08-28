"""
Side-by-side before/after tuner for the B-mode enhancement functions:
enhance_bmode_noise (medsam2_video_viewer.py) and wavelet_pyramid_enhance
(medsam2_wavelet_pyramid.py) -- a "Function:" toggle picks which one runs.

Same Frame / Plane / Slice navigation as Medsam2VideoPropViewer, but no
MedSAM2 model involved at all -- this is purely for tuning the enhancement
functions themselves. Two panels:
  - left  = the last-shown result (raw slice, before the first Update),
  - right = the selected function applied fresh to the current slice.

Workflow: edit the selected function's parameters (or body) in its .py file,
save, then click "Update" here -- both candidate modules are reloaded from
disk on every click regardless of which is selected. This is necessary
rather than relying on the notebook's %autoreload: autoreload only checks
for changed source files right before a *new cell* runs, not inside an
already-running widget button callback, so a stale function would otherwise
keep getting called no matter how many times the file is saved.

Clicking Update shifts the comparison forward: left becomes whatever was
just on the right, and right becomes the newly (re)processed image -- so
each click shows what changed between one parameter tweak and the next,
not just raw vs. processed every time. Moving the frame/plane/slice resets
back to raw (left) vs. current-on-disk-parameters (right).

Usage (inside the notebook, after seg_data/bmode_image_data are already defined):

    from medsam2_noise_tuner import EnhanceBmodeNoiseTuner

    tuner = EnhanceBmodeNoiseTuner(seg_data, bmode_image_data)
    tuner.show()
"""

import importlib
import traceback

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, ToggleButtons, Button, VBox, HTML, Output
from IPython.display import display

import medsam2_video_viewer as _viewer_module
import medsam2_wavelet_pyramid as _pyramid_module


class EnhanceBmodeNoiseTuner:
    PLANES = ("Axial", "Coronal", "Sagittal")
    FUNCTIONS = ("enhance_bmode_noise", "wavelet_pyramid_enhance")

    def __init__(self, seg_data, bmode_image_data):
        self.seg_data = seg_data
        self.bmode_image_data = bmode_image_data
        self.n_frames = bmode_image_data.pixel_data.shape[-1]
        self.volume_shape = bmode_image_data.pixel_data.shape[:3]   # (X, Y, Z)

        self.left_image = None
        self.right_image = None
        self.error = None

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
        self.function_toggle = ToggleButtons(options=list(self.FUNCTIONS), description="Function:")
        self.update_button = Button(description="Update", button_style="primary")

        self.frame_slider.observe(self._on_context_change, names="value")
        self.plane_toggle.observe(self._on_context_change, names="value")
        self.slice_slider.observe(self._on_context_change, names="value")
        self.function_toggle.observe(self._on_context_change, names="value")
        self.update_button.on_click(self._on_update_click)

        self.ui = VBox([
            self.frame_slider,
            self.plane_toggle,
            self.slice_slider,
            self.function_toggle,
            self.update_button,
            self.status,
            self.out,
        ])

    def _frame_geometry(self, frame_idx):
        """Cheap per-frame info (bbox + mid-slice indices) — no model involved."""
        bbox = self.seg_data.motion_compensation.tracked_bboxes[frame_idx]
        mc_mask = self.seg_data.motion_compensation.apply_to_mask(
            self.seg_data.seg_mask, frame_idx, 0
        )
        return {
            "z_min": int(bbox.z_min), "z_max": int(bbox.z_max),
            "z_mid": int(np.argmax(mc_mask.sum(axis=(0, 1)))),
            "y_mid": int(np.argmax(mc_mask.sum(axis=(0, 2)))),
            "x_mid": int(np.argmax(mc_mask.sum(axis=(1, 2)))),
        }

    def _sync_slice_slider(self):
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

    def _current_slice(self):
        frame_idx = self.frame_slider.value
        plane = self.plane_toggle.value
        idx = self.slice_slider.value
        volume = self.bmode_image_data.pixel_data[:, :, :, frame_idx]   # (X, Y, Z)
        return _viewer_module.get_plane_slice(volume, plane, idx)

    def _process(self, slice_2d):
        """
        Reload both candidate modules fresh, then run whichever function is
        currently selected -- both are reloaded regardless of selection so
        switching the toggle always reflects on-disk edits too.
        """
        importlib.reload(_viewer_module)
        importlib.reload(_pyramid_module)
        if self.function_toggle.value == "wavelet_pyramid_enhance":
            return _pyramid_module.wavelet_pyramid_enhance(slice_2d)
        return _viewer_module.enhance_bmode_noise(slice_2d)

    def _on_context_change(self, _change):
        # Only re-derive the slider range on frame/plane changes, not on the
        # slice slider's own change (that would just re-clamp it to itself).
        if self.frame_slider is not None:
            self._sync_slice_slider()
        self._compute_and_render(reset=True)

    def _on_update_click(self, _button):
        self._compute_and_render(reset=False)

    def _compute_and_render(self, reset):
        """
        Process (inside the Output capture, so enhance_bmode_noise's timing
        print lands with the images instead of spamming the cell below) and
        redraw both panels.
        """
        slice_2d = self._current_slice()

        with self.out:
            self.out.clear_output(wait=True)

            if reset:
                # New frame/plane/slice: back to raw (left) vs. current-params (right).
                self.left_image = _viewer_module.slice_to_rgb(slice_2d)[..., 0]
            elif self.right_image is not None:
                # Whatever was on the right becomes the new left -- compare
                # this tweak against the *last* tweak, not always against raw.
                self.left_image = self.right_image

            try:
                self.right_image = self._process(slice_2d)
                self.error = None
            except Exception:
                if reset:
                    self.right_image = self.left_image
                # else: keep the previous right_image so the panels don't go blank
                self.error = traceback.format_exc(limit=1)

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(self.left_image, cmap="gray")
            axes[0].set_title("Left (previous result)")
            axes[1].imshow(self.right_image, cmap="gray")
            axes[1].set_title(f"Right (current {self.function_toggle.value})")
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            plt.show()
            if self.error:
                print("enhance_bmode_noise raised an error on the last Update:")
                print(self.error)

        frame_idx, plane, idx = self.frame_slider.value, self.plane_toggle.value, self.slice_slider.value
        self.status.value = f"Frame {frame_idx} | {plane} slice {idx}"

    def show(self):
        display(self.ui)
        self._sync_slice_slider()
        self._compute_and_render(reset=True)
