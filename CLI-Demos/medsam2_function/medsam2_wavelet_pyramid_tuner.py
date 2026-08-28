"""
Stage-by-stage visual tuner for wavelet_pyramid_enhance (medsam2_wavelet_pyramid.py).

Same Frame / Plane / Slice navigation as the other tuners, plus a slider or
input box for every wavelet_pyramid_enhance parameter. Clicking "Update"
reloads medsam2_wavelet_pyramid fresh (so edits to the algorithm itself are
always picked up, same reasoning as the other tuners: %autoreload doesn't
fire inside a widget callback) and calls it with return_stages=True, then
shows every intermediate array from every pyramid level in a grid:

    approx_in | coherence | edge_mask | after_directional | after_guided |
    after_fractional | after_dehaze | reconstructed

one row per level (coarsest at top), plus a top row comparing the original
slice against the final output.

Usage (inside the notebook, after seg_data/bmode_image_data are already defined):

    from medsam2_wavelet_pyramid_tuner import WaveletPyramidStageViewer

    tuner = WaveletPyramidStageViewer(seg_data, bmode_image_data)
    tuner.show()
"""

import importlib
import traceback

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import (
    IntSlider, FloatSlider, ToggleButtons, Button, VBox, HBox, HTML, Output, Layout,
)
from IPython.display import display

import medsam2_video_viewer as _viewer_module
import medsam2_wavelet_pyramid as _pyramid_module

STAGE_COLUMNS = (
    "approx_in", "coherence", "edge_mask", "after_directional",
    "after_guided", "after_fractional", "after_dehaze", "reconstructed",
)

_SLIDER_WIDTH = Layout(width="420px")


class WaveletPyramidStageViewer:
    PLANES = ("Axial", "Coronal", "Sagittal")

    def __init__(self, seg_data, bmode_image_data):
        self.seg_data = seg_data
        self.bmode_image_data = bmode_image_data
        self.n_frames = bmode_image_data.pixel_data.shape[-1]
        self.volume_shape = bmode_image_data.pixel_data.shape[:3]   # (X, Y, Z)
        self.error = None

        self.status = HTML(value="")
        self.out = Output()

        # -- navigation --
        self.frame_slider = IntSlider(
            value=0, min=0, max=self.n_frames - 1, step=1,
            description="Frame:", continuous_update=False, layout=_SLIDER_WIDTH,
        )
        self.plane_toggle = ToggleButtons(options=list(self.PLANES), description="Plane:")
        self.slice_slider = IntSlider(
            description="Slice:", continuous_update=False, layout=_SLIDER_WIDTH,
        )

        # -- wavelet_pyramid_enhance parameters (see medsam2_wavelet_pyramid_tuning
        #    guide / the paper docstring for what each controls) --
        self.p_wavelet = ToggleButtons(options=["haar", "db1", "db2", "db4", "sym4"], value="db2",
                                        description="wavelet:")
        self.p_levels = IntSlider(value=2, min=1, max=4, step=1, description="levels:", layout=_SLIDER_WIDTH)
        self.p_tensor_sigma = FloatSlider(value=1.0, min=0.2, max=4.0, step=0.1,
                                           description="tensor_sigma:", layout=_SLIDER_WIDTH)
        self.p_edge_threshold = FloatSlider(value=0.3, min=0.0, max=1.0, step=0.05,
                                             description="edge_threshold:", layout=_SLIDER_WIDTH)
        self.p_alpha_e = FloatSlider(value=0.7, min=0.0, max=2.0, step=0.05,
                                      description="alpha_e:", layout=_SLIDER_WIDTH)
        self.p_alpha_n = FloatSlider(value=0.3, min=0.0, max=2.0, step=0.05,
                                      description="alpha_n:", layout=_SLIDER_WIDTH)
        self.p_guided_radius = IntSlider(value=4, min=1, max=15, step=1,
                                          description="guided_radius:", layout=_SLIDER_WIDTH)
        self.p_guided_eps = FloatSlider(value=1e-2, min=1e-4, max=1.0, step=1e-3, readout_format=".4f",
                                         description="guided_eps:", layout=_SLIDER_WIDTH)
        self.p_frac_order = FloatSlider(value=0.05, min=0.0, max=1.0, step=0.01,
                                         description="frac_order:", layout=_SLIDER_WIDTH)
        self.p_frac_terms = IntSlider(value=8, min=2, max=20, step=1,
                                       description="frac_terms:", layout=_SLIDER_WIDTH)
        self.p_dehaze_patch_radius = IntSlider(value=7, min=1, max=25, step=1,
                                                description="dehaze_patch_radius:", layout=_SLIDER_WIDTH)
        self.p_dehaze_omega = FloatSlider(value=0.95, min=0.0, max=1.0, step=0.01,
                                           description="dehaze_omega:", layout=_SLIDER_WIDTH)
        self.p_low_pct = FloatSlider(value=5.0, min=0.0, max=49.0, step=1.0,
                                      description="p_low_percentile:", layout=_SLIDER_WIDTH)
        self.p_high_pct = FloatSlider(value=98.0, min=51.0, max=100.0, step=1.0,
                                       description="p_high_percentile:", layout=_SLIDER_WIDTH)

        self.param_widgets = [
            self.p_wavelet, self.p_levels, self.p_tensor_sigma, self.p_edge_threshold,
            self.p_alpha_e, self.p_alpha_n, self.p_guided_radius, self.p_guided_eps,
            self.p_frac_order, self.p_frac_terms, self.p_dehaze_patch_radius,
            self.p_dehaze_omega, self.p_low_pct, self.p_high_pct,
        ]

        self.update_button = Button(description="Update", button_style="primary")

        self.frame_slider.observe(self._on_context_change, names="value")
        self.plane_toggle.observe(self._on_context_change, names="value")
        self.slice_slider.observe(self._on_slice_change, names="value")
        self.update_button.on_click(self._on_update_click)

        left_col = VBox([self.p_wavelet, self.p_levels, self.p_tensor_sigma, self.p_edge_threshold,
                          self.p_alpha_e, self.p_alpha_n, self.p_guided_radius])
        right_col = VBox([self.p_guided_eps, self.p_frac_order, self.p_frac_terms,
                           self.p_dehaze_patch_radius, self.p_dehaze_omega,
                           self.p_low_pct, self.p_high_pct])

        self.ui = VBox([
            self.frame_slider,
            self.plane_toggle,
            self.slice_slider,
            HBox([left_col, right_col]),
            self.update_button,
            self.status,
            self.out,
        ])

    def _frame_geometry(self, frame_idx):
        bbox = self.seg_data.motion_compensation.tracked_bboxes[frame_idx]
        mc_mask = self.seg_data.motion_compensation.apply_to_mask(self.seg_data.seg_mask, frame_idx, 0)
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

        self.slice_slider.min = min(lo, self.slice_slider.max, self.slice_slider.min)
        self.slice_slider.max = hi
        self.slice_slider.min = lo
        self.slice_slider.value = default

    def _current_slice(self):
        frame_idx = self.frame_slider.value
        plane = self.plane_toggle.value
        idx = self.slice_slider.value
        volume = self.bmode_image_data.pixel_data[:, :, :, frame_idx]
        return _viewer_module.get_plane_slice(volume, plane, idx)

    def _current_params(self):
        return dict(
            wavelet=self.p_wavelet.value,
            levels=self.p_levels.value,
            tensor_sigma=self.p_tensor_sigma.value,
            edge_threshold=self.p_edge_threshold.value,
            alpha_e=self.p_alpha_e.value,
            alpha_n=self.p_alpha_n.value,
            guided_radius=self.p_guided_radius.value,
            guided_eps=self.p_guided_eps.value,
            frac_order=self.p_frac_order.value,
            frac_terms=self.p_frac_terms.value,
            dehaze_patch_radius=self.p_dehaze_patch_radius.value,
            dehaze_omega=self.p_dehaze_omega.value,
            p_low_percentile=self.p_low_pct.value,
            p_high_percentile=self.p_high_pct.value,
        )

    def _on_context_change(self, _change):
        self._sync_slice_slider()
        self._compute_and_render()

    def _on_slice_change(self, _change):
        self._compute_and_render()

    def _on_update_click(self, _button):
        self._compute_and_render()

    def _compute_and_render(self):
        slice_2d = self._current_slice()

        with self.out:
            self.out.clear_output(wait=True)

            try:
                importlib.reload(_pyramid_module)
                result, stages = _pyramid_module.wavelet_pyramid_enhance(
                    slice_2d, return_stages=True, **self._current_params()
                )
                self.error = None
            except Exception:
                result, stages = None, None
                self.error = traceback.format_exc(limit=2)

            raw_disp = _viewer_module.slice_to_rgb(slice_2d)[..., 0]

            fig0, axes0 = plt.subplots(1, 2, figsize=(10, 5))
            axes0[0].imshow(raw_disp, cmap="gray")
            axes0[0].set_title("Original (raw)")
            axes0[1].imshow(result if result is not None else raw_disp, cmap="gray")
            axes0[1].set_title("Final wavelet_pyramid_enhance output")
            for ax in axes0:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            plt.show()

            if stages:
                n_levels = len(stages)
                fig, axes = plt.subplots(n_levels, len(STAGE_COLUMNS),
                                          figsize=(3 * len(STAGE_COLUMNS), 3 * n_levels),
                                          squeeze=False)
                for row, level_stage in enumerate(stages):
                    for col, key in enumerate(STAGE_COLUMNS):
                        ax = axes[row][col]
                        value = level_stage[key]
                        if key == "coherence":
                            ax.imshow(value, cmap="viridis", vmin=0, vmax=1)
                        elif key == "edge_mask":
                            ax.imshow(value, cmap="gray", vmin=0, vmax=1)
                        else:
                            ax.imshow(value, cmap="gray")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if row == 0:
                            ax.set_title(key, fontsize=9)
                        if col == 0:
                            ax.set_ylabel(f"level {row}", fontsize=9)
                plt.tight_layout()
                plt.show()

            if self.error:
                print("wavelet_pyramid_enhance raised an error on the last Update:")
                print(self.error)

        frame_idx, plane, idx = self.frame_slider.value, self.plane_toggle.value, self.slice_slider.value
        self.status.value = f"Frame {frame_idx} | {plane} slice {idx}"

    def show(self):
        display(self.ui)
        self._sync_slice_slider()
        self._compute_and_render()
