"""
Single consolidated napari viewer for the 4D (X, Y, Z, T) CEUS series.

Shows, all in one viewer with a frame slider driving every layer:
  - the CEUS volume across frames ("4D movement"),
  - the original (non-motion-compensated) reference mask — static, since it
    is defined once and motion compensation is what makes it frame-aware,
  - the motion-compensated mask for the current frame,
  - the MedSAM2 predicted mask for the current frame (adaptive-bbox
    reconstruction from medsam2_3d_mask.py),
  - the tracked 3D bounding box for the current frame, as a wireframe of its
    six faces.

Usage (inside the notebook, after seg_data/image_data/bmode_image_data/
model_cfg/checkpoint are already defined):

    from medsam2_napari_viewer import Medsam2Napari4DViewer

    viewer = Medsam2Napari4DViewer(seg_data, image_data, bmode_image_data, model_cfg, checkpoint)
    viewer.show()
"""

import os

import numpy as np
import napari

from medsam2_3d_mask import Medsam2AdaptiveBboxMasker

os.environ.setdefault("QT_API", "pyqt6")


def bbox_faces_zyx(bbox):
    """Six rectangular faces of the 3D bbox, in napari's (Z, Y, X) order."""
    z0, z1 = bbox.z_min, bbox.z_max
    y0, y1 = bbox.y_min, bbox.y_max
    x0, x1 = bbox.x_min, bbox.x_max
    return [
        np.array([[z0, y0, x0], [z0, y0, x1], [z0, y1, x1], [z0, y1, x0]]),  # z_min face
        np.array([[z1, y0, x0], [z1, y0, x1], [z1, y1, x1], [z1, y1, x0]]),  # z_max face
        np.array([[z0, y0, x0], [z0, y0, x1], [z1, y0, x1], [z1, y0, x0]]),  # y_min face
        np.array([[z0, y1, x0], [z0, y1, x1], [z1, y1, x1], [z1, y1, x0]]),  # y_max face
        np.array([[z0, y0, x0], [z0, y1, x0], [z1, y1, x0], [z1, y0, x0]]),  # x_min face
        np.array([[z0, y0, x1], [z0, y1, x1], [z1, y1, x1], [z1, y0, x1]]),  # x_max face
    ]


class Medsam2Napari4DViewer:
    """One napari viewer, one frame slider, driving CEUS/original-mask/MC-mask/predicted-mask/bbox layers."""

    def __init__(self, seg_data, image_data, bmode_image_data, model_cfg, checkpoint, device="cuda"):
        self.seg_data = seg_data
        self.image_data = image_data
        self.masker = Medsam2AdaptiveBboxMasker(seg_data, bmode_image_data, model_cfg, checkpoint, device=device)
        self.n_frames = image_data.pixel_data.shape[-1]

    def show(self):
        result0 = self.masker.compute_frame(0)

        self.viewer = napari.Viewer(title="MedSAM2 4D viewer")

        self.ceus_layer = self.viewer.add_image(
            self.image_data.pixel_data[:, :, :, 0].T,          # (Z, Y, X)
            name="CEUS volume",
            colormap="gray",
            blending="additive",
        )
        self.viewer.add_labels(
            self.seg_data.seg_mask.T.astype(np.uint8),          # static — no motion compensation
            name="Original mask (no MC)",
            opacity=0.3,
        )
        self.mc_layer = self.viewer.add_labels(
            result0["mc_mask"].T.astype(np.uint8),
            name="Motion-compensated mask (reference)",
            opacity=0.4,
        )
        self.pred_layer = self.viewer.add_labels(
            result0["mask_3d"].T.astype(np.uint8),
            name="MedSAM2 predicted mask",
            colormap={1: "lime", None: "transparent"},
            opacity=0.4,
        )
        self.bbox_layer = self.viewer.add_shapes(
            bbox_faces_zyx(result0["bbox"]),
            shape_type="polygon",
            edge_color="yellow",
            face_color="transparent",
            edge_width=2,
            name="Tracked 3D bbox",
        )

        self.status = self.viewer.text_overlay
        self.status.visible = True
        self._set_status(0, result0)

        self._add_frame_slider()
        return self.viewer

    def _set_status(self, frame_idx, result):
        self.status.text = (
            f"Frame {frame_idx}  |  Dice (predicted vs MC): {result['dice']:.3f}"
        )

    def _add_frame_slider(self):
        from qtpy.QtWidgets import QSlider, QLabel, QVBoxLayout, QWidget
        from qtpy.QtCore import Qt

        outer = self

        class FrameControlWidget(QWidget):
            def __init__(self):
                super().__init__()
                layout = QVBoxLayout()
                self.label = QLabel("Frame: 0")
                self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                self.slider = QSlider(Qt.Orientation.Horizontal)
                self.slider.setMinimum(0)
                self.slider.setMaximum(outer.n_frames - 1)
                self.slider.setValue(0)
                self.slider.setTickInterval(10)
                self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                self.slider.valueChanged.connect(self.on_slider_changed)

                layout.addWidget(self.label)
                layout.addWidget(self.slider)
                self.setLayout(layout)

            def on_slider_changed(self, frame_idx):
                self.label.setText(f"Frame: {frame_idx}  (computing...)")
                result = outer.masker.compute_frame(frame_idx)

                outer.ceus_layer.data = outer.image_data.pixel_data[:, :, :, frame_idx].T
                outer.mc_layer.data = result["mc_mask"].T.astype(np.uint8)
                outer.pred_layer.data = result["mask_3d"].T.astype(np.uint8)
                outer.bbox_layer.data = bbox_faces_zyx(result["bbox"])
                outer._set_status(frame_idx, result)

                self.label.setText(f"Frame: {frame_idx}")

        self.frame_widget = FrameControlWidget()
        self.viewer.window.add_dock_widget(self.frame_widget, name="Frame Control", area="bottom")
