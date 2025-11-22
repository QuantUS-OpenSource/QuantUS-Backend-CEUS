import itertools
import warnings

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

from ...data_objs.image import UltrasoundImage
from ...data_objs.seg import CeusSeg
from ..curve_types.functions import *

class_name = "CurvesAnalysis"

class CurvesAnalysis:
    """
    Class to complete RF analysis via the sliding window technique
    and generate a corresponding parametric map.
    """
    def __init__(self, image_data: UltrasoundImage, seg: CeusSeg,
                 curve_groups: List[str], **kwargs):
        # Type checking
        assert isinstance(image_data, UltrasoundImage), 'image_data must be an UltrasoundImage child class'
        assert isinstance(seg, CeusSeg), 'seg_data must be a CeusSeg'

        self.analysis_kwargs = kwargs
        self.curve_groups = curve_groups
        self.seg_data = seg
        self.image_data = image_data
        self.curves: List[Dict[str, List[float]]] = [{}]  # List to hold computed curves
        self.curve_funcs: Dict[str, callable] = {name: globals()[name] for name in self.curve_groups if name in globals()}
        self.curves_output_path = self.analysis_kwargs.get('curves_output_path', None)
        frame_rate = self.image_data.frame_rate if np.isfinite(self.image_data.frame_rate) else 1.0
        self._per_frame_masks: Optional[np.ndarray] = None
        if image_data.intensities_for_analysis.ndim == 4: # 3D + time
            self.time_arr = np.arange(self.image_data.intensities_for_analysis.shape[3]) * frame_rate
        elif image_data.intensities_for_analysis.ndim == 3: # 2D + time
            self.time_arr = np.arange(self.image_data.intensities_for_analysis.shape[0]) * frame_rate
            self._per_frame_masks = self._prepare_per_frame_masks()
        else:
            raise ValueError("Image data must be either 2D+time or 3D+time.")

    def compute_curves(self):
        """Compute UTC parameters for each window in the ROI, creating a parametric map.
        """
        if len(self.image_data.intensities_for_analysis.shape) == 4: # 3D + time
            for frame_ix, frame in tqdm(enumerate(range(self.image_data.intensities_for_analysis.shape[3])),
                                        desc="Computing curves", total=self.image_data.intensities_for_analysis.shape[3]):
                frame_data = self.image_data.intensities_for_analysis[:, :, :, frame]
                self.extract_frame_features(frame_data, self.seg_data.seg_mask, frame_ix)
        elif len(self.image_data.intensities_for_analysis.shape) == 3: # 2D + time
            for frame_ix, frame in tqdm(enumerate(range(self.image_data.intensities_for_analysis.shape[0])),
                                        desc="Computing curves", total=self.image_data.intensities_for_analysis.shape[0]):
                frame_data = self.image_data.intensities_for_analysis[frame]
                if self._per_frame_masks is not None:
                    frame_mask = self._per_frame_masks[frame_ix]
                else:
                    frame_mask = self.seg_data.seg_mask
                self.extract_frame_features(frame_data, frame_mask, frame_ix)

        if self.curves_output_path:
            self.save_curves()

    def extract_frame_features(self, frame: np.ndarray, mask: np.ndarray, frame_ix: int):
        """Compute parametric map values for a single frame.
        Args:
            frame (np.ndarray): The ultrasound frame data.
            mask (np.ndarray): The mask for the region of interest.
            frame_ix (int): The index of the frame being processed.
        Returns:
            np.ndarray: The computed parametric map values for the frame.
        """
        used_curve_names = []
        for curve_group in self.curve_groups:
            curve_function = self.curve_funcs[curve_group]
            curve_names, vals = curve_function(self.image_data, frame, mask, **self.analysis_kwargs)

            if not len(curve_names) and not len(vals):
                if frame_ix:
                    print(self.image_data.scan_name, self.seg_data.seg_name, frame_ix, "No curves computed for this frame.")
                    for name in self.curves[0].keys():
                        self.curves[0][name].append(self.curves[0][name][-1])  # Append last value if no new values
            else:
                for name, val in zip(curve_names, vals):
                    if name in used_curve_names:
                        raise ValueError(f"Curve name '{name}' is being computed multiple times.")
                    if not name in self.curves[0].keys():
                        self.curves[0][name] = []
                    self.curves[0][name].append(val)
                    used_curve_names.append(name)
                    if frame_ix == len(self.curves[0][name]):
                        self.curves[0][name].append(val)

    def _prepare_per_frame_masks(self) -> Optional[np.ndarray]:
        """Align segmentation mask with frame axis for 2D+time data."""
        mask = np.asarray(self.seg_data.seg_mask)
        mask = np.squeeze(mask)

        n_frames, height, width = self.image_data.intensities_for_analysis.shape

        if mask.ndim == 2:
            mask_2d = self._ensure_xy_alignment(mask, height, width)
            return np.repeat(mask_2d[np.newaxis, :, :], n_frames, axis=0)

        if mask.ndim != 3:
            return None

        mask_3d = mask > 0
        aligned = None
        for perm in itertools.permutations(range(3)):
            candidate = np.transpose(mask_3d, perm)
            if candidate.shape[1:] == (height, width):
                aligned = candidate
                break
            if candidate.shape[1:] == (width, height):
                aligned = np.swapaxes(candidate, 1, 2)
                break

        if aligned is None:
            raise ValueError(
                f"Segmentation mask spatial shape {mask.shape} cannot be aligned with expected {(height, width)}"
            )

        if aligned.shape[0] < n_frames:
            warnings.warn(
                f"Segmentation mask has {aligned.shape[0]} frames; padding to match pixel data frames ({n_frames}).",
                RuntimeWarning,
            )
            pad = np.repeat(aligned[-1:], n_frames - aligned.shape[0], axis=0)
            aligned = np.concatenate([aligned, pad], axis=0)
        elif aligned.shape[0] > n_frames:
            warnings.warn(
                f"Segmentation mask has {aligned.shape[0]} frames; truncating to match pixel data frames ({n_frames}).",
                RuntimeWarning,
            )
            aligned = aligned[:n_frames]

        flat_counts = aligned.reshape(aligned.shape[0], -1).sum(axis=1)
        if not np.all(flat_counts):
            active_indices = np.where(flat_counts > 0)[0]
            if active_indices.size == 0:
                warnings.warn("Segmentation mask is empty for all frames.", RuntimeWarning)
                return aligned
            first_active = active_indices[0]
            if first_active > 0:
                aligned[:first_active] = aligned[first_active]
                flat_counts[:first_active] = flat_counts[first_active]
            for idx in range(first_active + 1, aligned.shape[0]):
                if flat_counts[idx] == 0:
                    aligned[idx] = aligned[idx - 1]
                    flat_counts[idx] = flat_counts[idx - 1]

        return aligned

    @staticmethod
    def _ensure_xy_alignment(mask: np.ndarray, height: int, width: int) -> np.ndarray:
        """Ensure spatial axes align with (height, width)."""
        mask_bool = mask > 0
        if mask_bool.shape == (height, width):
            return mask_bool
        if mask_bool.shape == (width, height):
            return mask_bool.T
        raise ValueError(
            f"Segmentation mask spatial shape {mask.shape} does not match expected {(height, width)}"
        )

    def save_curves(self):
        """Save the computed curves to a CSV file."""
        assert isinstance(self.curves_output_path, str), "Export path must be a string."
        assert self.curves_output_path.endswith('.csv'), "Export path must end with .csv."

        Path(self.curves_output_path).parent.mkdir(parents=True, exist_ok=True)

        for ix, curves in enumerate(self.curves):
            cur_out_path = self.curves_output_path.replace('.csv', '')
            if len(self.curves) > 1:
                cur_out_path += f'_{ix}'
            cur_out_path += '.csv'

            curves_dict = curves.copy()
            curves_dict["Scan Name"] = self.image_data.scan_name
            curves_dict["Segmentation Name"] = self.seg_data.seg_name
            if len(self.curves) > 1:
                curves_dict["Window Index"] = ix
            curves_dict["Time Array"] = self.time_arr

            df = pd.DataFrame(curves_dict)
            df.to_csv(cur_out_path, index=False)
