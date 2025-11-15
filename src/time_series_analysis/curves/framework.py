import numpy as np
import pandas as pd
from typing import Dict, List
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
        if image_data.intensities_for_analysis.ndim == 4: # 3D + time
            self.time_arr = np.arange(self.image_data.intensities_for_analysis.shape[3]) * frame_rate
        elif image_data.intensities_for_analysis.ndim == 3: # 2D + time
            self.time_arr = np.arange(self.image_data.intensities_for_analysis.shape[0]) * frame_rate
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
                self.extract_frame_features(frame_data, self.seg_data.seg_mask, frame_ix)

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