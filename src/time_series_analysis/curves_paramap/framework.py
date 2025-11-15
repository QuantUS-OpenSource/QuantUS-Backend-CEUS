import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

from ...data_objs.image import UltrasoundImage
from ...data_objs.seg import CeusSeg
from ..curve_types.functions import *
from ..curves.framework import CurvesAnalysis

class_name = "CurvesParamap"

class CurvesParamapAnalysis(CurvesAnalysis):
    """
    Class to complete RF analysis via the sliding window technique
    and generate a corresponding parametric map.
    """
    required_kwargs = ['ax_vox_len', 'sag_vox_len', 'cor_vox_len', 'ax_vox_ovrlp', 'sag_vox_ovrlp', 'cor_vox_ovrlp']
    
    def __init__(self, image_data: UltrasoundImage, seg: CeusSeg, 
                 curve_groups: List[str], **kwargs):
        super().__init__(image_data, seg, curve_groups, **kwargs)
        assert 'ax_vox_len' in kwargs.keys(), 'Must include axial voxel length for parametric map computation'
        assert 'sag_vox_len' in kwargs.keys(), 'Must include sagittal voxel length for parametric map computation'
        assert 'ax_vox_ovrlp' in kwargs.keys(), 'Must include axial voxel overlap for parametric map computation'
        assert 'sag_vox_ovrlp' in kwargs.keys(), 'Must include sagittal voxel length for parametric map computation'
        assert 'cor_vox_len' in kwargs.keys() or image_data.intensities_for_analysis.ndim == 3, 'Must include coronal voxel length for parametric map computation'
        assert 'cor_vox_ovrlp' in kwargs.keys() or image_data.intensities_for_analysis.ndim == 3, 'Must include coronal voxel overlap for parametric map computation'

        if image_data.intensities_for_analysis.ndim == 4:
            self.cor_vox_len = kwargs['cor_vox_len']        # mm
            self.cor_vox_ovrlp = kwargs['cor_vox_ovrlp']    # %
            self.cor_res = self.image_data.pixdim[2]        # mm/px

        self.ax_vox_len = kwargs['ax_vox_len']              # mm
        self.sag_vox_len = kwargs['sag_vox_len']            # mm
        self.ax_vox_ovrlp = kwargs['ax_vox_ovrlp']          # %
        self.sag_vox_ovrlp = kwargs['sag_vox_ovrlp']        # %
        self.ax_res = self.image_data.pixdim[0]             # mm/px
        self.sag_res = self.image_data.pixdim[1]            # mm/px

        self.windows = self.generate_windows()              # Generate sliding windows for the parametric map
        self.curves: List[Dict[str, List[float]]] = []  # List to hold computed curves for each window

    def generate_windows(self):
        """Generate sliding windows for the parametric map.

        Returns:
            List[tuple]: List of tuples containing window coordinates.
        """
        windows = []
        ax_step = int(self.ax_vox_len * (1 - (self.ax_vox_ovrlp / 100)) / self.ax_res)
        sag_step = int(self.sag_vox_len * (1 - (self.sag_vox_ovrlp / 100)) / self.sag_res)
        if hasattr(self, 'cor_vox_len'):
            cor_step = int(self.cor_vox_len * (1 - (self.cor_vox_ovrlp / 100)) / self.cor_res)
        
        # Create minimum and maximum indices for the sliding windows based on the segmentation mask
        mask_ixs = np.where(self.seg_data.seg_mask > 0)
        if hasattr(self, 'cor_vox_len'):
            min_ax, max_ax = np.min(mask_ixs[2]), np.max(mask_ixs[2])
            min_sag, max_sag = np.min(mask_ixs[0]), np.max(mask_ixs[0])
            min_cor, max_cor = np.min(mask_ixs[1]), np.max(mask_ixs[1])
        else:
            min_ax, max_ax = np.min(mask_ixs[0]), np.max(mask_ixs[0])
            min_sag, max_sag = np.min(mask_ixs[1]), np.max(mask_ixs[1])

        for ax_start in range(min_ax, max_ax, ax_step):
            for sag_start in range(min_sag, max_sag, sag_step):
                if hasattr(self, 'cor_vox_len'):
                    for cor_start in range(min_cor, max_cor, cor_step):
                            # Determine if window is inside analysis volume
                            mask_vals = self.seg_data.seg_mask[
                                sag_start : (sag_start + sag_step),
                                cor_start : (cor_start + cor_step),
                                ax_start : (ax_start + ax_step),
                            ]
                            
                            # Define Percentage Threshold
                            total_number_of_elements_in_region = mask_vals.size
                            number_of_ones_in_region = len(np.where(mask_vals == True)[0])
                            percentage_ones = number_of_ones_in_region / total_number_of_elements_in_region

                            if percentage_ones > 0.2:
                                windows.append((ax_start, sag_start, cor_start, 
                                                ax_start + ax_step, sag_start + sag_step, cor_start + cor_step))
                else:
                    # Determine if window is inside analysis volume
                    mask_vals = self.seg_data.seg_mask[
                        ax_start : (ax_start + ax_step),
                        sag_start : (sag_start + sag_step),
                    ]
                    
                    # Define Percentage Threshold
                    total_number_of_elements_in_region = mask_vals.size
                    number_of_ones_in_region = len(np.where(mask_vals == True)[0])
                    percentage_ones = number_of_ones_in_region / total_number_of_elements_in_region

                    if percentage_ones > 0.2:
                        windows.append((ax_start, sag_start, 
                                        ax_start + ax_step, sag_start + sag_step))

        return windows
        
    def compute_curves(self):
        """Compute UTC parameters for each window in the ROI, creating a parametric map."""
        if self.image_data.intensities_for_analysis.ndim == 4: # 3D + time
            for frame_ix, frame in tqdm(enumerate(range(self.image_data.intensities_for_analysis.shape[3])), 
                                        desc="Computing curves", total=self.image_data.intensities_for_analysis.shape[3]):
                frame_data = self.image_data.intensities_for_analysis[:, :, :, frame]
                for window_ix, window in enumerate(self.windows):
                    mask = np.zeros_like(self.seg_data.seg_mask)
                    ax_start, sag_start, cor_start, ax_end, sag_end, cor_end = window
                    mask[sag_start:sag_end+1, 
                            cor_start:cor_end+1, 
                            ax_start:ax_end+1] = 1
                    self.extract_frame_features(frame_data, mask, frame_ix, window_ix)
        elif self.image_data.intensities_for_analysis.ndim == 3: # 2D + time
            for frame_ix, frame in tqdm(enumerate(range(self.image_data.intensities_for_analysis.shape[0])), 
                                        desc="Computing curves", total=self.image_data.intensities_for_analysis.shape[0]):
                frame_data = self.image_data.intensities_for_analysis[frame]
                for window_ix, window in enumerate(self.windows):
                    mask = np.zeros_like(self.seg_data.seg_mask)
                    ax_start, sag_start, ax_end, sag_end = window
                    mask[ax_start:ax_end+1, sag_start:sag_end+1] = 1
                    self.extract_frame_features(frame_data, mask, frame_ix, window_ix)
        else:
            raise ValueError("Image data must be either 2D+time or 3D+time.")

        if self.curves_output_path:
            self.save_curves()

    def extract_frame_features(self, frame: np.ndarray, mask: np.ndarray, frame_ix: int, window_ix: int):
        """Compute parametric map values for a single frame."""
        if len(self.curves) < window_ix:
            raise ValueError(f"Window index {window_ix} exceeds the number of computed curves.")
        elif len(self.curves) == window_ix:
            self.curves.append({})
            window = self.windows[window_ix]
            self.curves[window_ix]['Window-Axial Start Pix'] = window[0]
            self.curves[window_ix]['Window-Sagittal Start Pix'] = window[1]
            if hasattr(self, 'cor_vox_len'):
                self.curves[window_ix]['Window-Coronal Start Pix'] = window[2]
                self.curves[window_ix]['Window-Axial End Pix'] = window[3]
                self.curves[window_ix]['Window-Sagittal End Pix'] = window[4]
                self.curves[window_ix]['Window-Coronal End Pix'] = window[5]
            else:
                self.curves[window_ix]['Window-Axial End Pix'] = window[2]
                self.curves[window_ix]['Window-Sagittal End Pix'] = window[3]

        for curve_group in self.curve_groups:
            curve_function = self.curve_funcs[curve_group]
            curve_names, vals = curve_function(self.image_data, frame, mask, **self.analysis_kwargs)

            for name, val in zip(curve_names, vals):
                if name not in self.curves[window_ix]:
                    self.curves[window_ix][name] = []
                self.curves[window_ix][name].append(val)
