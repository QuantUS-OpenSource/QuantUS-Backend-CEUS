import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from typing import Tuple, List

from radiomics import featureextractor

from ...data_objs.image import UltrasoundImage
from .decorators import required_kwargs

logging.getLogger('radiomics').setLevel(logging.ERROR)

@required_kwargs('pyradiomics_config_paths', 'min_intensity', 'binwidth')
def pyradiomics(image_data: UltrasoundImage, frame: np.ndarray, mask: np.ndarray, **kwargs) -> Tuple[List[str], List[np.ndarray]]:
    """
    Extract features using Pyradiomics.
    
    Args:
        frame (np.ndarray): The ultrasound RF frame data.
        mask (np.ndarray): The mask for the region of interest.
        **kwargs: Additional keyword arguments for Pyradiomics feature extraction.
        
    Returns:
        Tuple[List[str], List[np.ndarray]]: A tuple containing the feature names and their corresponding values.
    """
    assert(type(kwargs['pyradiomics_config_paths']) == list), "pyradiomics_config_paths must be a list of paths to PyRadiomics YAML configuration files."
    
    cur_frame = frame.copy() # Avoid modifying the original frame
    cur_mask = mask.copy() # Avoid modifying the original mask
    min_intensity = kwargs['min_intensity']
    bin_width = kwargs['binwidth']

    def _manual_discretize_binwidth(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        seg = image[mask == 1]
        assert seg.size > 0
        # bins start at 1 inside ROI
        bins = np.zeros_like(image, dtype=np.float64)
        bins[mask == 1] = np.floor((seg - min_intensity) / bin_width).astype(np.float64) + 1.0
        return bins

    # Define the affine transformation matrix
    affine = np.eye(cur_frame.ndim + 1)
    reversed_dims = list(reversed(image_data.pixdim))
    for i in range(cur_frame.ndim):
        affine[i, i] = reversed_dims[i]
    origin = affine[:cur_frame.ndim, -1].tolist()
    direction_matrix = affine[:cur_frame.ndim, :cur_frame.ndim]
    spacing = np.linalg.norm(direction_matrix, axis=0).tolist()
    direction = (direction_matrix / spacing).flatten().tolist()

    # HACK: Phantom voxel to ensure proper binning
    if cur_frame.ndim == 3:
        cur_frame[0, 0, 0] = 0
        cur_mask[0, 0, 0] = 1
    elif cur_frame.ndim == 2:
        cur_frame[0, 0] = 0
        cur_mask[0, 0] = 1

    binned_im = _manual_discretize_binwidth(cur_frame, cur_mask)
    image = sitk.GetImageFromArray(binned_im)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    mask_im = sitk.GetImageFromArray(cur_mask)
    mask_im.SetSpacing(spacing)
    mask_im.SetOrigin(origin)
    mask_im.SetDirection(direction)

    feature_names = []; feature_vals = []
    for config_path in kwargs['pyradiomics_config_paths']:
        if not isinstance(config_path, str):
            raise ValueError("Each config path must be a string.")
        try:
            extractor = featureextractor.RadiomicsFeatureExtractor(config_path)
            features = extractor.execute(image, mask_im)
        except Exception as e:
            logging.error(f"Error occurred while extracting features: {e}")
            continue

        for name in features.keys():
            val_type = type(features[name])
            if val_type == list or val_type == dict or val_type == tuple or val_type == str:
                continue
            if "diagnostics" in name:
                continue
            feature_names.append(f"{Path(config_path).stem}_{name}"); feature_vals.append(features[name])

    return feature_names, feature_vals

@required_kwargs()
def tic(image_data: UltrasoundImage, frame: np.ndarray, mask: np.ndarray, **kwargs) -> Tuple[List[str], List[np.ndarray]]:
    """
    Extract Time Intensity Curve (TIC) features from the ultrasound image data.
    
    Args:
        image_data (UltrasoundImage): The ultrasound image data object.
        frame (np.ndarray): The ultrasound RF frame data.
        mask (np.ndarray): The mask for the region of interest.
        **kwargs: Additional keyword arguments (not used).
        
    Returns:
        Tuple[List[str], List[np.ndarray]]: A tuple containing the feature names and their corresponding values.
    """
    assert isinstance(image_data, UltrasoundImage), "image_data must be an instance of UltrasoundImage"
    
    tic_curve = np.mean(frame[mask > 0], axis=0)
    return ['TIC'], [tic_curve]
