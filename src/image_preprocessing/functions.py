import numpy as np
from skimage import exposure, filters
from skimage.restoration import denoise_wavelet, estimate_sigma
import cv2

from .decorators import required_kwargs
from ..data_objs.image import UltrasoundImage
from .transforms import resample_to_spacing_2d, resample_to_spacing_3d

def _get_volume_indices(image_data: UltrasoundImage, **kwargs):
    """Helper to determine which volumes to process based on volume_ix."""
    volume_ix = kwargs.get('volume_ix')
    if volume_ix is not None:
        return [volume_ix]
    
    # 2D+time: (t, y, x)
    if len(image_data.pixdim) == 2:
        return range(image_data.pixel_data.shape[0])
    # 3D+time: (x, y, z, t)
    elif len(image_data.pixdim) == 3:
        if image_data.pixel_data.ndim == 4:
            return range(image_data.pixel_data.shape[3])
    return [0]

def _get_volume_slice_idx(image_data: UltrasoundImage, v_ix: int):
    """Helper to get the slice index for a specific volume."""
    if len(image_data.pixdim) == 2:
        return (v_ix, slice(None), slice(None))
    elif len(image_data.pixdim) == 3:
        if image_data.pixel_data.ndim == 4:
            return (slice(None), slice(None), slice(None), v_ix)
    return slice(None)

@required_kwargs('arr_to_standardize')
def standardize(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Standardize the pixel data and/or intensities for analysis of an UltrasoundImage object.

    Kwargs:
        arr_to_standardize (str): One of 'both', 'intensities', 'pixel_data' to specify which arrays to standardize.
                                   Default is 'both'.
        volume_ix (int): Optional index of the volume to standardize. If None, standardizes across all volumes.
    """
    arr_to_standardize = kwargs.get('arr_to_standardize', 'both')
    volume_ix = kwargs.get('volume_ix')
    assert arr_to_standardize in ['both', 'intensities', 'pixel_data'], "arr_to_standardize must be one of ['both', 'intensities', 'pixel_data']"
    
    def apply_standardization(arr, v_ix):
        if v_ix is not None:
            # Standardize a single volume
            idx = _get_volume_slice_idx(image_data, v_ix)
            mean = np.mean(arr[idx])
            std = np.std(arr[idx])
            if std > 0:
                arr[idx] = (arr[idx] - mean) / std
            else:
                arr[idx] = arr[idx] - mean
        else:
            # Standardize across all volumes
            mean = np.mean(arr)
            std = np.std(arr)
            if std > 0:
                arr[:] = (arr - mean) / std
            else:
                arr[:] = arr - mean
        return arr

    if arr_to_standardize in ['both', 'intensities']:
        apply_standardization(image_data.intensities_for_analysis, volume_ix)
    if arr_to_standardize in ['both', 'pixel_data']:
        apply_standardization(image_data.pixel_data, volume_ix)

    return image_data

@required_kwargs('target_vox_size', 'interp')
def resample(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Resample the image data to a new spacing.

    Kwargs:
        target_vox_size: tuple of (z, y, x) spacing in mm to resample the image to.
        interp: interpolation method, one of 'nearest', 'linear', 'cubic'.
        volume_ix (int): Optional index of the volume to resample.
    """
    target_vox_size = kwargs['target_vox_size']
    interp = kwargs['interp']
    volume_ix = kwargs.get('volume_ix')

    if image_data.intensities_for_analysis.ndim == 4:
        image_data.pixel_data = resample_to_spacing_3d(image_data.pixel_data, image_data.pixdim, target_vox_size, interp=interp, volume_ix=volume_ix)
        image_data.intensities_for_analysis = resample_to_spacing_3d(image_data.intensities_for_analysis, image_data.pixdim, target_vox_size, interp=interp, volume_ix=volume_ix)
    elif image_data.intensities_for_analysis.ndim == 3:
        image_data.pixel_data = resample_to_spacing_2d(image_data.pixel_data, image_data.pixdim, target_vox_size, interp=interp, volume_ix=volume_ix)
        image_data.intensities_for_analysis = resample_to_spacing_2d(image_data.intensities_for_analysis, image_data.pixdim, target_vox_size, interp=interp, volume_ix=volume_ix)
    else:
        raise ValueError("Image data must be either 3D or 4D for resampling.")

    image_data.extras_dict['original_spacing'] = image_data.pixdim
    image_data.pixdim = target_vox_size

    return image_data

@required_kwargs('scale_factor', 'interp')
def enhance_spatial_resolution(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Enhance spatial resolution by upsampling the image (increasing pixel count).
    
    Kwargs:
        scale_factor: Multiplier for the current resolution (e.g., 2.0 to double pixels)
        interp: Interpolation method ('nearest', 'linear', 'cubic')
        volume_ix (int): Optional index of the volume to enhance.
    """
    scale_factor = kwargs.get('scale_factor', 1.0)
    interp = kwargs.get('interp', 'cubic')
    volume_ix = kwargs.get('volume_ix')

    if scale_factor == 1.0:
        return image_data

    # Adjust spacing inversely to scale factor
    target_vox_size = tuple(np.array(image_data.pixdim) / scale_factor)
    return resample(image_data, target_vox_size=target_vox_size, interp=interp, volume_ix=volume_ix)

@required_kwargs('clip_limit', 'tile_grid_size')
def enhance_clahe(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Kwargs:
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of grid for histogram equalization.
        volume_ix (int): Optional index of the volume to enhance.
    """
    clip_limit = kwargs.get('clip_limit', 3.0)
    tile_grid_size = kwargs.get('tile_grid_size', (8, 8))
    volume_ix = kwargs.get('volume_ix')
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v_indices = _get_volume_indices(image_data, volume_ix=volume_ix)
    
    # If we are only processing one volume for a preview, 
    # we might want to return a smaller object or just modify the slice.
    # For now, let's modify the slices in place if possible.
    
    for v_idx in v_indices:
        vol_idx = _get_volume_slice_idx(image_data, v_idx)
        volume = image_data.pixel_data[vol_idx]
        
        is_2d_vol = volume.ndim == 2
        if is_2d_vol:
            volume = volume[:, :, np.newaxis]
            
        v_min, v_max = volume.min(), volume.max()
        v_range = v_max - v_min
        enhanced = np.zeros_like(volume)
        
        for z in range(volume.shape[2]):
            slice_2d = volume[:, :, z]
            if slice_2d.dtype not in [np.uint8, np.uint16]:
                if v_range > 0:
                    slice_to_proc = ((slice_2d - v_min) / v_range * 255).astype(np.uint8)
                else:
                    slice_to_proc = slice_2d.astype(np.uint8)
            else:
                slice_to_proc = slice_2d
                
            enhanced[:, :, z] = clahe.apply(slice_to_proc)
        
        image_data.pixel_data[vol_idx] = enhanced[:, :, 0] if is_2d_vol else enhanced
        
    return image_data

@required_kwargs('gamma')
def enhance_gamma(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Enhance contrast using Gamma correction.
    
    Kwargs:
        gamma (float): Gamma value.
        volume_ix (int): Optional index of the volume to enhance.
    """
    gamma = kwargs.get('gamma', 1.0)
    volume_ix = kwargs.get('volume_ix')
    v_indices = _get_volume_indices(image_data, volume_ix=volume_ix)
    
    for v_idx in v_indices:
        idx = _get_volume_slice_idx(image_data, v_idx)
        image_data.pixel_data[idx] = exposure.adjust_gamma(image_data.pixel_data[idx], gamma)
        
    return image_data

@required_kwargs('gain')
def enhance_log(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Enhance contrast using Logarithmic correction.
    
    Kwargs:
        gain (float): Gain value.
        volume_ix (int): Optional index of the volume to enhance.
    """
    gain = kwargs.get('gain', 1.0)
    volume_ix = kwargs.get('volume_ix')
    v_indices = _get_volume_indices(image_data, volume_ix=volume_ix)
    
    for v_idx in v_indices:
        idx = _get_volume_slice_idx(image_data, v_idx)
        image_data.pixel_data[idx] = exposure.adjust_log(image_data.pixel_data[idx], gain=gain)
        
    return image_data

@required_kwargs('cutoff', 'gain')
def enhance_sigmoid(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Enhance contrast using Sigmoid correction.
    
    Kwargs:
        cutoff (float): Cutoff value.
        gain (float): Gain value.
        volume_ix (int): Optional index of the volume to enhance.
    """
    cutoff = kwargs.get('cutoff', 0.5)
    gain = kwargs.get('gain', 10.0)
    volume_ix = kwargs.get('volume_ix')
    v_indices = _get_volume_indices(image_data, volume_ix=volume_ix)
    
    for v_idx in v_indices:
        idx = _get_volume_slice_idx(image_data, v_idx)
        image_data.pixel_data[idx] = exposure.adjust_sigmoid(image_data.pixel_data[idx], cutoff=cutoff, gain=gain)
        
    return image_data

@required_kwargs('wavelet', 'sigma_scale')
def denoise_ceus_wavelet(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Gentler wavelet denoising.
    
    Kwargs:
        wavelet (str): Wavelet type.
        sigma_scale (float): Scale factor for noise estimate.
        volume_ix (int): Optional index of the volume to denoise.
    """
    wavelet = kwargs.get('wavelet', 'db1')
    sigma_scale = kwargs.get('sigma_scale', 0.8)
    volume_ix = kwargs.get('volume_ix')
    
    v_indices = _get_volume_indices(image_data, volume_ix=volume_ix)
    
    for v_idx in v_indices:
        vol_idx = _get_volume_slice_idx(image_data, v_idx)
        volume_3d = image_data.pixel_data[vol_idx]
        
        is_2d_vol = volume_3d.ndim == 2
        if is_2d_vol:
            volume_3d = volume_3d[:, :, np.newaxis]

        v_min, v_max = volume_3d.min(), volume_3d.max()
        v_range = v_max - v_min

        denoised = np.zeros_like(volume_3d, dtype=np.float32)
        
        for z in range(volume_3d.shape[2]):
            slice_2d = volume_3d[:, :, z].astype(np.float32)
            if v_range > 0:
                slice_norm = (slice_2d - v_min) / v_range
            else:
                slice_norm = slice_2d
            
            sigma_est = estimate_sigma(slice_norm, average_sigmas=True)
            denoised_slice = denoise_wavelet(
                slice_norm,
                method='BayesShrink',
                mode='soft',
                wavelet=wavelet,
                rescale_sigma=True,
                sigma=sigma_est * sigma_scale
            )
            
            if v_range > 0:
                denoised[:, :, z] = denoised_slice * v_range + v_min
            else:
                denoised[:, :, z] = denoised_slice
        
        image_data.pixel_data[vol_idx] = denoised[:, :, 0] if is_2d_vol else denoised
        
    return image_data