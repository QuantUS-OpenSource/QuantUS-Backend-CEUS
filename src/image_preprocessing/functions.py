import numpy as np
from skimage import exposure, filters
from skimage.restoration import denoise_wavelet, estimate_sigma
import cv2

from .decorators import required_kwargs
from ..data_objs.image import UltrasoundImage
from .transforms import resample_to_spacing_2d, resample_to_spacing_3d

@required_kwargs('arr_to_standardize')
def standardize(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Standardize the pixel data and/or intensities for analysis of an UltrasoundImage object.

    Kwargs:
        arr_to_standardize (str): One of 'both', 'intensities', 'pixel_data' to specify which arrays to standardize.
                                   Default is 'both'.
    """
    arr_to_standardize = kwargs.get('arr_to_standardize', 'both')
    assert arr_to_standardize in ['both', 'intensities', 'pixel_data'], "arr_to_standardize must be one of ['both', 'intensities', 'pixel_data']"
    
    if arr_to_standardize in ['both', 'intensities']:
        mean = np.mean(image_data.intensities_for_analysis)
        std = np.std(image_data.intensities_for_analysis)
        if std > 0:
            image_data.intensities_for_analysis = (image_data.intensities_for_analysis - mean) / std
        else:
            image_data.intensities_for_analysis = image_data.intensities_for_analysis - mean
    if arr_to_standardize in ['both', 'pixel_data']:
        mean = np.mean(image_data.pixel_data)
        std = np.std(image_data.pixel_data)
        if std > 0:
            image_data.pixel_data = (image_data.pixel_data - mean) / std
        else:
            image_data.pixel_data = image_data.pixel_data - mean

    return image_data

@required_kwargs('target_vox_size', 'interp')
def resample(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Resample the image data to a new spacing.

    Kwargs:
        target_vox_size: tuple of (z, y, x) spacing in mm to resample the image to.
        interp: interpolation method, one of 'nearest', 'linear', 'cubic'.
    """
    target_vox_size = kwargs['target_vox_size']
    interp = kwargs['interp']

    if image_data.intensities_for_analysis.ndim == 4:
        image_data.pixel_data = resample_to_spacing_3d(image_data.pixel_data, image_data.pixdim, target_vox_size, interp=interp)
        image_data.intensities_for_analysis = resample_to_spacing_3d(image_data.intensities_for_analysis, image_data.pixdim, target_vox_size, interp=interp)
    elif image_data.intensities_for_analysis.ndim == 3:
        image_data.pixel_data = resample_to_spacing_2d(image_data.pixel_data, image_data.pixdim, target_vox_size, interp=interp)
        image_data.intensities_for_analysis = resample_to_spacing_2d(image_data.intensities_for_analysis, image_data.pixdim, target_vox_size, interp=interp)
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
    """
    scale_factor = kwargs.get('scale_factor', 1.0)
    interp = kwargs.get('interp', 'cubic')

    if scale_factor == 1.0:
        return image_data

    # Adjust spacing inversely to scale factor
    target_vox_size = tuple(np.array(image_data.pixdim) / scale_factor)
    return resample(image_data, target_vox_size=target_vox_size, interp=interp)

@required_kwargs('clip_limit', 'tile_grid_size')
def enhance_clahe(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Kwargs:
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of grid for histogram equalization.
    """
    clip_limit = kwargs.get('clip_limit', 3.0)
    tile_grid_size = kwargs.get('tile_grid_size', (8, 8))
    
    volume = image_data.pixel_data
    is_2d = volume.ndim == 2
    if is_2d:
        volume = volume[:, :, np.newaxis]
        
    v_min, v_max = volume.min(), volume.max()
    v_range = v_max - v_min

    enhanced = np.zeros_like(volume)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
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
        
    image_data.pixel_data = enhanced[:, :, 0] if is_2d else enhanced
    return image_data

@required_kwargs('gamma')
def enhance_gamma(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Enhance contrast using Gamma correction.
    
    Kwargs:
        gamma (float): Gamma value.
    """
    gamma = kwargs.get('gamma', 1.0)
    image_data.pixel_data = exposure.adjust_gamma(image_data.pixel_data, gamma)
    return image_data

@required_kwargs('gain')
def enhance_log(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Enhance contrast using Logarithmic correction.
    
    Kwargs:
        gain (float): Gain value.
    """
    gain = kwargs.get('gain', 1.0)
    image_data.pixel_data = exposure.adjust_log(image_data.pixel_data, gain=gain)
    return image_data

@required_kwargs('cutoff', 'gain')
def enhance_sigmoid(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Enhance contrast using Sigmoid correction.
    
    Kwargs:
        cutoff (float): Cutoff value.
        gain (float): Gain value.
    """
    cutoff = kwargs.get('cutoff', 0.5)
    gain = kwargs.get('gain', 10.0)
    image_data.pixel_data = exposure.adjust_sigmoid(image_data.pixel_data, cutoff=cutoff, gain=gain)
    return image_data

@required_kwargs('wavelet', 'sigma_scale')
def denoise_ceus_wavelet(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Gentler wavelet denoising.
    
    Kwargs:
        wavelet (str): Wavelet type.
        sigma_scale (float): Scale factor for noise estimate.
    """
    wavelet = kwargs.get('wavelet', 'db1')
    sigma_scale = kwargs.get('sigma_scale', 0.8)
    
    volume_3d = image_data.pixel_data
    is_2d = volume_3d.ndim == 2
    if is_2d:
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
    
    image_data.pixel_data = denoised[:, :, 0] if is_2d else denoised
    return image_data