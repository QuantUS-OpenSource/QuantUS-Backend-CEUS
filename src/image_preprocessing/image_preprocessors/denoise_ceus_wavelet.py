import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma
from ...data_objs.image import UltrasoundImage
from ..decorators import required_kwargs

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
