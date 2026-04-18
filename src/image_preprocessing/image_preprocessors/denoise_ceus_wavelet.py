import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma

from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage

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

    frames = image_data.pixel_data
    
    v_min, v_max = frames.min(), frames.max()
    v_range = v_max - v_min

    denoised = np.zeros_like(frames, dtype=frames.dtype)

    is_2d = frames.ndim == 3

    for z in range(frames.shape[-1]):
        cur_slice = frames[:, :, z].astype(np.float32) if is_2d else frames[:, :, :, z].astype(np.float32)
        if v_range > 0:
            slice_norm = (cur_slice - v_min) / v_range
        else:
            slice_norm = cur_slice
        
        if is_2d:
            sigma_est = estimate_sigma(slice_norm, average_sigmas=True)
            denoised_slice = denoise_wavelet(
                slice_norm,
                method='BayesShrink',
                mode='soft',
                wavelet=wavelet,
                rescale_sigma=True,
                sigma=sigma_est * sigma_scale
            )
        else:
            denoised_slice = np.zeros_like(slice_norm)
            for c in range(slice_norm.shape[2]):
                sigma_est = estimate_sigma(slice_norm[:, :, c], average_sigmas=True)
                denoised_slice[:, :, c] = denoise_wavelet(
                    slice_norm[:, :, c],
                    method='BayesShrink',
                    mode='soft',
                    wavelet=wavelet,
                    rescale_sigma=True,
                    sigma=sigma_est * sigma_scale
                )
        
        if v_range > 0:
            denoised_slice = denoised_slice * v_range + v_min
        
        if is_2d:
            denoised[:, :, z] = denoised_slice
        else:
            denoised[:, :, :, z] = denoised_slice
    
    for z in range(frames.shape[-1]):
        if is_2d:
            image_data.pixel_data[:, :, z] = denoised[:, :, z]
        else:
            image_data.pixel_data[:, :, :, z] = denoised[:, :, :, z]
    
    return image_data
