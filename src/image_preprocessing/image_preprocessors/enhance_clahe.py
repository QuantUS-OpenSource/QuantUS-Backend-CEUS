import cv2
import numpy as np

from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage

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

        clahe_result = clahe.apply(slice_to_proc)

        if slice_2d.dtype not in [np.uint8, np.uint16]:
            if v_range > 0:
                enhanced[:, :, z] = (clahe_result.astype(np.float32) / 255 * v_range + v_min).astype(volume.dtype)
            else: # no dynamic range - keep original values
                enhanced[:, :, z] = slice_2d
        else:
            enhanced[:, :, z] = clahe_result

    image_data.pixel_data = enhanced[:, :, 0] if is_2d else enhanced
    return image_data
