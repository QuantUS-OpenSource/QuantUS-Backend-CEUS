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

    frames = image_data.pixel_data
        
    v_min, v_max = frames.min(), frames.max()
    v_range = v_max - v_min

    is_2d = frames.ndim == 3

    enhanced = np.zeros_like(frames, dtype=frames.dtype)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    for z in range(frames.shape[-1]):
        cur_slice = frames[:, :, z] if is_2d else frames[:, :, :, z].astype(np.float32)
        if cur_slice.dtype not in [np.uint8, np.uint16]:
            if v_range > 0:
                slice_to_proc = ((cur_slice - v_min) / v_range * 255).astype(np.uint8)
            else:
                slice_to_proc = cur_slice.astype(np.uint8)
        else:
            slice_to_proc = cur_slice

        if not is_2d:
            for c in range(slice_to_proc.shape[2]):
                slice_to_proc[:, :, c] = clahe.apply(slice_to_proc[:, :, c])
            clahe_result = slice_to_proc
        else:
            clahe_result = clahe.apply(slice_to_proc)

        if cur_slice.dtype not in [np.uint8, np.uint16]:
            if v_range > 0:
                clahe_result = (clahe_result.astype(np.float32) / 255 * v_range + v_min).astype(cur_slice.dtype)
            else: # no dynamic range - keep original values
                clahe_result = cur_slice
        
        if is_2d:
            enhanced[:, :, z] = clahe_result
        else:
            enhanced[:, :, :, z] = clahe_result

    image_data.pixel_data = enhanced
    return image_data
