import numpy as np

from ...data_objs.image import UltrasoundImage
from ..decorators import required_kwargs
from .resample import resample

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
    assert scale_factor > 0, 'Scale factor must be positive'
    target_vox_size = tuple(np.array(image_data.pixdim) / scale_factor)
    return resample(image_data, target_vox_size=target_vox_size, interp=interp)
