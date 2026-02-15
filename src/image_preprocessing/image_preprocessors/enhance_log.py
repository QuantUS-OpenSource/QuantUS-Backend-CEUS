from skimage import exposure

from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage

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
