from skimage import exposure

from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage

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
