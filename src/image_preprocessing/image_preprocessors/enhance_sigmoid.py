from skimage import exposure
from ...data_objs.image import UltrasoundImage
from ..decorators import required_kwargs

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
