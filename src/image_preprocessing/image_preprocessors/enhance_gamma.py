from skimage import exposure

from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage

@required_kwargs('gamma', 'frame_ix')
def enhance_gamma(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Enhance contrast using Gamma correction.
    
    Kwargs:
        gamma (float): Gamma value.
        frame_ix (int): Frame index to process. If -1, process all frames.
    """
    gamma = kwargs.get('gamma', 1.0)
    frame_ix = kwargs.get('frame_ix', -1)

    if frame_ix == -1:
        image_data.pixel_data = exposure.adjust_gamma(image_data.pixel_data, gamma)
    else:
        if image_data.pixel_data.ndim == 3: # 2D case
            image_data.pixel_data[:, :, frame_ix] = exposure.adjust_gamma(image_data.pixel_data[:, :, frame_ix], gamma)
        else: # 3D case
            image_data.pixel_data[:, :, :, frame_ix] = exposure.adjust_gamma(image_data.pixel_data[:, :, :, frame_ix], gamma)

    return image_data
