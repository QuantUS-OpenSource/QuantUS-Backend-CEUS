from skimage import exposure

from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage

@required_kwargs('gain', 'frame_ix')
def enhance_log(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Enhance contrast using Logarithmic correction.
    
    Kwargs:
        gain (float): Gain value.
        frame_ix (int): Frame index to process. If -1, process all frames.
    """
    gain = kwargs.get('gain', 1.0)
    frame_ix = kwargs.get('frame_ix', -1)

    if frame_ix == -1:
        image_data.pixel_data = exposure.adjust_log(image_data.pixel_data, gain=gain)
    else:
        if image_data.pixel_data.ndim == 3: # 2D case
            image_data.pixel_data[:, :, frame_ix] = exposure.adjust_log(image_data.pixel_data[:, :, frame_ix], gain=gain)
        else: # 3D case
            image_data.pixel_data[:, :, :, frame_ix] = exposure.adjust_log(image_data.pixel_data[:, :, :, frame_ix], gain=gain)

    return image_data
