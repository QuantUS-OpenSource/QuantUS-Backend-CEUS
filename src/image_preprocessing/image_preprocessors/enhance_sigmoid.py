from skimage import exposure

from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage

@required_kwargs('cutoff', 'gain', 'frame_ix')
def enhance_sigmoid(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Enhance contrast using Sigmoid correction.
    
    Kwargs:
        cutoff (float): Cutoff value.
        gain (float): Gain value.
        frame_ix (int): Frame index to process. If -1, process all frames.
    """
    cutoff = kwargs.get('cutoff', 0.5)
    gain = kwargs.get('gain', 10.0)
    frame_ix = kwargs.get('frame_ix', -1)
    
    if frame_ix == -1:
        image_data.pixel_data = exposure.adjust_sigmoid(image_data.pixel_data, cutoff=cutoff, gain=gain)
    else:
        if image_data.pixel_data.ndim == 3: # 2D case
            image_data.pixel_data[:, :, frame_ix] = exposure.adjust_sigmoid(image_data.pixel_data[:, :, frame_ix], cutoff=cutoff, gain=gain)
        else: # 3D case
            image_data.pixel_data[:, :, :, frame_ix] = exposure.adjust_sigmoid(image_data.pixel_data[:, :, :, frame_ix], cutoff=cutoff, gain=gain)
    return image_data
