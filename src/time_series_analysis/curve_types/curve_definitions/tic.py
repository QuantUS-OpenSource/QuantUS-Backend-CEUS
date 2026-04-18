import numpy as np
from typing import Tuple, List

from ....data_objs.image import UltrasoundImage
from ..decorators import required_kwargs

@required_kwargs()
def tic(image_data: UltrasoundImage, frame: np.ndarray, mask: np.ndarray, **kwargs) -> Tuple[List[str], List[np.ndarray]]:
    """
    Extract Time Intensity Curve (TIC) features from the ultrasound image data.
    
    Args:
        image_data (UltrasoundImage): The ultrasound image data object.
        frame (np.ndarray): The ultrasound RF frame data.
        mask (np.ndarray): The mask for the region of interest.
        **kwargs: Additional keyword arguments (not used).
        
    Returns:
        Tuple[List[str], List[np.ndarray]]: A tuple containing the feature names and their corresponding values.
    """
    assert isinstance(image_data, UltrasoundImage), "image_data must be an instance of UltrasoundImage"
    
    tic_curve = np.mean(frame[mask > 0], axis=0)
    return ['TIC'], [tic_curve]
