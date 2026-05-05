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
    
    # Calculate mean intensity over the mask
    # For 2D frame and 2D mask, frame[mask > 0] is a 1D array of pixels
    # For 3D frame and 3D mask, frame[mask > 0] is a 1D array of pixels
    # Handle 2D mask with 3D frame (slice or broadcast)
    if mask.ndim == 2 and frame.ndim == 3:
        # If the mask is 2D and the frame is 3D, we'll assume the mask applies to all slices
        # or that the user intended a specific slice. For TIC, we usually want the mean of the 3D ROI.
        # However, if it's (H, W) vs (D, H, W), we broadcast.
        tic_val = np.mean(frame[:, mask > 0])
    else:
        tic_val = np.mean(frame[mask > 0])
    
    return ['TIC'], [tic_val]
