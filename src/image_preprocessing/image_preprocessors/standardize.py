import numpy as np

from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage

@required_kwargs('arr_to_standardize')
def standardize(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Standardize the pixel data and/or intensities for analysis of an UltrasoundImage object.

    Kwargs:
        arr_to_standardize (str): One of 'both', 'intensities', 'pixel_data' to specify which arrays to standardize.
                                   Default is 'both'.
    """
    arr_to_standardize = kwargs.get('arr_to_standardize', 'both')
    assert arr_to_standardize in ['both', 'intensities', 'pixel_data'], "arr_to_standardize must be one of ['both', 'intensities', 'pixel_data']"
    
    if arr_to_standardize in ['both', 'intensities']:
        mean = np.mean(image_data.intensities_for_analysis)
        std = np.std(image_data.intensities_for_analysis)
        if std > 0:
            image_data.intensities_for_analysis = (image_data.intensities_for_analysis - mean) / std
        else:
            image_data.intensities_for_analysis = image_data.intensities_for_analysis - mean
    if arr_to_standardize in ['both', 'pixel_data']:
        mean = np.mean(image_data.pixel_data)
        std = np.std(image_data.pixel_data)
        if std > 0:
            image_data.pixel_data = (image_data.pixel_data - mean) / std
        else:
            image_data.pixel_data = image_data.pixel_data - mean

    return image_data