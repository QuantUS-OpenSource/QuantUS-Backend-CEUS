from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage
from ..transforms import resample_to_spacing_2d, resample_to_spacing_3d

@required_kwargs('target_vox_size', 'interp')
def resample(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Resample the image data to a new spacing.

    Kwargs:
        target_vox_size: tuple of (z, y, x) spacing in mm to resample the image to.
        interp: interpolation method, one of 'nearest', 'linear', 'cubic'.
    """
    target_vox_size = kwargs['target_vox_size']
    interp = kwargs['interp']

    if image_data.intensities_for_analysis.ndim == 4:
        image_data.pixel_data = resample_to_spacing_3d(image_data.pixel_data, image_data.pixdim, target_vox_size, interp=interp)
        image_data.intensities_for_analysis = resample_to_spacing_3d(image_data.intensities_for_analysis, image_data.pixdim, target_vox_size, interp=interp)
    elif image_data.intensities_for_analysis.ndim == 3:
        image_data.pixel_data = resample_to_spacing_2d(image_data.pixel_data, image_data.pixdim, target_vox_size, interp=interp)
        image_data.intensities_for_analysis = resample_to_spacing_2d(image_data.intensities_for_analysis, image_data.pixdim, target_vox_size, interp=interp)
    else:
        raise ValueError("Image data must be either 3D or 4D for resampling.")

    image_data.extras_dict['original_spacing'] = image_data.pixdim
    image_data.pixdim = target_vox_size

    return image_data
