import numpy as np
from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage

def _compute_ceus_noise_floor(image_data: UltrasoundImage, 
                              n_ref_frames: int, 
                              noise_std_multiplier: float) -> float:
    """
    Compute the noise floor (p_low scalar) from pre-contrast frames of a 4D CEUS scan.
    Returns the noise floor value as a float.
    """
    pixel_data = image_data.pixel_data
    if pixel_data.ndim != 4:
        raise ValueError("compute_ceus_noise_floor expects 4D data (H x W x Z x T).")

    n_frames = pixel_data.shape[-1]
    if n_ref_frames >= n_frames or n_ref_frames <= 0:
        raise ValueError(f"n_ref_frames ({n_ref_frames}) must be less than total frames ({n_frames}) and greater than 0.")

    ref_frames = pixel_data[..., :n_ref_frames]
    ref_nonzero = ref_frames[ref_frames != 0]
    if ref_nonzero.size == 0:
        raise ValueError("Pre-contrast reference frames contain no non-zero values.")

    noise_mean = np.mean(ref_nonzero)
    noise_std = np.std(ref_nonzero)
    return float(noise_mean + noise_std_multiplier * noise_std)


@required_kwargs('n_ref_frames', 'noise_std_multiplier', 'p_high_percentile')
def enhance_ceus_noise_norm(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Normalize 3D frames (H x W x Z x T) using a precomputed noise floor.

    Kwargs:
        n_ref_frames (int): Number of pre-contrast frames for noise estimation.
        noise_std_multiplier (float): Multiplier k for noise std.
        p_high_percentile (float): Percentile for signal ceiling on non-zero pixels.
    """
    n_ref_frames = int(kwargs.get('n_ref_frames', 10))
    noise_std_multiplier = float(kwargs.get('noise_std_multiplier', 1.0))
    p_high_percentile = float(kwargs.get('p_high_percentile', 99.5))

    p_low = _compute_ceus_noise_floor(image_data, n_ref_frames, noise_std_multiplier)

    pixel_data = image_data.pixel_data  # H x W x Z x T
    if pixel_data.ndim != 4:
        raise ValueError("enhance_ceus_noise_norm expects 4D data (H x W x Z x T).")

    enhanced = np.zeros_like(pixel_data, dtype=np.uint8)
    for t in range(pixel_data.shape[-1]):
        frame = pixel_data[..., t].astype(np.float32)
        nonzero_vals = frame[frame != 0]
        if nonzero_vals.size == 0:
            enhanced[..., t] = 0
            continue
        p_high = np.percentile(nonzero_vals, p_high_percentile)
        if p_high <= p_low:
            enhanced[..., t] = 0
            continue
        clipped = np.clip(frame, p_low, p_high)
        enhanced[..., t] = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)

    image_data.pixel_data = enhanced
    return image_data