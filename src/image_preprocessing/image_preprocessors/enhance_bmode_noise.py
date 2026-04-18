import numpy as np

from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage


@required_kwargs('p_low_percentile', 'p_high_percentile')
def enhance_bmode_noise(image_data: UltrasoundImage, **kwargs) -> UltrasoundImage:
    """
    Enhance B-mode images by clipping and normalizing to [0, 255] using
    per-frame percentiles computed on non-zero pixels.

    Kwargs:
        p_low_percentile (float): Lower percentile for clipping (default 2).
        p_high_percentile (float): Upper percentile for clipping (default 98).
    """
    p_low_pct = float(kwargs.get('p_low_percentile', 2))
    p_high_pct = float(kwargs.get('p_high_percentile', 98))

    if not (0.0 <= p_low_pct <= 100) or not (0.0 <= p_high_pct <= 100.0):
        raise ValueError(
            f"p_low_percentile and p_high_percentile must be within [0, 100]; "
            f"got p_low_percentile={p_low_pct}, p_high_percentile={p_high_pct}"
        )
    
    if p_high_pct <= p_low_pct:
        raise ValueError(
            f"p_low_percentile must be less than p_high_percentile; "
            f"got p_low_percentile={p_low_pct}, p_high_percentile={p_high_pct}"
        )

    pixel_data = image_data.pixel_data  # H x W x Z x T
    if pixel_data.ndim != 4:
        raise ValueError("enhance_bmode_noise expects 4D data (H x W x Z x T).")

    enhanced = np.zeros_like(pixel_data, dtype=np.uint8)

    for t in range(pixel_data.shape[-1]):
        frame = pixel_data[..., t].astype(np.float32)
        nonzero = frame[frame != 0]

        if nonzero.size == 0:
            enhanced[..., t] = 0
            continue

        p_low = np.percentile(nonzero, p_low_pct)
        p_high = np.percentile(nonzero, p_high_pct)

        clipped = np.clip(frame, p_low, p_high)
        enhanced[..., t] = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)

    image_data.pixel_data = enhanced
    return image_data
