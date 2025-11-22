import itertools
import warnings
from typing import Optional

import numpy as np

def _ensure_2d_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return a boolean mask with shape (height, width), transposing if needed."""
    mask_bool = mask > 0
    if mask_bool.shape == (height, width):
        return mask_bool
    if mask_bool.shape == (width, height):
        return mask_bool.T
    raise ValueError(
        f"Segmentation mask spatial shape {mask.shape} does not match expected {(height, width)}"
    )

def _normalize_seg_mask(seg_mask: np.ndarray, height: int, width: int, n_frames: Optional[int]):
    """
    Convert seg_mask to per-frame and union masks aligned with pixel_data.

    Returns
    -------
    per_frame_mask : np.ndarray or None
        Boolean mask with shape (n_frames, height, width) when n_frames is provided.
    union_mask : np.ndarray
        Boolean mask with shape (height, width) representing the ROI union over time.
    """
    mask = np.asarray(seg_mask)
    mask = np.squeeze(mask)

    if mask.ndim == 2:
        mask_2d = _ensure_2d_mask(mask, height, width)
        per_frame = (
            np.repeat(mask_2d[np.newaxis, :, :], n_frames, axis=0) if n_frames is not None else None
        )
        return per_frame, mask_2d

    if mask.ndim == 3:
        mask_3d = mask > 0
        for perm in itertools.permutations(range(3)):
            candidate = np.transpose(mask_3d, perm)
            if candidate.shape[1:] == (height, width):
                if n_frames is not None and candidate.shape[0] != n_frames:
                    mask_frames = candidate.shape[0]
                    if mask_frames < n_frames:
                        warnings.warn(
                            f"Segmentation mask has {mask_frames} frames; padding to match pixel data frames ({n_frames}).",
                            RuntimeWarning,
                        )
                        pad = np.repeat(candidate[mask_frames - 1:mask_frames], n_frames - mask_frames, axis=0)
                        candidate = np.concatenate([candidate, pad], axis=0)
                    else:
                        warnings.warn(
                            f"Segmentation mask has {mask_frames} frames; truncating to match pixel data frames ({n_frames}).",
                            RuntimeWarning,
                        )
                        candidate = candidate[:n_frames]
                per_frame = candidate if n_frames is not None else None
                union_mask = np.any(candidate, axis=0)
                return per_frame, union_mask

    raise ValueError(f"Unsupported segmentation mask shape {seg_mask.shape}")

def generate_t0_map(pixel_data, seg_mask, threshold=150, start_frame=50, end_frame=250, min_consecutive_frames=1):
    """
    Generate T0 map showing when pixels first reach threshold intensity.

    The T0 map assigns higher values to pixels that light up earlier, creating a
    heatmap where brighter regions indicate earlier contrast arrival.

    Parameters
    ----------
    pixel_data : numpy.ndarray
        Image data with shape (n_frames, height, width)
    seg_mask : numpy.ndarray
        Binary segmentation mask with shape (height, width) or a per-frame mask
        with shape (n_frames, height, width) (motion-compensated ROI).
        Only pixels where seg_mask > 0 will be analyzed
    threshold : float, optional
        Intensity threshold for detecting contrast arrival (default: 150)
        Pixels must exceed this value to be considered "lit up"
    start_frame : int, optional
        First frame to analyze (default: 50)
    end_frame : int, optional
        Last frame to analyze (default: 250)
    min_consecutive_frames : int, optional
        Minimum number of consecutive frames a pixel must be above threshold
        to be considered truly enhanced (default: 1). Higher values (e.g., 3)
        help filter out noise and spurious detections.

    Returns
    -------
    t0_map : numpy.ndarray
        Map with shape (height, width) where each pixel value represents
        the reverse time index when that pixel first exceeded the threshold.
        Higher values = earlier activation.
        Pixels that never exceeded threshold will have value 0.

    Examples
    --------
    >>> t0_map = generate_t0_map(image_data.pixel_data, seg_data.seg_mask,
    ...                          threshold=150, start_frame=50, end_frame=250,
    ...                          min_consecutive_frames=3)
    """
    n_frames, height, width = pixel_data.shape

    # Validate frame range
    if end_frame > n_frames:
        end_frame = n_frames
        print(f"Warning: end_frame adjusted to {n_frames} (max available)")

    # Initialize T0 map with zeros
    t0_map = np.zeros((height, width), dtype=np.float32)

    # Prepare per-frame ROI masks (handles 2D or motion-compensated 3D masks)
    per_frame_mask, _ = _normalize_seg_mask(seg_mask, height, width, n_frames=n_frames)

    # Track consecutive frames above threshold for each pixel
    consecutive_count = np.zeros((height, width), dtype=np.int32)

    # Loop through frames from start to end
    for i in range(start_frame, end_frame):
        # Get current frame
        current_frame = pixel_data[i, :, :]
        frame_mask = per_frame_mask[i]

        # Check which pixels in ROI exceed threshold
        above_threshold = (current_frame >= threshold) & frame_mask

        # Update consecutive count: increment if above threshold, reset if not
        consecutive_count = np.where(above_threshold, consecutive_count + 1, 0)

        # Find pixels that just reached min_consecutive_frames AND haven't been assigned yet
        newly_detected = (consecutive_count == min_consecutive_frames) & (t0_map == 0)

        # Assign reverse time index: earlier frames get higher values
        # Use the frame when it first crossed (i - min_consecutive_frames + 1)
        if min_consecutive_frames > 1:
            first_cross_frame = i - min_consecutive_frames + 1
            t0_map[newly_detected] = end_frame - first_cross_frame
        else:
            t0_map[newly_detected] = end_frame - i

    return t0_map

def get_t0_statistics(t0_map, seg_mask):
    """
    Calculate statistics from the T0 map within the ROI.

    Parameters
    ----------
    t0_map : numpy.ndarray
        T0 map generated by generate_t0_map()
    seg_mask : numpy.ndarray
        Segmentation mask defining the ROI

    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'mean_t0': Mean T0 value in ROI (excluding zeros)
        - 'median_t0': Median T0 value in ROI (excluding zeros)
        - 'std_t0': Standard deviation of T0 values
        - 'min_t0': Minimum T0 value (excluding zeros)
        - 'max_t0': Maximum T0 value
        - 'coverage': Percentage of ROI pixels that were activated
    """
    _, roi_mask = _normalize_seg_mask(seg_mask, t0_map.shape[0], t0_map.shape[1], n_frames=None)
    t0_in_roi = t0_map[roi_mask]

    # Get non-zero values (activated pixels)
    activated = t0_in_roi[t0_in_roi > 0]

    stats = {
        'mean_t0': np.mean(activated) if len(activated) > 0 else 0,
        'median_t0': np.median(activated) if len(activated) > 0 else 0,
        'std_t0': np.std(activated) if len(activated) > 0 else 0,
        'min_t0': np.min(activated) if len(activated) > 0 else 0,
        'max_t0': np.max(activated) if len(activated) > 0 else 0,
        'coverage': (len(activated) / len(t0_in_roi) * 100) if len(t0_in_roi) > 0 else 0
    }

    return stats

def mask_t0_map(t0_map, seg_mask):
    """
    Create a masked version of T0 map showing only the ROI region.

    Pixels outside the ROI will be set to NaN for proper visualization.

    Parameters
    ----------
    t0_map : numpy.ndarray
        T0 map generated by generate_t0_map()
    seg_mask : numpy.ndarray
        Binary segmentation mask with shape (height, width) or per-frame ROI mask

    Returns
    -------
    masked_t0_map : numpy.ndarray
        T0 map with NaN values outside the ROI
    """
    masked_t0_map = t0_map.copy().astype(float)
    _, roi_mask = _normalize_seg_mask(seg_mask, t0_map.shape[0], t0_map.shape[1], n_frames=None)
    # Set pixels outside ROI to NaN (will appear transparent in matplotlib)
    masked_t0_map[~roi_mask] = np.nan
    return masked_t0_map
