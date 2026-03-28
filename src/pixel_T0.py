import warnings
from typing import Optional, Tuple

import numpy as np

from src.data_objs.image import UltrasoundImage
from src.data_objs.seg import CeusSeg


def _normalize_seg_mask_3d(
    seg_mask: np.ndarray,
    spatial_shape: Tuple[int, int, int],
    n_frames: Optional[int],
):
    """
    Convert seg_mask to per-frame and union masks for 3D volumetric data.

    The QuantUS-CEUS framework stores 3D+time image data as
    (sag, cor, ax, time) and static 3D segmentation masks as (sag, cor, ax).
    A motion-compensated mask may be stored as (sag, cor, ax, time).

    Parameters
    ----------
    seg_mask : np.ndarray
        Binary segmentation mask. Accepted shapes:
        - 3D static VOI: ``(sag, cor, ax)`` matching *spatial_shape*
        - 4D per-frame VOI: ``(sag, cor, ax, n_frames)``
    spatial_shape : tuple of int
        Expected ``(sag, cor, ax)`` spatial dimensions of the image volume.
    n_frames : int or None
        Number of temporal frames.  When *None* only the union mask is returned.

    Returns
    -------
    per_frame_mask : np.ndarray or None
        Boolean mask with shape ``(sag, cor, ax, n_frames)`` when *n_frames*
        is not None, otherwise None.
    union_mask : np.ndarray
        Boolean mask with shape ``(sag, cor, ax)`` — the union over time.
    """
    mask = np.asarray(seg_mask)
    mask = np.squeeze(mask)
    sag, cor, ax = spatial_shape

    # --- 3D static mask (same VOI every frame) ---
    if mask.ndim == 3:
        mask_bool = mask > 0
        if mask_bool.shape != (sag, cor, ax):
            raise ValueError(
                f"Segmentation mask spatial shape {mask_bool.shape} does not "
                f"match expected {(sag, cor, ax)}"
            )
        if n_frames is not None:
            per_frame = np.repeat(mask_bool[..., np.newaxis], n_frames, axis=3)
        else:
            per_frame = None
        return per_frame, mask_bool

    # --- 4D per-frame mask (motion-compensated VOI) ---
    if mask.ndim == 4:
        mask_bool = mask > 0
        # Expect (sag, cor, ax, time)
        if mask_bool.shape[:3] != (sag, cor, ax):
            raise ValueError(
                f"Segmentation mask spatial shape {mask_bool.shape[:3]} does not "
                f"match expected {(sag, cor, ax)}"
            )
        mask_frames = mask_bool.shape[3]
        if n_frames is not None and mask_frames != n_frames:
            if mask_frames < n_frames:
                warnings.warn(
                    f"Segmentation mask has {mask_frames} frames; "
                    f"padding to match pixel data frames ({n_frames}).",
                    RuntimeWarning,
                )
                pad = np.repeat(
                    mask_bool[..., mask_frames - 1 : mask_frames],
                    n_frames - mask_frames,
                    axis=3,
                )
                mask_bool = np.concatenate([mask_bool, pad], axis=3)
            else:
                warnings.warn(
                    f"Segmentation mask has {mask_frames} frames; "
                    f"truncating to match pixel data frames ({n_frames}).",
                    RuntimeWarning,
                )
                mask_bool = mask_bool[..., :n_frames]
        per_frame = mask_bool if n_frames is not None else None
        union_mask = np.any(mask_bool, axis=3)
        return per_frame, union_mask

    raise ValueError(f"Unsupported segmentation mask shape {seg_mask.shape}")


def _shift_mask(
    static_mask: np.ndarray,
    dx: int, dy: int, dz: int,
) -> np.ndarray:
    """
    Shift a 3-D boolean mask by integer translation (dx, dy, dz).

    Uses ``np.roll`` followed by zeroing out the wrapped-around border so
    that voxels that roll past an edge are *not* included.

    The axis mapping follows the paramap convention:
        - dy → axis 0 (sagittal)
        - dz → axis 1 (coronal)
        - dx → axis 2 (axial)

    Parameters
    ----------
    static_mask : np.ndarray
        Boolean mask with shape ``(sag, cor, ax)``.
    dx, dy, dz : int
        Translation in axial, sagittal, and coronal voxels respectively.

    Returns
    -------
    shifted : np.ndarray
        Shifted boolean mask (same shape).
    """
    shifted = static_mask.copy()

    # Shift along sagittal (axis 0) by dy
    if dy != 0:
        shifted = np.roll(shifted, dy, axis=0)
        if dy > 0:
            shifted[:dy, :, :] = False
        else:
            shifted[dy:, :, :] = False

    # Shift along coronal (axis 1) by dz
    if dz != 0:
        shifted = np.roll(shifted, dz, axis=1)
        if dz > 0:
            shifted[:, :dz, :] = False
        else:
            shifted[:, dz:, :] = False

    # Shift along axial (axis 2) by dx
    if dx != 0:
        shifted = np.roll(shifted, dx, axis=2)
        if dx > 0:
            shifted[:, :, :dx] = False
        else:
            shifted[:, :, dx:] = False

    return shifted


def generate_t0_map_3d(
    image_data: UltrasoundImage,
    seg_data: CeusSeg,
    threshold: float = 150,
    start_frame: int = 50,
    end_frame: int = 250,
    min_consecutive_frames: int = 1,
) -> np.ndarray:
    """
    Generate a 3D T0 map showing when each voxel first reaches the
    intensity threshold, with motion compensation applied per-frame.

    Motion compensation is handled in the same way as the paramap
    ``compute_curves``: if ``seg_data.use_mc`` is True, the segmentation
    mask is shifted each frame by the translation vector from
    ``seg_data.motion_compensation.get_translation(frame_ix)``.

    """
    vol = image_data.intensities_for_analysis  # (sag, cor, ax, n_frames)

    if vol.ndim != 4:
        raise ValueError(
            f"Expected 4-D (sag, cor, ax, time) intensities_for_analysis, "
            f"got ndim={vol.ndim}"
        )

    sag, cor, ax, n_frames = vol.shape
    spatial_shape = (sag, cor, ax)

    # Clamp frame range
    if end_frame > n_frames:
        end_frame = n_frames
        print(f"Warning: end_frame adjusted to {n_frames} (max available)")

    # Static mask (used as the reference that gets shifted each frame)
    static_mask = seg_data.seg_mask > 0
    if static_mask.shape != spatial_shape:
        raise ValueError(
            f"Segmentation mask shape {static_mask.shape} does not match "
            f"image spatial shape {spatial_shape}"
        )

    # Initialise outputs
    t0_map = np.zeros(spatial_shape, dtype=np.float32)
    consecutive_count = np.zeros(spatial_shape, dtype=np.int32)

    for i in range(start_frame, end_frame):
        current_vol = vol[:, :, :, i]

        # Get motion-compensated mask for this frame
        if seg_data.use_mc:
            dx, dy, dz = seg_data.motion_compensation.get_translation(i)
            dx, dy, dz = int(round(dx)), int(round(dy)), int(round(dz))
            frame_mask = _shift_mask(static_mask, dx, dy, dz)
        else:
            frame_mask = _shift_mask(static_mask, 0, 0, 0)

        above_threshold = (current_vol >= threshold) & frame_mask

        consecutive_count = np.where(above_threshold, consecutive_count + 1, 0)

        newly_detected = (consecutive_count == min_consecutive_frames) & (t0_map == 0)

        if min_consecutive_frames > 1:
            first_cross_frame = i - min_consecutive_frames + 1
            t0_map[newly_detected] = end_frame - first_cross_frame
        else:
            t0_map[newly_detected] = end_frame - i

    return t0_map


def get_t0_statistics_3d(
    t0_map: np.ndarray,
    seg_data: CeusSeg,
) -> dict:
    """
    Calculate statistics from the 3D T0 map within the VOI.

    Parameters
    ----------
    t0_map : np.ndarray
        T0 volume generated by :func:`generate_t0_map_3d`.
    seg_data : CeusSeg
        Segmentation data whose ``seg_mask`` defines the VOI.

    Returns
    -------
    stats : dict
        ``mean_t0``, ``median_t0``, ``std_t0``, ``min_t0``, ``max_t0``,
        and ``coverage`` (% of VOI voxels that were activated).
    """
    _, union_mask = _normalize_seg_mask_3d(
        seg_data.seg_mask, t0_map.shape, n_frames=None
    )
    t0_in_voi = t0_map[union_mask]
    activated = t0_in_voi[t0_in_voi > 0]

    return {
        "mean_t0": float(np.mean(activated)) if len(activated) > 0 else 0,
        "median_t0": float(np.median(activated)) if len(activated) > 0 else 0,
        "std_t0": float(np.std(activated)) if len(activated) > 0 else 0,
        "min_t0": float(np.min(activated)) if len(activated) > 0 else 0,
        "max_t0": float(np.max(activated)) if len(activated) > 0 else 0,
        "coverage": float(len(activated) / len(t0_in_voi) * 100) if len(t0_in_voi) > 0 else 0,
    }


def mask_t0_map_3d(
    t0_map: np.ndarray,
    seg_data: CeusSeg,
) -> np.ndarray:
    """
    Mask the 3D T0 map so that voxels outside the VOI are NaN.

    Parameters
    ----------
    t0_map : np.ndarray
        T0 volume from :func:`generate_t0_map_3d`.
    seg_data : CeusSeg
        Segmentation defining the VOI.

    Returns
    -------
    masked_t0_map : np.ndarray
        Copy of *t0_map* with NaN outside the VOI (for visualization).
    """
    masked = t0_map.copy().astype(float)
    _, union_mask = _normalize_seg_mask_3d(
        seg_data.seg_mask, t0_map.shape, n_frames=None
    )
    masked[~union_mask] = np.nan
    return masked