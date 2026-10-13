from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage
from ...data_objs.seg import CeusSeg
from ..demon_motion_compensation_3d import track_motion_demons
import numpy as np

@required_kwargs('bmode_image_data','reference_frame','frame_window','crop_padding','n_iterations','sigma_fluid','sigma_diffusion','alpha','warm_start','warm_start_reject_factor','denoise','nlm_patch_size','nlm_patch_distance','nlm_h_factor','nlm_fast_mode')
def demon_motion_compensation_3d(image_data: UltrasoundImage, seg_data: CeusSeg, **kwargs) -> CeusSeg:
    """
    Apply 3D deformable motion compensation using symmetric Demons registration.

    Unlike the rigid ILSA tracker (motion_compensation_3d), this produces a
    per-voxel dense displacement field for each frame rather than a single
    translation vector, so the result is stored on `seg_data.deformable_compensation`
    rather than `seg_data.motion_compensation` and does not set `seg_data.use_mc`.

    Kwargs:
        bmode_image_data (UltrasoundImage): B-mode data for motion tracking [REQUIRED]
        reference_frame (int): Reference frame index (default: 0)
        frame_window (int): If set, only register frames within +/- frame_window of
                             reference_frame (clamped to the sequence bounds), e.g.
                             frame_window=5 registers 11 frames centered on the
                             reference. Takes precedence over start_frame/end_frame.
        start_frame (int): First frame index to register, inclusive (default: 0)
        end_frame (int): Last frame index to register, exclusive (default: n_frames).
                          Frames outside [start_frame, end_frame) are left untracked
                          (identity displacement field) instead of being registered,
                          so registration can be limited to e.g. the bolus window.
        crop_padding (int): Voxels of margin around the VOI bbox defining the
                             registration crop (default: 25)
        n_iterations (int): Demons iterations per frame pair (default: 150)
        sigma_fluid (float): Fluid regularization sigma for the update field (default: 1.0)
        sigma_diffusion (float): Diffusion regularization sigma for the total field;
                                  larger -> smoother field, good for respiration (default: 2.0)
        alpha (float): Demons force stabilization term (default: 1.0)
        warm_start (bool): Warm-start each frame from its neighbour (default: True)
        warm_start_reject_factor (float): Redo a warm-started frame cold if its MSE
                                           exceeds this factor times the neighbour's (default: 2.0)
        denoise (bool): Apply non-local means to the registration crop before Demons
                         (default: True). Raw B-mode speckle makes the Demons gradient
                         noisy, which can drag the field off the real tumor; NLM
                         smooths speckle while preserving the tumor edge.
        nlm_patch_size (int): NLM patch side length in voxels (default: 5)
        nlm_patch_distance (int): NLM search window radius in voxels (default: 6)
        nlm_h_factor (float): NLM filter strength as a multiple of estimated noise
                               sigma; higher = smoother (default: 0.8)
        nlm_fast_mode (bool): Use skimage's FFT-accelerated NLM (default: True)

    Returns:
        CeusSeg: Segmentation with the deformable motion compensation result stored
    """
    bmode_image_data = kwargs['bmode_image_data']
    reference_frame = kwargs.get('reference_frame', 0)
    crop_padding = kwargs.get('crop_padding', 25)
    n_iterations = kwargs.get('n_iterations', 150)
    sigma_fluid = kwargs.get('sigma_fluid', 1.0)
    sigma_diffusion = kwargs.get('sigma_diffusion', 2.0)
    alpha = kwargs.get('alpha', 1.0)
    warm_start = kwargs.get('warm_start', True)
    warm_start_reject_factor = kwargs.get('warm_start_reject_factor', 2.0)
    denoise = kwargs.get('denoise', True)
    nlm_patch_size = kwargs.get('nlm_patch_size', 5)
    nlm_patch_distance = kwargs.get('nlm_patch_distance', 6)
    nlm_h_factor = kwargs.get('nlm_h_factor', 0.8)
    nlm_fast_mode = kwargs.get('nlm_fast_mode', True)

    # Validate inputs
    if not isinstance(bmode_image_data, UltrasoundImage):
        raise TypeError("bmode_image_data must be an UltrasoundImage object")

    bmode_shape = bmode_image_data.pixel_data.shape
    if bmode_image_data.pixel_data.ndim != 4:
        raise ValueError(f"B-mode data must be 4D (X,Y,Z,T), got shape {bmode_shape}")

    voi_mask = seg_data.seg_mask
    if voi_mask.ndim != 3:
        raise ValueError(f"Segmentation mask must be 3D (X, Y, Z), got shape {voi_mask.shape}")

    n_frames = bmode_shape[-1]
    frame_window = kwargs.get('frame_window', None)
    if frame_window is not None:
        start_frame = max(0, reference_frame - frame_window)
        end_frame = min(n_frames, reference_frame + frame_window + 1)
    else:
        start_frame = kwargs.get('start_frame', 0)
        end_frame = kwargs.get('end_frame', n_frames)

    print("\n" + "="*60)
    print("3D Deformable Motion Compensation with Demons Registration")
    print("="*60)
    print(f"  Reference frame: {reference_frame}")
    print(f"  Frame range: [{start_frame}, {end_frame})")

    deformable_result = track_motion_demons(
        bmode_image_data.pixel_data,
        reference_frame,
        voi_mask,
        crop_padding=crop_padding,
        n_iterations=n_iterations,
        sigma_fluid=sigma_fluid,
        sigma_diffusion=sigma_diffusion,
        alpha=alpha,
        warm_start=warm_start,
        warm_start_reject_factor=warm_start_reject_factor,
        frame_range=(start_frame, end_frame),
        denoise=denoise,
        nlm_patch_size=nlm_patch_size,
        nlm_patch_distance=nlm_patch_distance,
        nlm_h_factor=nlm_h_factor,
        nlm_fast_mode=nlm_fast_mode,
    )

    seg_data.deformable_compensation = deformable_result

    tracked_sims = deformable_result.final_similarities[start_frame:end_frame]
    image_data.extras_dict['deformable_compensation'] = {
        'applied': True,
        'reference_frame': reference_frame,
        'frame_range': (start_frame, end_frame),
        'mean_mse': float(np.nanmean(tracked_sims)),
        'max_mse': float(np.nanmax(tracked_sims)),
    }

    print("\n" + "="*60)
    print("Deformable Motion Compensation Complete!")
    print(f"  Mean MSE: {np.nanmean(tracked_sims):.4f}")
    print("="*60 + "\n")

    return seg_data
