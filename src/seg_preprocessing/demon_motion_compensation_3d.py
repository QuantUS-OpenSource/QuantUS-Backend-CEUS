import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.restoration import denoise_nl_means, estimate_sigma
from dataclasses import dataclass
from typing import Tuple, List, Optional
from tqdm import tqdm
from .motion_compensation_3d import BoundingBox3D

@dataclass
class DeformableResult:
    """Per-frame dense displacement fields mapping the reference frame onto
    each frame, plus the crop the fields live in and the VOI in crop-local
    coordinates.
 
    A field has shape (Xc, Yc, Zc, 3) where (Xc, Yc, Zc) is the crop shape;
    field[..., i] is the displacement along axis i, in voxels. With
    fixed=reference and moving=frame, the field answers: "for each reference
    voxel, where does it sample in this frame?" -> reference->frame.
    """
    displacement_fields: List[np.ndarray]
    reference_frame: int
    final_similarities: np.ndarray          # per-frame final MSE (lower = better)
    crop: "BoundingBox3D"                    # where the field sits in the full volume
    voi_crop: np.ndarray                     # VOI mask, crop-local coords
 
    # --- reading motion inside the VOI (the analog of "tracking the ROI") ---
    def voi_field(self, frame_idx: int) -> np.ndarray:
        """Per-voxel displacement for every VOI voxel: shape (N, 3).
        This is the deformable richness the correlation tracker couldn't give."""
        return self.displacement_fields[frame_idx][self.voi_crop > 0]
 
    def mean_voi_displacement(self, frame_idx: int) -> np.ndarray:
        """Bulk (mean) VOI motion at this frame: shape (3,).
        Directly comparable to your old per-frame translation_vectors, so it's
        a good cross-check against the correlation tracker."""
        vf = self.voi_field(frame_idx)
        if vf.size == 0:
            return np.zeros(3, dtype=np.float32)
        return vf.mean(axis=0)
 
    def mean_voi_displacement_all(self) -> np.ndarray:
        """(T, 3) array of mean VOI motion per frame."""
        n = len(self.displacement_fields)
        return np.stack([self.mean_voi_displacement(t) for t in range(n)], axis=0)
 
    # --- propagating the reference VOI into a frame (analog of apply_to_mask) ---
    def warp_reference_crop_to(self, ref_crop_volume: np.ndarray,
                               frame_idx: int, order: int = 1) -> np.ndarray:
        """Warp a reference-crop volume forward into frame `frame_idx`."""
        return warp_by_field(ref_crop_volume,
                             self.displacement_fields[frame_idx], order=order)
 
    def deform_voi_mask_to(self, frame_idx: int, order: int = 0) -> np.ndarray:
        """Return the reference VOI mask deformed into frame `frame_idx`,
        in crop-local coordinates. order=0 keeps it a binary mask."""
        return warp_by_field(self.voi_crop.astype(np.float32),
                             self.displacement_fields[frame_idx], order=order)
 
 
# ---------------------------------------------------------------------------
# Warping
# ---------------------------------------------------------------------------
def warp_by_field(image: np.ndarray, field: np.ndarray, order: int = 1) -> np.ndarray:
    """Warp `image` by a dense displacement field of shape (X, Y, Z, 3).
    Sampling position for output voxel p is p + field[p]."""
    coords = np.indices(image.shape, dtype=np.float32)
    coords += np.moveaxis(field, -1, 0)
    return map_coordinates(image, coords, order=order, mode="nearest")
 
 
# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------
def _denoise_nlm(volume: np.ndarray, patch_size: int = 5, patch_distance: int = 6,
                  h_factor: float = 0.8, fast_mode: bool = True) -> np.ndarray:
    """Non-local means denoise a 3D crop before Demons registration.

    Speckle in B-mode makes np.gradient noisy voxel-to-voxel, which lets the
    demons force lock onto speckle instead of real tumor structure and drags
    the field far from the true tumor location. NLM averages each voxel over
    similar-looking patches elsewhere in the crop, smoothing speckle while
    preserving real edges (unlike a Gaussian blur, which would also blur the
    tumor boundary the force field needs to track).
    """
    vol = volume.astype(np.float32)
    sigma_est = float(np.mean(estimate_sigma(vol, channel_axis=None)))
    return denoise_nl_means(
        vol, h=h_factor * sigma_est, sigma=sigma_est,
        patch_size=patch_size, patch_distance=patch_distance,
        fast_mode=fast_mode, channel_axis=None, preserve_range=True,
    )


# ---------------------------------------------------------------------------
# Demon force
# ---------------------------------------------------------------------------
def _demons_velocity(fixed: np.ndarray, warped_moving: np.ndarray,
                     grad_fixed: Tuple[np.ndarray, ...],
                     grad_moving: Tuple[np.ndarray, ...],
                     alpha: float) -> np.ndarray:
    """Symmetric-forces demon velocity, shape (X, Y, Z, 3).
 
    v = 2 (m - f)(gf + gm) / ( |gf + gm|^2 + alpha^2 (m - f)^2 )
    The alpha^2 (m-f)^2 term stabilises flat/quiet regions where gradients
    vanish; without it the step blows up on speckle-flat voxels.
    """
    diff = warped_moving - fixed
    v = np.empty(fixed.shape + (3,), dtype=np.float32)
    for i in range(3):
        g = grad_fixed[i] + grad_moving[i]
        denom = g * g + (alpha ** 2) * diff * diff
        denom[denom < 1e-9] = 1e-9
        v[..., i] = (2.0 * diff * g) / denom
    return v
 
 
# ---------------------------------------------------------------------------
# Single-pair registration (single resolution)
# ---------------------------------------------------------------------------
def register_demons_3d(
    fixed: np.ndarray,
    moving: np.ndarray,
    n_iterations: int = 150,
    sigma_fluid: float = 1.0,
    sigma_diffusion: float = 2.0,   # larger -> smoother field; good for respiration
    alpha: float = 1.0,
    tol: float = 1e-5,
    initial_field: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Symmetric Demons between two equally-shaped 3D volumes.
 
    Returns (field, final_mse). The field warps `moving` to match `fixed`.
 
    `initial_field` is the warm-start hook: pass a neighbouring frame's field
    to begin from a good guess. Additive (compositive) update is used as a
    cheap stand-in for full scaling-and-squaring exp(v); adequate for a
    prototype and kept diffeomorphic-ish by the diffusion smoothing.
    """
    fixed = np.ascontiguousarray(fixed, dtype=np.float32)
    moving = np.ascontiguousarray(moving, dtype=np.float32)
 
    if initial_field is None:
        field = np.zeros(fixed.shape + (3,), dtype=np.float32)
    else:
        field = initial_field.astype(np.float32).copy()
 
    grad_fixed = np.gradient(fixed)     # computed once; fixed never changes
    prev_metric = np.inf
 
    for _ in range(n_iterations):
        warped = warp_by_field(moving, field, order=1)
        grad_moving = np.gradient(warped)
 
        v = _demons_velocity(fixed, warped, grad_fixed, grad_moving, alpha)
        for i in range(3):                       # fluid regularisation of update
            v[..., i] = gaussian_filter(v[..., i], sigma_fluid)
 
        field = field + v                        # compositive-additive update
 
        for i in range(3):                       # diffusion regularisation of total
            field[..., i] = gaussian_filter(field[..., i], sigma_diffusion)
 
        metric = float(np.mean((fixed - warp_by_field(moving, field, order=1)) ** 2))
        if abs(prev_metric - metric) < tol:
            break
        prev_metric = metric
 
    return field, prev_metric if prev_metric != np.inf else metric
 
 
# ---------------------------------------------------------------------------
# Full-sequence driver: reference -> each frame, with warm-start
# ---------------------------------------------------------------------------
def track_motion_demons(
    volumes: np.ndarray,
    reference_frame_idx: int,
    voi_mask: np.ndarray,
    crop_padding: int = 25,
    n_iterations: int = 150,
    sigma_fluid: float = 1.0,
    sigma_diffusion: float = 2.0,
    alpha: float = 1.0,
    warm_start: bool = True,
    warm_start_reject_factor: float = 2.0,
    frame_range: Optional[Tuple[int, int]] = None,
    denoise: bool = True,
    nlm_patch_size: int = 5,
    nlm_patch_distance: int = 6,
    nlm_h_factor: float = 0.8,
    nlm_fast_mode: bool = True,
) -> DeformableResult:
    """
    Register the reference frame to every frame with Demons, within a crop
    around the VOI, warm-starting each frame from its neighbour.

    Args:
        volumes:              (X, Y, Z, T) volume sequence.
        reference_frame_idx:  fixed/reference frame.
        voi_mask:             (X, Y, Z) VOI mask (your interpolated 3-oval).
                              Defines the crop AND the read-out region.
        crop_padding:         voxels of margin added around the VOI bbox so the
                              lesion stays in view across the respiratory cycle.
                              Set this >= the largest reference->frame excursion.
        warm_start:           init each frame's field from the previously solved
                              neighbour (walking outward from the reference, the
                              same forward/backward structure as your ILSA code).
        warm_start_reject_factor:
                              if a warm-started frame's MSE ends up worse than
                              this * the neighbour's MSE, redo it cold. Stops one
                              bad frame from poisoning the chain of guesses.
        frame_range:          optional (start, end) frame indices, end exclusive,
                              restricting which frames are registered. Defaults to
                              the full sequence. reference_frame_idx must lie
                              inside this range. Frames outside the range are left
                              as an identity (zero) field rather than tracked.
        denoise:               apply non-local means to the fixed/moving crops
                              before registration (default: True). Demons drives
                              off np.gradient, which is noisy on raw B-mode
                              speckle; NLM smooths speckle while preserving the
                              tumor edge, giving the force field a cleaner signal
                              to lock onto instead of drifting off toward speckle.
        nlm_patch_size:       NLM patch side length in voxels (default: 5).
        nlm_patch_distance:   NLM search window radius in voxels (default: 6).
        nlm_h_factor:         NLM filter strength as a multiple of the crop's
                              estimated noise sigma (default: 0.8). Higher =
                              smoother/more aggressive denoising.
        nlm_fast_mode:        use skimage's FFT-accelerated NLM (default: True).

    Returns:
        DeformableResult (fields are in crop-local coordinates).
    """
    volume_shape = volumes.shape[:-1]
    n_frames = volumes.shape[-1]

    start_frame, end_frame = frame_range if frame_range is not None else (0, n_frames)
    if not (0 <= start_frame < end_frame <= n_frames):
        raise ValueError(f"frame_range {(start_frame, end_frame)} is out of bounds for {n_frames} frames")
    if not (start_frame <= reference_frame_idx < end_frame):
        raise ValueError(
            f"reference_frame_idx {reference_frame_idx} must lie within frame_range "
            f"{(start_frame, end_frame)}"
        )

    # VOI -> crop box. This is your existing helper doing a new job:
    # "registration region" rather than "search template".
    crop = BoundingBox3D.from_mask(voi_mask, padding=crop_padding).clip(volume_shape)
    fixed = crop.extract_from_volume(volumes[..., reference_frame_idx])
    if denoise:
        fixed = _denoise_nlm(fixed, nlm_patch_size, nlm_patch_distance, nlm_h_factor, nlm_fast_mode)
    voi_crop = crop.extract_from_volume(voi_mask)

    fields: List[Optional[np.ndarray]] = [None] * n_frames
    sims = np.full(n_frames, np.nan, dtype=np.float32)

    fields[reference_frame_idx] = np.zeros(fixed.shape + (3,), dtype=np.float32)
    sims[reference_frame_idx] = 0.0
 
    def _register(frame_idx: int, guess: Optional[np.ndarray]):
        moving = crop.extract_from_volume(volumes[..., frame_idx])
        field, mse = register_demons_3d(
            fixed, moving,
            n_iterations=n_iterations,
            sigma_fluid=sigma_fluid,
            sigma_diffusion=sigma_diffusion,
            alpha=alpha,
            initial_field=guess,
        )
        # Safeguard: if the warm-started result is much worse than the guess's
        # own score, the guess was misleading -> redo cold.
        if (warm_start and guess is not None
                and np.isfinite(mse) and mse > warm_start_reject_factor * (sims_guess := _guess_mse)):
            field, mse = register_demons_3d(
                fixed, moving,
                n_iterations=n_iterations,
                sigma_fluid=sigma_fluid,
                sigma_diffusion=sigma_diffusion,
                alpha=alpha,
                initial_field=None,
            )
        return field, mse
 
    print(f"\n=== Demons deformable tracking ===")
    print(f"crop shape {fixed.shape}, padding {crop_padding}, "
          f"warm_start={warm_start}, frame_range={(start_frame, end_frame)}, "
          f"denoise={denoise}")

    # Forward pass: reference+1 .. end_frame-1, each warm-started from its predecessor.
    for frame_idx in tqdm(range(reference_frame_idx + 1, end_frame),
                          desc="Demons forward", unit="frame"):
        guess = fields[frame_idx - 1] if warm_start else None
        _guess_mse = sims[frame_idx - 1] if warm_start else np.inf
        moving = crop.extract_from_volume(volumes[..., frame_idx])
        if denoise:
            moving = _denoise_nlm(moving, nlm_patch_size, nlm_patch_distance, nlm_h_factor, nlm_fast_mode)
        field, mse = register_demons_3d(
            fixed, moving, n_iterations, sigma_fluid, sigma_diffusion, alpha,
            initial_field=guess)
        if (warm_start and guess is not None and np.isfinite(mse)
                and np.isfinite(_guess_mse)
                and mse > warm_start_reject_factor * _guess_mse):
            field, mse = register_demons_3d(
                fixed, moving, n_iterations, sigma_fluid, sigma_diffusion,
                alpha, initial_field=None)
        fields[frame_idx] = field
        sims[frame_idx] = mse
 
    # Backward pass: reference-1 .. start_frame, warm-started from its successor.
    for frame_idx in tqdm(range(reference_frame_idx - 1, start_frame - 1, -1),
                          desc="Demons backward", unit="frame"):
        guess = fields[frame_idx + 1] if warm_start else None
        _guess_mse = sims[frame_idx + 1] if warm_start else np.inf
        moving = crop.extract_from_volume(volumes[..., frame_idx])
        if denoise:
            moving = _denoise_nlm(moving, nlm_patch_size, nlm_patch_distance, nlm_h_factor, nlm_fast_mode)
        field, mse = register_demons_3d(
            fixed, moving, n_iterations, sigma_fluid, sigma_diffusion, alpha,
            initial_field=guess)
        if (warm_start and guess is not None and np.isfinite(mse)
                and np.isfinite(_guess_mse)
                and mse > warm_start_reject_factor * _guess_mse):
            field, mse = register_demons_3d(
                fixed, moving, n_iterations, sigma_fluid, sigma_diffusion,
                alpha, initial_field=None)
        fields[frame_idx] = field
        sims[frame_idx] = mse
 
    # Frames outside [start_frame, end_frame) weren't registered; treat them as
    # untracked/identity rather than leaving None, so DeformableResult stays
    # indexable by absolute frame index across the full sequence.
    identity_field = np.zeros(fixed.shape + (3,), dtype=np.float32)
    n_untracked = 0
    for frame_idx in range(n_frames):
        if fields[frame_idx] is None:
            fields[frame_idx] = identity_field
            n_untracked += 1

    print(f"Mean MSE: {np.nanmean(sims):.4f}   Max MSE: {np.nanmax(sims):.4f}")
    if n_untracked:
        print(f"{n_untracked}/{n_frames} frames were outside frame_range "
              f"{(start_frame, end_frame)} and left untracked (identity field)")

    return DeformableResult(fields, reference_frame_idx, sims, crop, voi_crop)