"""
Best-effort implementation of the multi-scale wavelet pipeline from:

  "An Efficient Multi-Scale Wavelet Approach for Dehazing and Denoising
  Ultrasound Images Using Fractional-Order Filtering"
  Fractal and Fractional 2024, 8(9), 549. https://www.mdpi.com/2504-3110/8/9/549
  (doi: 10.3390/fractalfract8090549)

CAVEAT: MDPI blocked a direct fetch of the PDF/HTML (403); the pipeline below
was reconstructed from a third-party HTML-to-text render of the page, not the
original PDF. Treat this as a best-effort reproduction, not a guaranteed exact
transcription -- verify against the actual PDF if you have access, especially
for anything listed as "not stated precisely" below.

Per-pyramid-level pipeline (applied to the wavelet approximation/LL subband
at each scale, coarsest to finest):
  1. Structure tensor -> per-pixel coherence + dominant orientation, used to
     classify edge vs. non-edge pixels.
  2. Directional filter on edge pixels: interpolate along the edge tangent
     direction (sharpen) and subtract along the edge normal (suppress
     blur across the boundary).
  3. Guided filter on non-edge (homogeneous/speckle) pixels.
  4. Fractional-order (Grunwald-Letnikov) filter, order v=0.05, added back
     as a texture/detail-preserving term.
  5. Dark-channel dehazing (classic single-image dehazing, adapted to a
     single-channel image) to remove the low-contrast "haze" veiling common
     in ultrasound.
  6. Clip to a valid range, then reconstruct one wavelet level up using the
     ORIGINAL detail (LH/HL/HH) coefficients at that scale, and repeat until
     back at full resolution.

Stated precisely in the extracted text (used as-is):
  - structure tensor coherence: (|lambda1 - lambda2|) / (|lambda1 + lambda2|)
  - directional filter: x' = (1 - a_e + a_n)*x + a_e*(e1+e2)/2 - a_n*(n1+n2)/2
  - guided filter kernel (the standard He et al. box-filter formulation)
  - fractional order v = 0.05, Grunwald-Letnikov recursion
    w_0 = 1; w_j = (1 - (v+1)/j) * w_{j-1}
  - dark-channel dehazing with omega = 0.95, atmospheric-light-like level
    estimated from the brightest 0.1% of dark-channel pixels

NOT stated precisely in the extracted text (exposed as tunable parameters
with reasonable defaults instead of hardcoded guesses):
  - wavelet family and number of decomposition levels
  - the coherence threshold used to classify edge vs. non-edge pixels
  - the directional-filter blend weights alpha_e / alpha_n
  - guided-filter radius/epsilon
  - fractional-filter truncation length (number of GL series terms)
  - dark-channel local-patch radius

Usage:
    from medsam2_wavelet_pyramid import wavelet_pyramid_enhance

    result = wavelet_pyramid_enhance(image_slice)
"""

import time

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter, uniform_filter, minimum_filter, map_coordinates, convolve1d


def _structure_tensor_coherence(image, sigma=1.0):
    """Gaussian-smoothed structure tensor -> per-pixel coherence in [0, 1] and dominant orientation (radians)."""
    Iy, Ix = np.gradient(image)
    Jxx = gaussian_filter(Ix * Ix, sigma)
    Jxy = gaussian_filter(Ix * Iy, sigma)
    Jyy = gaussian_filter(Iy * Iy, sigma)

    tmp = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    lambda1 = 0.5 * (Jxx + Jyy + tmp)
    lambda2 = 0.5 * (Jxx + Jyy - tmp)
    coherence = np.abs(lambda1 - lambda2) / (np.abs(lambda1 + lambda2) + 1e-8)

    # Dominant eigenvector direction (the gradient/edge-normal direction).
    theta = 0.5 * np.arctan2(2 * Jxy, (Jxx - Jyy) + 1e-8)
    return coherence, theta


def _sample_along_direction(image, theta, step=1.0):
    """Bilinearly sample neighbor values at +step and -step along direction theta (radians), per pixel."""
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy, dx = np.sin(theta), np.cos(theta)

    plus = map_coordinates(image, [yy + step * dy, xx + step * dx], order=1, mode="nearest")
    minus = map_coordinates(image, [yy - step * dy, xx - step * dx], order=1, mode="nearest")
    return plus, minus


def directional_filter(image, edge_mask, theta, alpha_e=0.7, alpha_n=0.3):
    """
    Anisotropic edge filter: sharpen along the edge tangent (e1, e2), suppress
    along the edge normal (n1, n2) -- theta is the structure tensor's dominant
    eigenvector (the normal direction); the tangent is perpendicular to it.
    Only applied where edge_mask is True.
    """
    tangent = theta + np.pi / 2
    e1, e2 = _sample_along_direction(image, tangent)
    n1, n2 = _sample_along_direction(image, theta)

    filtered = (1 - alpha_e + alpha_n) * image + alpha_e * (e1 + e2) / 2 - alpha_n * (n1 + n2) / 2
    return np.where(edge_mask, filtered, image)


def guided_filter(image, guide, radius=4, eps=1e-2):
    """Standard (He et al.) box-filter guided filter; guide == image here (self-guided)."""
    mean_I = uniform_filter(guide, radius)
    mean_p = uniform_filter(image, radius)
    mean_Ip = uniform_filter(guide * image, radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = uniform_filter(guide * guide, radius)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, radius)
    mean_b = uniform_filter(b, radius)
    return mean_a * guide + mean_b


def _grunwald_letnikov_weights(order, num_terms):
    """w_0 = 1; w_j = (1 - (order+1)/j) * w_{j-1}, per the paper's stated recursion."""
    w = np.empty(num_terms, dtype=np.float64)
    w[0] = 1.0
    for j in range(1, num_terms):
        w[j] = (1.0 - (order + 1.0) / j) * w[j - 1]
    return w


def fractional_order_filter(image, order=0.05, num_terms=8):
    """
    Separable Grunwald-Letnikov fractional derivative (order v) along rows and
    columns, combined into a detail/texture magnitude and added back to the
    image -- a pure fractional derivative has near-zero DC response, so this
    acts as a texture-preserving enhancement term rather than replacing image.
    """
    w = _grunwald_letnikov_weights(order, num_terms)
    dx = convolve1d(image, w, axis=1, mode="nearest")
    dy = convolve1d(image, w, axis=0, mode="nearest")
    detail = np.hypot(dx, dy)
    return image + detail


def dark_channel_dehaze(image, patch_radius=7, omega=0.95, eps=1e-3):
    """
    Single-channel adaptation of the classic dark-channel dehazing (He et al.
    2010): dark channel -> global "noise"/atmospheric-light-like level from
    the brightest 0.1% of dark-channel pixels -> transmission -> recovery.
    """
    dark = minimum_filter(image, size=2 * patch_radius + 1, mode="nearest")

    flat_dark = dark.ravel()
    n_bright = max(1, int(0.001 * flat_dark.size))
    noise_level = float(np.mean(np.sort(flat_dark)[-n_bright:]))
    noise_level = max(noise_level, eps)

    transmission = 1 - omega * (dark / noise_level)
    transmission = np.clip(transmission, eps, 1.0)

    return (image - noise_level) / np.maximum(transmission, eps) + noise_level


def _to_uint8_display(array):
    """Percentile-free, plain min-max stretch to uint8 -- for displaying an
    intermediate stage on its own scale, not the final output's convention."""
    a = array.astype(np.float32)
    a_min, a_max = a.min(), a.max()
    return ((a - a_min) / (a_max - a_min + 1e-8) * 255).astype(np.uint8)


def wavelet_pyramid_enhance(
    image_slice,
    wavelet="db2",
    levels=2,
    tensor_sigma=1.0,
    edge_threshold=0.3,
    alpha_e=0.7,
    alpha_n=0.3,
    guided_radius=4,
    guided_eps=1e-2,
    frac_order=0.05,
    frac_terms=8,
    dehaze_patch_radius=7,
    dehaze_omega=0.95,
    p_low_percentile=5.0,
    p_high_percentile=98.0,
    return_stages=False,
):
    """
    Multi-scale wavelet dehazing/denoising, per the pipeline documented in
    this module's docstring. Prints wall-clock time taken.

    At each pyramid level (coarsest to finest), the approximation subband is
    passed through: structure-tensor edge classification -> directional
    filter (edges) / guided filter (non-edges) -> fractional-order detail
    enhancement -> dark-channel dehazing -> clip, then reconstructed one
    level up with that scale's original detail coefficients. The final
    result is percentile-clipped and rescaled to uint8, same convention as
    enhance_bmode_noise.

    return_stages: if True, returns (result, stages) instead of just result.
    `stages` is a list with one dict per pyramid level (coarsest to finest),
    each holding every intermediate array for that level as plain uint8
    (min-max stretched for display, NOT the final percentile convention) --
    for inspecting what each step actually does rather than only the final
    output. Keys per level:
        "approx_in"         - approximation subband entering this level
        "coherence"          - structure-tensor coherence map (0-1, float)
        "edge_mask"          - boolean edge classification (not stretched)
        "after_directional"  - after the directional filter (edge pixels changed)
        "after_guided"       - after blending in the guided filter (non-edge pixels changed)
        "after_fractional"   - after the fractional-order detail term is added
        "after_dehaze"       - after dark-channel dehazing, this level's output
        "reconstructed"      - after idwt2 back up one level (next level's
                               approx_in, or the raw final image on the last level)
    """
    start = time.time()
    stages = []

    image = image_slice.astype(np.float64)
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=levels)
    approx, detail_levels = coeffs[0], list(coeffs[1:])   # coarsest -> finest order

    for i, (cH, cV, cD) in enumerate(detail_levels):
        approx_in = approx

        coherence, theta = _structure_tensor_coherence(approx, sigma=tensor_sigma)
        edge_mask = coherence > edge_threshold

        after_directional = directional_filter(approx, edge_mask, theta, alpha_e, alpha_n)
        smoothed = guided_filter(after_directional, after_directional, guided_radius, guided_eps)
        after_guided = np.where(edge_mask, after_directional, smoothed)

        after_fractional = fractional_order_filter(after_guided, frac_order, frac_terms)
        after_dehaze = dark_channel_dehaze(after_fractional, dehaze_patch_radius, dehaze_omega)
        approx = np.clip(after_dehaze, 0, 255)

        approx = pywt.idwt2((approx, (cH, cV, cD)), wavelet)
        # pywt's boundary handling doesn't guarantee a clean doubling in size
        # each level (depends on wavelet length and odd/even source sizes) --
        # trim to whatever shape comes next: the following level's detail
        # coefficients, or the original image on the last iteration.
        target_shape = detail_levels[i + 1][0].shape if i + 1 < len(detail_levels) else image_slice.shape
        approx = approx[: target_shape[0], : target_shape[1]]

        if return_stages:
            stages.append({
                "approx_in": _to_uint8_display(approx_in),
                "coherence": coherence,
                "edge_mask": edge_mask,
                "after_directional": _to_uint8_display(after_directional),
                "after_guided": _to_uint8_display(after_guided),
                "after_fractional": _to_uint8_display(after_fractional),
                "after_dehaze": _to_uint8_display(after_dehaze),
                "reconstructed": _to_uint8_display(approx),
            })

    approx = np.clip(approx, 0, 255)

    non_zero = approx[approx != 0]
    if non_zero.size == 0:
        result = np.zeros_like(image_slice, dtype=np.uint8)
    else:
        p_low = np.percentile(non_zero, p_low_percentile)
        p_high = np.percentile(non_zero, p_high_percentile)
        clipped = np.clip(approx, p_low, p_high)
        result = ((clipped - p_low) / (p_high - p_low + 1e-8) * 255).astype(np.uint8)

    elapsed_ms = (time.time() - start) * 1000
    print(f"wavelet_pyramid_enhance: {elapsed_ms:.1f} ms")

    if return_stages:
        return result, stages
    return result
