"""
CuPy backend for the 3D normalized cross-correlation in motion_compensation_3d.

Mirrors the scipy 'valid'-mode FFT path in
MotionCompensation3D.compute_3d_correlation_vectorized. GPU_AVAILABLE is False
when CuPy or a CUDA device is missing, and the caller stays on the CPU path.
"""

from scipy.fft import next_fast_len

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    GPU_AVAILABLE = False

# conj(FFT) of the all-ones kernel, keyed by (ref_shape, fft_shape). It depends
# only on shapes, so one entry is reused across every frame of a block.
_ones_spectrum_cache = {}


def _ones_spectrum(ref_shape, fft_shape):
    key = (ref_shape, fft_shape)
    spectrum = _ones_spectrum_cache.get(key)
    if spectrum is None:
        kernel = cp.zeros(fft_shape)
        kernel[:ref_shape[0], :ref_shape[1], :ref_shape[2]] = 1
        spectrum = cp.conj(cp.fft.rfftn(kernel))
        _ones_spectrum_cache[key] = spectrum
    return spectrum


def _normalize(volume):
    """Zero-mean, unit-std, matching MotionCompensation3D.normalize_volume.

    Dividing by 1 where std is 0 is the same as not dividing, so this stays
    branch-free and avoids a device sync per call.
    """
    std = volume.std()
    return (volume - volume.mean()) / cp.where(std == 0, 1, std)


def compute_ncc_map_gpu(search_region, ref_voi):
    """Valid-mode normalized cross-correlation of ref_voi over search_region.

    Runs in float64 to match the CPU path it replaces; at these sizes the FFT is
    bandwidth-bound, so float32 buys little.

    Args:
        search_region: 3D search volume (host array)
        ref_voi: 3D template to slide over it (host array)

    Returns:
        Host array of shape search_shape - ref_shape + 1.
    """
    ref_shape = tuple(ref_voi.shape)
    corr_shape = tuple(s - r + 1 for s, r in zip(search_region.shape, ref_shape))

    # Circular correlation at an FFT size >= search_shape has no wraparound over
    # the valid region, so the transform is sized by the search region alone
    # rather than the usual search + ref - 1.
    fft_shape = tuple(next_fast_len(s) for s in search_region.shape)

    search = _normalize(cp.asarray(search_region, dtype=cp.float64))
    ref = _normalize(cp.asarray(ref_voi, dtype=cp.float64))

    ref_padded = cp.zeros(fft_shape)
    ref_padded[:ref_shape[0], :ref_shape[1], :ref_shape[2]] = ref
    correlation = cp.fft.irfftn(
        cp.fft.rfftn(search, fft_shape) * cp.conj(cp.fft.rfftn(ref_padded)), fft_shape)

    local_sum_sq = cp.fft.irfftn(
        cp.fft.rfftn(search ** 2, fft_shape) * _ones_spectrum(ref_shape, fft_shape),
        fft_shape)

    valid = tuple(slice(0, c) for c in corr_shape)
    denominator = cp.sqrt(cp.sum(ref ** 2) * local_sum_sq[valid])
    denominator = cp.where(denominator == 0, 1e-10, denominator)

    return cp.asnumpy(correlation[valid] / denominator)
