"""
CuPy handle for the 3D masked NCC in motion_compensation_3d.

GPU_AVAILABLE is False when CuPy or a CUDA device is missing, and the caller
stays on the NumPy path.
"""

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    GPU_AVAILABLE = False
