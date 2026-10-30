"""
TIC + log-normal fit + wash rates for the MedSAM2 notebooks, run through the QuantUS
framework rather than re-implemented:

  - the curve itself is `CurvesAnalysis` with the 'tic' curve definition
    (src/time_series_analysis), fed one mask per frame; `use_gpu=True` swaps in a
    CuPy version of that curve definition (the masked mean over a 3D volume is the
    cost; the single small curve fit stays on the CPU);
  - the fit and the wash-in/wash-out rates are `curve_quantification_step` with
    'lognormal_fit_full' and 'wash_rates' (src/curve_quantification), whose results
    land in `quant.data_dict[0]`.

The framework's own `compute_curves` takes its masks from `seg_data`, so it cannot
score a mask that comes from somewhere else (MedSAM2's per-frame prediction), which
is why the frame loop is here and calls `extract_frame_features` directly. `src` must
be importable (engines/ceus on sys.path).
"""

import copy
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from src.curve_quantification.transforms import bolus_lognormal
from src.entrypoints import curve_quantification_step
from src.time_series_analysis.curves.framework import CurvesAnalysis


_gpu_frame = [None, None]     # last (host frame, flat float64 device copy): one upload serves every mask scored on it


def _tic_gpu(image_data, frame, mask, **kwargs):
    """
    CuPy twin of the framework's `tic` curve definition: mean of `frame` inside `mask`.

    Written as dot products rather than `sum`/boolean indexing: once torch has been
    imported, CuPy's CUB reductions fail to compile on large arrays here (NVRTC picks
    up torch's CUDA 13 headers), while cuBLAS dot does not need to compile anything.
    """
    import cupy as cp

    if _gpu_frame[0] is not frame:
        _gpu_frame[0], _gpu_frame[1] = frame, cp.asarray(frame).ravel().astype(cp.float64)
    m = (cp.asarray(mask).ravel() > 0).astype(cp.float64)
    n = float(cp.dot(m, m))
    return ["TIC"], [float(cp.dot(_gpu_frame[1], m)) / n if n else float("nan")]


def make_analysis(image_data, seg_data, frames=None, use_gpu=False):
    """
    A `CurvesAnalysis` for the 'tic' curve, ready for `extract_frame_features`. `frames`
    (indices into the series) trims its time axis to the frames that will be fed in.
    """
    analysis = CurvesAnalysis(image_data, seg_data, ["tic"])
    if frames is not None:
        analysis.time_arr = analysis.time_arr[np.asarray(list(frames), dtype=int)]
    if use_gpu:
        try:
            import cupy
            cupy.cuda.runtime.getDeviceCount()
            analysis.curve_funcs["tic"] = _tic_gpu
        except Exception as e:
            print(f"CuPy unavailable ({type(e).__name__}); computing the TIC on the CPU")
    return analysis


def tic_analysis(image_data, seg_data, mask_for_frame, frames=None, use_gpu=False, desc="Computing TIC"):
    """
    Mean-intensity TIC over `frames` (default: all) of `image_data.intensities_for_analysis`,
    ((X, Y, Z, T) or (T, H, W)), using `mask_for_frame(frame_ix)` as the ROI.
    Returns the `CurvesAnalysis`; the curve is `analysis.curves[0]["TIC"]`.
    """
    data = image_data.intensities_for_analysis
    n_frames = data.shape[-1] if data.ndim == 4 else data.shape[0]
    frames = list(range(n_frames)) if frames is None else list(frames)
    analysis = make_analysis(image_data, seg_data, frames, use_gpu)
    for pos, ix in enumerate(tqdm(frames, desc=desc)):
        frame = data[..., ix] if data.ndim == 4 else data[ix]
        analysis.extract_frame_features(frame, mask_for_frame(ix), pos)
    return analysis


def lognormal_quantification(analysis, n_frames_to_analyze=None):
    """
    Log-normal fit of the TIC on the full frame range, plus wash-in/wash-out rates;
    results are in `quant.data_dict[0]`.

    NaN frames (an empty mask: the lesion left the volume/plane) are dropped before
    fitting, as the framework's fit cannot take them. The wash rates are then slopes
    per *valid frame*, in TIC units, over the first `n_frames_to_analyze` of them
    (default: all). With fewer than 5 valid frames nothing is fitted and every
    parameter is NaN.
    """
    tic = np.asarray(analysis.curves[0]["TIC"], dtype=float)
    ok = np.isfinite(tic)
    n_valid = int(ok.sum())
    if n_valid < 5:
        print(f"    Too few valid points to fit ({n_valid}).")
        names = ["AUC", "PE", "TP", "MTT", "T0", "Mu", "Sigma", "PE_Ix"]
        data = {f"{n}_full_TIC": np.nan for n in names}
        data.update({"WashInRate_TIC": np.nan, "WashOutRate_TIC": np.nan})
        return SimpleNamespace(data_dict=[data])
    if (~ok).any():
        print(f"    skipping {int((~ok).sum())} NaN frame(s) in the fit")

    valid = copy.copy(analysis)
    valid.curves = [{"TIC": tic[ok].tolist()}]            # a list: the framework's fit shifts its input in place
    valid.time_arr = analysis.time_arr[ok]
    n_wash = n_valid if n_frames_to_analyze is None else min(int(n_frames_to_analyze), n_valid)
    return curve_quantification_step(
        valid, ["lognormal_fit_full", "wash_rates"], None,
        curves_to_fit=["TIC"], tic_name="TIC", n_frames_to_analyze=n_wash,
    )


def fitted_curve(analysis, quant):
    """The fitted curve in the TIC's own units (baseline added back), or all-NaN if the fit failed."""
    d = quant.data_dict[0]
    tic = np.asarray(analysis.curves[0]["TIC"], dtype=float)
    time = analysis.time_arr
    if np.isnan(d["AUC_full_TIC"]):
        return np.full_like(time, np.nan, dtype=float)
    # lognormal_fit_full subtracts the curve minimum before fitting
    return bolus_lognormal(
        time, d["AUC_full_TIC"], d["Mu_full_TIC"], d["Sigma_full_TIC"], d["T0_full_TIC"]
    ) + np.nanmin(tic)


def fit_quality(analysis, quant):
    """Fit R^2 plus curve roughness and SNR, alongside the fitted parameters and wash rates."""
    d = quant.data_dict[0]
    tic = np.asarray(analysis.curves[0]["TIC"], dtype=float)
    fitted = fitted_curve(analysis, quant)
    valid = np.isfinite(fitted) & np.isfinite(tic)
    ss_res = np.sum((tic[valid] - fitted[valid]) ** 2)
    ss_tot = np.sum((tic[valid] - tic[valid].mean()) ** 2)
    tic_ok = tic[np.isfinite(tic)]
    return {
        "r2": 1 - ss_res / (ss_tot + 1e-8),
        "roughness": np.mean(np.abs(np.diff(tic_ok))),
        "snr": d["PE_full_TIC"] / (np.std(tic_ok[:5]) + 1e-8),
        "auc": d["AUC_full_TIC"], "pe": d["PE_full_TIC"], "tp": d["TP_full_TIC"],
        "mtt": d["MTT_full_TIC"], "t0": d["T0_full_TIC"],
        "wash_in": d["WashInRate_TIC"], "wash_out": d["WashOutRate_TIC"],
    }
