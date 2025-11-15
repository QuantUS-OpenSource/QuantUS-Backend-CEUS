import numpy as np
from scipy.stats import skew, kurtosis, entropy
from typing import Dict, List, Any
from collections.abc import Iterable

from ..time_series_analysis.curves.framework import CurvesAnalysis
from .decorators import required_kwargs, dependencies
from .transforms import fit_lognormal_curve

def _compute_firstorder_stats(curve: np.ndarray, data_dict: dict, name_prefix: str, name_suffix: str = '') -> None:
    """
    Compute first-order statistics for a given curve and store them in the data dictionary.
    
    Args:
        curve (np.ndarray): The curve data to analyze.
        data_dict (dict): Dictionary to store the computed statistics.
        name_prefix (str): Prefix for the keys in the data dictionary.
        name_suffix (str): Suffix for the keys in the data dictionary.
    """
    if len(curve) == 0:
        return

    data_dict[f'{name_prefix}Mean{name_suffix}'] = np.mean(curve)
    data_dict[f'{name_prefix}Std{name_suffix}'] = np.std(curve)
    data_dict[f'{name_prefix}Max{name_suffix}'] = np.max(curve)
    data_dict[f'{name_prefix}Min{name_suffix}'] = np.min(curve)
    data_dict[f'{name_prefix}Median{name_suffix}'] = np.median(curve)
    data_dict[f'{name_prefix}Variance{name_suffix}'] = np.var(curve)
    if np.var(curve) < 1e-10:
        data_dict[f'{name_prefix}Skewness{name_suffix}'] = 0.0
        data_dict[f'{name_prefix}Kurtosis{name_suffix}'] = 3.0  # Normal kurtosis
    else:
        data_dict[f'{name_prefix}Skewness{name_suffix}'] = skew(curve)
        data_dict[f'{name_prefix}Kurtosis{name_suffix}'] = kurtosis(curve)
    data_dict[f'{name_prefix}Range{name_suffix}'] = np.max(curve) - np.min(curve)
    data_dict[f'{name_prefix}InterquartileRange{name_suffix}'] = np.percentile(curve, 75) - np.percentile(curve, 25)
    data_dict[f'{name_prefix}Entropy{name_suffix}'] = entropy(curve)
    data_dict[f'{name_prefix}Energy{name_suffix}'] = np.sum(curve ** 2)

@required_kwargs('start_time', 'end_time')
def first_order_select(analysis_objs: CurvesAnalysis, curves: Dict[str, List[float]], data_dict: dict, **kwargs) -> None:
    """
    Compute first-order statistics from the analysis objects and store them in the data dictionary on a selected frame range.
    """
    assert isinstance(analysis_objs, CurvesAnalysis), 'analysis_objs must be a CurvesAnalysis'
    assert isinstance(data_dict, dict), 'data_dict must be a dictionary'

    start_time = kwargs.get('start_time', 0)
    end_time = kwargs.get('end_time', len(analysis_objs.time_arr))

    assert 0 <= start_time < end_time <= len(analysis_objs.time_arr), 'Invalid frame range. Start frame must be >= 0, end frame must be <= total frames, and start frame must be < end frame.'
    
    start_frame = int(np.searchsorted(analysis_objs.time_arr, start_time, side='left'))
    end_frame = int(np.searchsorted(analysis_objs.time_arr, end_time, side='right'))

    # Compute first-order statistics
    for name, curve in curves.items():
        if not isinstance(curve, Iterable) or isinstance(curve, str):
            continue
        curve = np.array(curve[start_frame:end_frame])
        _compute_firstorder_stats(curve, data_dict, name_prefix='', name_suffix=f'_select_{name}')
        
def first_order_full(analysis_objs: CurvesAnalysis, curves: Dict[str, List[float]], data_dict: dict, **kwargs) -> None:
    """
    Compute first-order statistics from the analysis objects and store them in the data dictionary on the full frame range.
    """
    assert isinstance(analysis_objs, CurvesAnalysis), 'analysis_objs must be a CurvesAnalysis'
    assert isinstance(data_dict, dict), 'data_dict must be a dictionary'

    # Compute first-order statistics
    for name, curve in curves.items():
        if not isinstance(curve, Iterable) or isinstance(curve, str):
            continue
        _compute_firstorder_stats(curve, data_dict, name_prefix='', name_suffix=f'_full_{name}')

@required_kwargs('curves_to_fit', 'start_time', 'end_time')
def lognormal_fit_select(analysis_objs: CurvesAnalysis, curves: Dict[str, List[float]], data_dict: dict, **kwargs) -> None:
    """
    Fit a log-normal distribution to the given curves on a selected frame range.
    """
    curves_to_fit = kwargs.get('curves_to_fit', [])
    start_time = kwargs.get('start_time', 0)
    end_time = kwargs.get('end_time', len(analysis_objs.time_arr))

    assert 0 <= start_time < end_time <= len(analysis_objs.time_arr), 'Invalid frame range. Start frame must be >= 0, end frame must be <= total frames, and start frame must be < end frame.'
    
    start_frame = int(np.searchsorted(analysis_objs.time_arr, start_time, side='left'))
    end_frame = int(np.searchsorted(analysis_objs.time_arr, end_time, side='right'))

    all_curve_names = curves.keys()
    for curve_name in curves_to_fit:
        matching_names = [name for name in all_curve_names if curve_name.lower() in name.lower()]
        for name in matching_names:
            if not isinstance(curves[name], Iterable) or  isinstance(curves[name], str):
                continue
            curve = curves[name][start_frame:end_frame]
            auc, pe, tp, mtt, t0, mu, sigma, pe_loc = fit_lognormal_curve(
                analysis_objs.time_arr[start_frame:end_frame], curve)
            data_dict[f'AUC_select_{name}'] = auc
            data_dict[f'PE_select_{name}'] = pe
            data_dict[f'TP_select_{name}'] = tp
            data_dict[f'MTT_select_{name}'] = mtt
            data_dict[f'T0_select_{name}'] = t0 if t0 >= 0 else 0
            data_dict[f'Mu_select_{name}'] = mu
            data_dict[f'Sigma_select_{name}'] = sigma
            data_dict[f'PE_Ix_select_{name}'] = pe_loc
            
@required_kwargs('curves_to_fit')
def lognormal_fit_full(analysis_objs: CurvesAnalysis, curves: Dict[str, List[float]], data_dict: dict, **kwargs) -> None:
    """
    Fit a log-normal distribution to the given curves on the full frame range.
    """
    curves_to_fit = kwargs.get('curves_to_fit', [])

    all_curve_names = curves.keys()
    for curve_name in curves_to_fit:
        matching_names = [name for name in all_curve_names if curve_name.lower() in name.lower()]
        for name in matching_names:
            if not isinstance(curves[name], Iterable) or  isinstance(curves[name], str):
                continue
            curve = curves[name]
            auc, pe, tp, mtt, t0, mu, sigma, pe_loc = fit_lognormal_curve(
                analysis_objs.time_arr, curve)
            data_dict[f'AUC_full_{name}'] = auc
            data_dict[f'PE_full_{name}'] = pe
            data_dict[f'TP_full_{name}'] = tp
            data_dict[f'MTT_full_{name}'] = mtt
            data_dict[f'T0_full_{name}'] = t0 if t0 >= 0 else 0
            data_dict[f'Mu_full_{name}'] = mu
            data_dict[f'Sigma_full_{name}'] = sigma
            data_dict[f'PE_Ix_full_{name}'] = pe_loc

@dependencies('lognormal_fit')
@required_kwargs('tic_name', 'curves_to_fit', 'n_frames_to_analyze')
def wash_rates(analysis_objs: CurvesAnalysis, curves: Dict[str, List[float]], data_dict: dict,  **kwargs) -> None:
    """
    Compute wash-in and wash-out rates from the curves.
    """
    assert isinstance(analysis_objs, CurvesAnalysis), 'analysis_objs must be a CurvesAnalysis'
    assert isinstance(data_dict, dict), 'data_dict must be a dictionary'

    curves_to_fit = kwargs.get('curves_to_fit', [])
    tic_name = kwargs.get('tic_name', None)
    n_frames_to_analyze = kwargs.get('n_frames_to_analyze', len(analysis_objs.time_arr))
    fitted_curves = [name for curve_name in curves_to_fit for name in curves.keys() 
                     if curve_name.lower() in name.lower()]
    
    assert tic_name is not None, 'tic_name must be provided'
    assert tic_name in curves, f'{tic_name} not found in curves'
    assert tic_name in fitted_curves, f'{tic_name} not found in fitted curves'

    pe_ix = data_dict[f'PE_Ix_{tic_name}']

    for name, curve in curves.items():
        if not isinstance(curve, Iterable) or isinstance(curve, str):
            continue
        # Compute wash-in rate as the slope of the line of best fit for curve[:pe_ix]
        if pe_ix > 1:
            cutoff_ix = min(pe_ix, n_frames_to_analyze)
            x_in = np.arange(cutoff_ix)
            y_in = curve[:cutoff_ix]
            coeffs = np.polyfit(x_in, y_in, 1)
            wash_in_rate = coeffs[0]
        else:
            wash_in_rate = np.nan
        # Compute wash-out rate as the slope of the line of best fit for curve[pe_ix:]
        if pe_ix < n_frames_to_analyze - 1:
            x_out = np.arange(pe_ix, n_frames_to_analyze)
            y_out = curve[pe_ix:n_frames_to_analyze]
            coeffs = np.polyfit(x_out, y_out, 1)
            wash_out_rate = coeffs[0]
        else:
            wash_out_rate = np.nan
        data_dict[f'WashInRate_{name}'] = wash_in_rate
        data_dict[f'WashOutRate_{name}'] = wash_out_rate

@required_kwargs('tic_name')
def dte(analysis_objs: CurvesAnalysis, curves: Dict[str, List[float]], data_dict: dict, **kwargs) -> None:
    """
    Compute the DTE (Dynamic Time Elasticity) from the curves.
    """
    assert isinstance(analysis_objs, CurvesAnalysis), 'analysis_objs must be a CurvesAnalysis'
    assert isinstance(data_dict, dict), 'data_dict must be a dictionary'

    tic_name = kwargs.get('tic_name', None)
    assert tic_name is not None, 'tic_name must be provided'
    assert tic_name in curves, f'{tic_name} not found in curves'

    flash_ix = np.argmax(curves[tic_name])
    preflash_ix = flash_ix - 5 if flash_ix - 5 >= 0 else 0
    postflash_ix = flash_ix + 5 if flash_ix + 5 < len(curves[tic_name]) else len(curves[tic_name]) - 1

    for curve_name, curve in curves.items():
        if not isinstance(curve, Iterable) or isinstance(curve, str):
            continue
        data_dict[f'DTE_{curve_name}'] = np.median(curve[preflash_ix-4:preflash_ix+1]) - np.median(curve[postflash_ix: postflash_ix+5])

@dependencies('lognormal_fit')
@required_kwargs('tic_name')
def cmus_firstorder(analysis_objs: CurvesAnalysis, curves: Dict[str, List[float]], data_dict: Dict[str,Any], **kwargs) -> None:
    """
    Compute first-order statistics for each of the 3 major sections
    of a C-MUS curve: wash-in, wash-out before flash, and post-flash.
    """
    assert isinstance(analysis_objs, CurvesAnalysis), 'analysis_objs must be a CurvesAnalysis'
    assert isinstance(data_dict, dict), 'data_dict must be a dictionary'

    tic_name = kwargs.get('tic_name', None)
    assert tic_name is not None, 'tic_name must be provided'
    assert tic_name in curves, f'{tic_name} not found in curves'

    flash_ix = np.argmax(curves[tic_name])
    preflash_ix = flash_ix - 5 if flash_ix - 5 >= 0 else 0
    postflash_ix = flash_ix + 5 if flash_ix + 5 < len(curves[tic_name]) else len(curves[tic_name]) - 1
    pe_ix = data_dict[f'PE_Ix_{tic_name}']; pe_ix = max(pe_ix, 0)
    pe_ix = pe_ix if pe_ix < postflash_ix else postflash_ix - 1

    for curve_name, curve in curves.items():
        if not isinstance(curve, Iterable) or isinstance(curve, str):
            continue
        for section, ix_range in zip(['WashIn', 'WashOutPreFlash', 'PostFlash'], 
                                 [(0, pe_ix), (pe_ix, preflash_ix), (postflash_ix, len(curve))]):
            section_curve = np.array(curve[ix_range[0]:ix_range[1]])
            _compute_firstorder_stats(section_curve, data_dict, name_prefix=f'{section}_', name_suffix=f'_{curve_name}')

@required_kwargs('tic_name')
def auc_no_fit(analysis_objs: CurvesAnalysis, curves: Dict[str, List[float]], data_dict: dict, **kwargs) -> None:
    """Compute the area under the curve (AUC) of the entire TIC without fitting a log-normal curve.
    """
    assert isinstance(analysis_objs, CurvesAnalysis), 'analysis_objs must be a CurvesAnalysis'
    assert isinstance(data_dict, dict), 'data_dict must be a dictionary'

    tic_name = kwargs.get('tic_name', None)
    assert tic_name is not None, 'tic_name must be provided'
    assert tic_name in curves, f'{tic_name} not found in curves'

    curve = np.array(curves[tic_name])
    curve /= np.max(curve)  # Normalize the curve
    data_dict[f'AUC_NoFit_{tic_name}'] = np.trapz(curve, analysis_objs.time_arr)
