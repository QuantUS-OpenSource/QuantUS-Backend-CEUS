import numpy as np
from scipy.stats import skew, kurtosis, entropy
from typing import Dict, List, Any
from collections.abc import Iterable

from ...time_series_analysis.curves.framework import CurvesAnalysis
from ..decorators import required_kwargs, dependencies
from ..transforms import fit_lognormal_curve

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