import numpy as np
from scipy.stats import skew, kurtosis, entropy
from typing import Dict, List, Any
from collections.abc import Iterable

from ...time_series_analysis.curves.framework import CurvesAnalysis
from ..decorators import required_kwargs, dependencies
from ..transforms import fit_lognormal_curve

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