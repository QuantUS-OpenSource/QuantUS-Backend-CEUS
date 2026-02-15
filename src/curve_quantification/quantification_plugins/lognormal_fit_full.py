import numpy as np
from scipy.stats import skew, kurtosis, entropy
from typing import Dict, List, Any
from collections.abc import Iterable

from ...time_series_analysis.curves.framework import CurvesAnalysis
from ..decorators import required_kwargs, dependencies
from ..transforms import fit_lognormal_curve

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