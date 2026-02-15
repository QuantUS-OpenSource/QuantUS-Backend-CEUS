from typing import Dict, List
from collections.abc import Iterable

from ...time_series_analysis.curves.framework import CurvesAnalysis
from ._compute_firstorder_stats import _compute_firstorder_stats

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