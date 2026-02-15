import numpy as np
from typing import Dict, List
from collections.abc import Iterable

from ...time_series_analysis.curves.framework import CurvesAnalysis
from ..decorators import required_kwargs
import _compute_firstorder_stats

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