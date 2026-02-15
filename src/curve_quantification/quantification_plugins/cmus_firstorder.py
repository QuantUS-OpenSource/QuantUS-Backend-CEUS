import numpy as np
from typing import Dict, List, Any
from collections.abc import Iterable

from ...time_series_analysis.curves.framework import CurvesAnalysis
from ..decorators import required_kwargs, dependencies
import _compute_firstorder_stats

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