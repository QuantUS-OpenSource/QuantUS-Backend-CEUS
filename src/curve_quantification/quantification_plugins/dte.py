import numpy as np
from typing import Dict, List
from collections.abc import Iterable

from ...time_series_analysis.curves.framework import CurvesAnalysis
from ..decorators import required_kwargs

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