import numpy as np
from typing import Dict, List

from ...time_series_analysis.curves.framework import CurvesAnalysis
from ..decorators import required_kwargs

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