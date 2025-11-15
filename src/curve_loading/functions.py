import pandas as pd

from ..data_objs.image import UltrasoundImage
from ..data_objs.seg import CeusSeg
from ..time_series_analysis.curves.framework import CurvesAnalysis

def load_ttc_curves(curves_path: str, **kwargs) -> CurvesAnalysis:
    """
    Load TTC curves from a directory containing CSV files.

    Args:
        curves_path (str): The path to the CSV file containing TTC curves.
    
    Returns:
        CurvesAnalysis: An instance of CurvesAnalysis containing the loaded curves.
    """
    curves_df = pd.read_csv(curves_path)
    image_data = UltrasoundImage(curves_df['Scan Name'].iloc[0])
    seg_data = CeusSeg()
    seg_data.seg_name = curves_df['Segmentation Name'].iloc[0]

    analysis_obj = CurvesAnalysis(image_data, seg_data, [])
    analysis_obj.curves_output_path = curves_path
    
    analysis_obj.time_arr = curves_df['Time Array'].values

    for key in curves_df.columns[3:]:
        analysis_obj.curves[key] = curves_df[key].values

    return analysis_obj
