import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

from ..time_series_analysis.curves.framework import CurvesAnalysis
from .functions import *

class_name = "CurveQuantifications"

class CurveQuantifications:
    """
    Class to complete RF analysis via the sliding window technique
    and generate a corresponding parametric map.
    """
    def __init__(self, analysis_objs: CurvesAnalysis, function_names: List[str],
                 output_path: str, **kwargs):
        """
        Args:
            analysis_objs (CurvesAnalysis): The analysis object containing the curves.
            function_names (List[str]): List of function names to apply for quantification.
            output_path (str): The path to save the output CSV file.
            **kwargs: Additional keyword arguments for the quantification functions.
        """
        assert isinstance(analysis_objs, CurvesAnalysis), 'analysis_objs must be a CurvesAnalysis'
        assert isinstance(function_names, list), 'function_names must be a list of function names'
        assert isinstance(output_path, str) or output_path is None, 'output_path must be a string'
        
        self.analysis_objs = analysis_objs
        self.function_names = function_names
        self.output_path = output_path
        self.kwargs = kwargs

        self.determine_func_order()

    def determine_func_order(self):
        """Determine the order of functions to be applied to the data.
        
        This function is called in the constructor and sets the order of functions
        to be applied to the data based on the provided function names.
        """
        self.ordered_funcs = []; self.ordered_func_names = []; self.results_names = []
        
        def process_deps(func_name):
            if func_name in self.ordered_func_names:
                return
            if func_name in globals():
                # Handle function dependencies and outputs
                function = globals()[func_name]
                deps = getattr(function, 'dependencies', [])
                for dep in deps:
                    process_deps(dep)
            else:
                raise ValueError(f"Function '{func_name}' not found!")
            
            self.ordered_funcs.append(function)
            self.ordered_func_names.append(func_name)

        for function_name in self.function_names:
            process_deps(function_name)

    def compute_quantifications(self):
        """
        Compute the quantifications based on the provided function names.
        
        Returns:
            Dict[str, float]: A dictionary containing the computed quantifications.
        """
        self.data_dict = [{} for _ in range(len(self.analysis_objs.curves))]

        for curves, data_dict in zip(self.analysis_objs.curves, self.data_dict):
            data_dict['Scan Name'] = self.analysis_objs.image_data.scan_name
            data_dict['Segmentation Name'] = self.analysis_objs.seg_data.seg_name
            if curves.get('Window-Axial Start Pix'):
                data_dict['Window-Axial Start Pix'] = curves['Window-Axial Start Pix']
                data_dict['Window-Sagittal Start Pix'] = curves['Window-Sagittal Start Pix']
                data_dict['Window-Axial End Pix'] = curves['Window-Axial End Pix']
                data_dict['Window-Sagittal End Pix'] = curves['Window-Sagittal End Pix']
                if curves.get('Window-Coronal Start Pix'):
                    data_dict['Window-Coronal Start Pix'] = curves['Window-Coronal Start Pix']
                    data_dict['Window-Coronal End Pix'] = curves['Window-Coronal End Pix']

            for func in self.ordered_funcs:
                func(self.analysis_objs, curves, data_dict, **self.kwargs)

        # Assert all data_dicts have the same keys
        key_sets = [set(d.keys()) for d in self.data_dict]
        first_keys = key_sets[0]
        assert all(keys == first_keys for keys in key_sets), "Not all data_dicts have the same keys"

        if self.output_path:
            assert self.output_path.endswith('.csv'), 'output_path must end with .csv to export to CSV format'
            df_all = pd.DataFrame(self.data_dict)
            df_all.to_csv(self.output_path, index=False)
