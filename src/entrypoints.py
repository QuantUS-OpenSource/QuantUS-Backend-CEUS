import copy
import os

import numpy as np

from .data_objs import UltrasoundImage, CeusSeg
from .image_loading.options import get_scan_loaders
from .image_preprocessing.options import get_im_preproc_funcs, get_required_im_preproc_kwargs
from .seg_loading.options import get_seg_loaders
from .seg_preprocessing.options import get_seg_preproc_funcs, get_required_seg_preproc_kwargs
from .time_series_analysis.options import get_analysis_types, get_required_kwargs
from .time_series_analysis.curves.framework import CurvesAnalysis
from .curve_loading.options import get_curves_loaders
from .curve_quantification.framework import CurveQuantifications
from .curve_quantification.options import get_quantification_funcs
from .visualizations.options import get_visualization_types
from .visualizations.paramap.framework import ParamapVisualizations

def scan_loading_step(scan_type: str, scan_path: str, **scan_loader_kwargs) -> UltrasoundImage:
    """Load the scan data using the specified scan loader.

    Args:
        scan_type (str): The type of scan loader to use.
        scan_path (str): Path to the scan file.
        phantom_path (str): Path to the phantom file.
        **scan_loader_kwargs: Additional keyword arguments for the scan loader.

    Returns:
        UltrasoundRfImage: Loaded ultrasound RF image data.
    """
    scan_loaders = get_scan_loaders()
    
    # Find the scan loader
    try:
        scan_loader = scan_loaders[scan_type]['cls']
        if scan_loaders[scan_type]['file_exts'] == ["FOLDER"]:
            assert os.path.isdir(scan_path), "Input path must be a folder!"
        else:
            assertions = [scan_path.endswith(ext) for ext in scan_loaders[scan_type]['file_exts']]
            assert max(assertions), f"Scan file must end with {', '.join(scan_loaders[scan_type]['file_exts'])}"
    except KeyError:
        print(f'Parser "{scan_type}" is not available!')
        print(f"Available parsers: {', '.join(scan_loaders.keys())}")
        return 1
    
    image_data: UltrasoundImage = scan_loader(scan_path, **scan_loader_kwargs)
    return image_data

def scan_preprocessing_step(preproc_func_names: list[str], image_data: UltrasoundImage, **preproc_kwargs) -> UltrasoundImage:
    """Preprocess the scan data using the specified preprocessing functions.

    Args:
        preproc_func_names (list[str]): List of preprocessing function names to apply.
        image_data (UltrasoundImage): Loaded ultrasound image data.
        **preproc_kwargs: Additional keyword arguments for the preprocessing functions.

    Returns:
        UltrasoundImage: Preprocessed ultrasound image data.
    """
    if preproc_func_names == ['none']:
        return image_data
    preproc_funcs = get_im_preproc_funcs()
    for func_name in preproc_func_names:
        if func_name == 'none':
            continue
        if func_name not in preproc_funcs.keys():
            raise ValueError(f"Function '{func_name}' not found in preprocessing functions.\nAvailable functions: {', '.join(preproc_funcs.keys())}")
        required_kwargs = get_required_im_preproc_kwargs([func_name])
        for kwarg in required_kwargs:
            if kwarg not in preproc_kwargs:
                raise ValueError(f"preproc_kwargs: Missing required keyword argument '{kwarg}' for function '{func_name}'.")
    
    for func_name in preproc_func_names:
        func = preproc_funcs[func_name]
        image_data = func(image_data, **preproc_kwargs)

    return image_data

def seg_loading_step(seg_type: str, image_data: UltrasoundImage, seg_path: str,
                     scan_path: str, **seg_loader_kwargs) -> CeusSeg:
    """Load the segmentation data using the specified segmentation loader.

    Args:
        seg_type (str): The type of segmentation loader to use.
        image_data (UltrasoundImage): Loaded ultrasound image data.
        seg_path (str): Path to the segmentation file.
        scan_path (str): Path to the scan file.
        phantom_path (str): Path to the phantom file.
        **seg_loader_kwargs: Additional keyword arguments for the segmentation loader.

    Returns:
        CeusSeg: Loaded segmentation data.
    """
    seg_loaders = get_seg_loaders()
    
    # Find the segmentation loader
    try:
        seg_loader = seg_loaders[seg_type]
    except KeyError:
        print(f'Segmentation loader "{seg_type}" is not available!')
        print(f"Available segmentation loaders: {', '.join(seg_loaders.keys())}")
        return 1
    
    return seg_loader(image_data, seg_path, scan_path=scan_path, **seg_loader_kwargs)

def seg_preprocessing_step(preproc_func_names: list[str], image_data: UltrasoundImage, seg_data: CeusSeg, **preproc_kwargs) -> CeusSeg:
    """Preprocess the segmentation data using the specified preprocessing functions.

    Args:
        preproc_func_names (list[str]): List of preprocessing function names to apply.
        seg_data (CeusSeg): Loaded segmentation data.
        **preproc_kwargs: Additional keyword arguments for the preprocessing functions.

    Returns:
        CeusSeg: Preprocessed segmentation data.
    """
    if preproc_func_names == ['none']:
        return seg_data
    preproc_funcs = get_seg_preproc_funcs()
    for func_name in preproc_func_names:
        if func_name == 'none':
            continue
        if func_name not in preproc_funcs.keys():
            raise ValueError(f"Function '{func_name}' not found in preprocessing functions.\nAvailable functions: {', '.join(preproc_funcs.keys())}")
        required_kwargs = get_required_seg_preproc_kwargs([func_name])
        for kwarg in required_kwargs:
            if kwarg not in preproc_kwargs:
                raise ValueError(f"preproc_kwargs: Missing required keyword argument '{kwarg}' for function '{func_name}'.")
    
    for func_name in preproc_func_names:
        func = preproc_funcs[func_name]
        seg_data = func(image_data, seg_data, **preproc_kwargs)

    return seg_data

def analysis_step(analysis_type: str, image_data: UltrasoundImage, seg_data: CeusSeg, 
                  analysis_funcs: list, **analysis_kwargs) -> CurvesAnalysis:
    """Perform analysis using the specified analysis type.
    
    Args:
        analysis_type (str): The type of analysis to perform.
        image_data (UltrasoundImage): Loaded ultrasound image data.
        config (RfAnalysisConfig): Loaded analysis configuration.
        seg_data (CeusSeg): Loaded segmentation data.
        analysis_funcs (list): List of analysis functions to apply.
        **analysis_kwargs: Additional keyword arguments for the analysis.
    Returns:
        CurvesAnalysis: Analysis object containing the results.
    """
    all_analysis_types, all_analysis_funcs = get_analysis_types()
    
    # Find the analysis class
    try:
        analysis_class = all_analysis_types[analysis_type]
    except KeyError:
        print(f'Analysis type "{analysis_type}" is not available!')
        print(f"Available analysis types: {', '.join(all_analysis_types.keys())}")
        return 1
    
    # Check analysis setup
    for name in analysis_funcs:   
        if name not in all_analysis_funcs:
            raise ValueError(f"Function '{name}' not found in {analysis_type} analysis type.\nAvailable functions: {', '.join(all_analysis_funcs.keys())}")
        required_analysis_kwargs = get_required_kwargs(analysis_type, [name])
        for kwarg in required_analysis_kwargs:
            if (kwarg == 'cor_vox_len' or kwarg == 'cor_vox_ovrlp') and image_data.intensities_for_analysis.ndim == 3:
                # Skip coronal requirements for 2D+time data
                continue
            if kwarg not in analysis_kwargs:
                raise ValueError(f"analysis_kwargs: Missing required keyword argument '{kwarg}' for function '{name}' in {analysis_type} analysis type.")
            
    # Perform analysis
    analyzed_image_data = copy.deepcopy(image_data)
    
    analysis_obj = analysis_class(analyzed_image_data, seg_data, analysis_funcs, **analysis_kwargs)
    analysis_obj.compute_curves()
    
    return analysis_obj

def load_curves_step(curves_path: str, curves_loader_type: str,
                     **kwargs) -> CurvesAnalysis:
    """Load TTC curves from a specified path.
    
    Args:
        curves_path (str): Path to the CSV file containing TTC curves.
        **kwargs: Additional keyword arguments for loading curves.
    
    Returns:
        CurvesAnalysis: Loaded TTC curves analysis object.
    """
    curves_loaders = get_curves_loaders()

    try:
        curve_loader = curves_loaders[curves_loader_type]
    except KeyError:
        print(f'Curve loader "{curves_loader_type}" is not available!')
        print(f"Available curve loaders: {', '.join(curves_loaders.keys())}")
        return 1

    return curve_loader(curves_path, **kwargs)

def curve_quantification_step(analysis_obj: CurvesAnalysis, function_names: list[str],
                          output_path: str, **kwargs) -> CurveQuantifications:
    """
    Perform curve quantifications using the specified analysis objects and function names.

    Args:
        analysis_obj (CurvesAnalysis): The analysis object containing the curves.
        function_names (List[str]): List of function names to apply for quantification.
        output_path (str): The path to save the output CSV file.
        **kwargs: Additional keyword arguments for the quantification functions.

    Returns:
        Dict[str, float]: A dictionary containing the computed quantifications.
    """
    quant_funcs = get_quantification_funcs()

    if not len(function_names):
        function_names = list(quant_funcs.keys())

    for func_name in function_names:
        if func_name not in quant_funcs.keys():
            raise ValueError(f"Function '{func_name}' not found in quantification functions.\nAvailable functions: {', '.join(quant_funcs.keys())}")
        required_kwargs = getattr(quant_funcs[func_name], 'kwarg_names', [])
        for kwarg in required_kwargs:
            if kwarg not in kwargs:
                raise ValueError(f"kwargs: Missing required keyword argument '{kwarg}' for function '{func_name}'.")

    curve_quant = CurveQuantifications(analysis_obj, function_names, output_path, **kwargs)
    curve_quant.compute_quantifications()

    return curve_quant

def visualization_step(quants_obj: CurveQuantifications, vis_type: str, params: list[str], custom_funcs: list[str], **kwargs) -> None:
    """Perform visualizations using the specified parameters and visualization functions.
    
    Args:
        quants_obj (CurveQuantifications): The quantification object containing the curves.
        vis_type (str): The type of visualization to perform.
        params (list[str]): List of parameters to visualize.
        custom_funcs (list[str]): List of custom visualization functions to apply.
        **kwargs: Additional keyword arguments for the visualizations.
    """

    all_visualization_types, all_visualization_funcs = get_visualization_types()

    # Check visualization inputs
    assert vis_type in all_visualization_types.keys(), f"Visualization type '{vis_type}' not found. Available types: {', '.join(all_visualization_types.keys())}"
    for func_name in custom_funcs:
        if func_name not in all_visualization_funcs[vis_type].keys():
            raise ValueError(f"Function '{func_name}' not found in visualization functions.\nAvailable functions: {', '.join(all_visualization_funcs.keys())}")

    vis_obj = ParamapVisualizations(quants_obj, params, custom_funcs, **kwargs)
    vis_obj.generate_visualizations()

    return vis_obj