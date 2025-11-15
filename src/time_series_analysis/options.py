import importlib
import inspect
from pathlib import Path
from typing import Tuple

from argparse import ArgumentParser

def analysis_args(parser: ArgumentParser):
    parser.add_argument('analysis_type', type=str, default='spectral_paramap',
                        help='Analysis type to complete. Available analysis types: ' + ', '.join(get_analysis_types()[0].keys()))
    parser.add_argument('--analysis_kwargs', type=str, default='{}',
                        help='Analysis kwargs in JSON format needed for analysis class.')
    
def get_required_kwargs(analysis_type: str, analysis_funcs: list) -> list:
    """Get required kwargs for a given list of analysis functions.

    Args:
        analysis_type (str): the type of analysis to perform.
        analysis_funcs (list): list of analysis functions to apply.

    Returns:
        list: List of required kwargs for the specified analysis functions.
    """
    
    all_analysis_types, all_analysis_funcs = get_analysis_types()
    analysis_objs = [all_analysis_types[analysis_type]] + [all_analysis_funcs[name] for name in analysis_funcs]
    required_kwargs = []

    for obj in analysis_objs:
        required_kwargs.extend(getattr(obj, 'required_kwargs', []))
    required_kwargs = list(set(required_kwargs))  # Remove duplicates
    
    return required_kwargs
    
def get_analysis_types() -> Tuple[dict, dict]:
    """Get analysis types for the CLI.
    
    Returns:
        dict: Dictionary of analysis types.
        dict: Dictionary of analysis functions for each type.
    """
    types = {}
    current_dir = Path(__file__).parent
    for folder in current_dir.iterdir():
        # Check if the item is a directory and not a hidden directory
        if folder.is_dir() and not folder.name.startswith("_") and folder.name != "curve_types":
            try:
                # Attempt to import the module
                module = importlib.import_module(
                    __package__ + f".{folder.name}.framework"
                )
                entry_class_name = ''.join(word.capitalize() for word in folder.name.split('_')) + "Analysis"
                entry_class = getattr(module, entry_class_name, None)
                if entry_class:
                    types[folder.name] = entry_class
            except ModuleNotFoundError as e:
                print(f"Module not found: {e}")
                # Handle the case where the module cannot be found
                pass
            
    module = importlib.import_module(__package__ + '.curve_types.functions')
    module_file = module.__file__
    defined_funcs = set()
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("_") and inspect.getsourcefile(obj) == module_file:
            defined_funcs.add(name)
    functions = {name: obj for name, obj in inspect.getmembers(module, inspect.isfunction) if name in defined_funcs}
            
    return types, functions
