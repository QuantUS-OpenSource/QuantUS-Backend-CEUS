from pathlib import Path

from argparse import ArgumentParser

from .functions import *

def seg_preproc_args(parser: ArgumentParser):
    parser.add_argument('--seg_preproc_func', type=str, default='none',
                        help='Segmentation preprocessing function to use. Available options: ' + ', '.join(get_seg_preproc_funcs().keys()))
    parser.add_argument('--seg_preproc_kwargs', type=str, default='{}',
                        help='Segmentation preprocessing kwargs in JSON format needed for the preprocessing function.')

def get_seg_preproc_funcs() -> dict:
    """Get preprocessing functions for the CLI.

    Returns:
        dict: Dictionary of preprocessing functions.
    """
    functions = {name: obj for name, obj in globals().items() if callable(obj) and obj.__module__ == __package__ + '.functions'}
    return functions

def get_required_seg_preproc_kwargs(preproc_func_names: list) -> list:
    """Get required kwargs for a given list of preprocessing functions.

    Args:
        preproc_func_names (list): list of preprocessing function names to apply.

    Returns:
        list: List of required kwargs for the specified preprocessing functions.
    """
    preproc_funcs = get_seg_preproc_funcs()
    required_kwargs = []

    for func_name in preproc_func_names:
        func = preproc_funcs[func_name]
        required_kwargs.extend(getattr(func, 'required_kwargs', []))
    
    required_kwargs = list(set(required_kwargs))  # Remove duplicates
    return required_kwargs
