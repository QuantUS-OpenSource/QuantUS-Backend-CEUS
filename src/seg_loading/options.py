from pathlib import Path

from argparse import ArgumentParser

from .functions import *

def seg_loader_args(parser: ArgumentParser):
    parser.add_argument('seg_path', type=str, help='Path to segmentation file')
    parser.add_argument('--seg_loader', type=str, default='pkl_roi',
                        help='Segmentation loader to use. Available options: ' + ', '.join(get_seg_loaders().keys()))
    parser.add_argument('--seg_loader_kwargs', type=str, default='{}',
                        help='Segmentation kwargs in JSON format needed for analysis class.')
    
def get_seg_loaders() -> dict:
    """Get scan loaders for the CLI.
    
    Returns:
        dict: Dictionary of scan loaders.
    """
    functions = {name: obj for name, obj in globals().items() if callable(obj) and obj.__module__ == __package__ + '.functions'}
    return functions