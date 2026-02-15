import importlib
from pathlib import Path

from argparse import ArgumentParser

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
    functions = {}
    for file in (Path(__file__).parent / "seg_loaders").iterdir():
        module = importlib.import_module(f'.seg_loaders.{file.stem}', package=__package__)
        for name, obj in vars(module).items():
            if callable(obj) and obj.__module__ == module.__name__:
                functions[name] = obj
    return functions