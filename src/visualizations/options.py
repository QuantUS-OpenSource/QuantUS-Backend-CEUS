import importlib
from pathlib import Path
from typing import Tuple

from argparse import ArgumentParser

def visualization_args(parser: ArgumentParser):
    parser.add_argument('visualization_type', type=str, default='paramap_drawing',
                        help='Visualization type to use. Available visualization types: ' + ', '.join(get_visualization_types().keys()))
    parser.add_argument('--visualization_kwargs', type=str, default='{}',
                        help='Visualization kwargs in JSON format needed for visualization class.')
    parser.add_argument('--visualization_output_path', type=str, default='visualizations.pkl',
                        help='Path to output visualization class instance')
    parser.add_argument('--save_visualization_class', type=bool, default=False,
                        help='Save visualization class instance to VISUALIZATION_OUTPUT_PATH')

def get_visualization_types() -> Tuple[dict, dict]:
    """Get visualization types for the CLI.
    
    Returns:
        dict: Dictionary of visualization types.
        dict: Dictionary of visualization functions for each type.
    """
    types = {}
    current_dir = Path(__file__).parent
    for folder in current_dir.iterdir():
        # Check if the item is a directory and not a hidden directory
        if folder.is_dir() and not folder.name.startswith("_"):
            try:
                # Attempt to import the module
                module = importlib.import_module(
                    __package__ + f".{folder.name}.framework"
                )
                entry_class = getattr(module, f"{folder.name.capitalize()}Visualizations", None)
                if entry_class:
                    types[folder.name] = entry_class
            except ModuleNotFoundError as e:
                # Handle the case where the module cannot be found
                print(f"Module src.visualizations.{folder.name}.framework could not be found: {e}")
                pass
            
    functions = {}
    for type_name, type_class in types.items():
        try:
            module = importlib.import_module(__package__ + f'.{type_name}.functions')
            for name, obj in vars(module).items():
                try:
                    if callable(obj) and obj.__module__ == __package__ + f'.{type_name}.functions':
                        functions[type_name] = functions.get(type_name, {})
                        functions[type_name][name] = obj
                except (TypeError, KeyError):
                    pass
        except ModuleNotFoundError as e:
            # Handle the case where the functions module cannot be found
            print(f"Module src.visualizations.{type_name}.functions could not be found: {e}")
            functions[type_name] = {}
            
    return types, functions
