import importlib
from pathlib import Path

from argparse import ArgumentParser

def scan_loader_args(parser: ArgumentParser):
    parser.add_argument('scan_path', type=str, help='Path to scan signals')
    parser.add_argument('scan_loader', type=str,
                        help='Scan loader to use. Available options: ' + ', '.join(get_scan_loaders().keys()))
    parser.add_argument('--parser_output_path', type=str, default='parsed_data.pkl', help='Path to output parser results')
    parser.add_argument('--save_parsed_results', type=bool, default=False, 
                        help='Save parsed results to PARSER_OUTPUT_PATH')
    parser.add_argument('--scan_loader_kwargs', type=dict, default=None,
                        help='Additional arguments for the scan loader')
    
def get_scan_loaders() -> dict:
    """Get scan loaders for the CLI.
    
    Returns:
        dict: Dictionary of scan loaders.
    """
    current_dir = Path(__file__).parent
    classes = {}
    for folder in current_dir.iterdir():
        # Check if the item is a directory and not a hidden directory
        if folder.is_dir() and not folder.name.startswith("_"):
            try:
                # Attempt to import the module
                module = importlib.import_module(__package__ + f".{folder.name}.main")
                entry_class = getattr(module, "EntryClass", None)
                if entry_class:
                    classes[folder.name] = {}
                    classes[folder.name]['cls'] = entry_class
                    classes[folder.name]['file_exts'] = entry_class.extensions
            except ModuleNotFoundError as e:
                # print(e)
                # Handle the case where the module cannot be found
                pass
    
    return classes