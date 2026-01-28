import importlib
import sys
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
    classes = {}
    
    # 1. Load from internal-TUL if available
    project_root = Path(__file__).parents[4]
    internal_tul_path = project_root / "Internal-TUL" / "QuantUS-CEUS" / "processing"
    
    if internal_tul_path.exists():
        # Note: CEUS internal structure seems to be files, not folders like QUS
        if str(internal_tul_path) not in sys.path:
            sys.path.append(str(internal_tul_path))
            
        # Internal modules in CEUS depend on src.full_workflow from engines/ceus/src
        ceus_src_path = project_root / "engines" / "ceus" / "src"
        if ceus_src_path.exists() and str(ceus_src_path) not in sys.path:
            # We add it so 'import src.full_workflow' works if 'src' is a package
            # OR we add parent of src if it needs 'src.xxx'
            ceus_engine_root = project_root / "engines" / "ceus"
            if str(ceus_engine_root) not in sys.path:
                sys.path.append(str(ceus_engine_root))
            
        for item in internal_tul_path.iterdir():
            if item.is_file() and not item.name.startswith("_") and item.suffix == ".py":
                try:
                    module_name = item.stem
                    module = importlib.import_module(module_name)
                    entry_class = getattr(module, "EntryClass", None)
                    if entry_class:
                        classes[module_name] = {}
                        classes[module_name]['cls'] = entry_class
                        classes[module_name]['file_exts'] = getattr(entry_class, 'extensions', [])
                except Exception as e:
                    print(f"Internal module {item.name} could not be loaded: {e}")

    # 2. Load from current directory (public loaders)
    current_dir = Path(__file__).parent
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