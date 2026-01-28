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
    import sys
    import importlib
    from pathlib import Path

    functions = {}

    # 1. Load from internal-TUL if available
    project_root = Path(__file__).parents[4]
    internal_tul_path = project_root / "Internal-TUL" / "QuantUS-CEUS" / "processing"

    if internal_tul_path.exists():
        if str(internal_tul_path) not in sys.path:
            sys.path.append(str(internal_tul_path))
            
        # Internal modules in CEUS depend on src.full_workflow from engines/ceus/src
        ceus_engine_root = project_root / "engines" / "ceus"
        if ceus_engine_root.exists() and str(ceus_engine_root) not in sys.path:
            sys.path.append(str(ceus_engine_root))

        for item in internal_tul_path.iterdir():
            if item.is_file() and not item.name.startswith("_") and item.suffix == ".py":
                try:
                    module_name = item.stem
                    module = importlib.import_module(module_name)
                    for name, obj in vars(module).items():
                        if callable(obj) and obj.__module__ == module_name:
                            functions[name] = obj
                except Exception as e:
                    print(f"Internal module {item.name} could not be loaded: {e}")

    # 2. Load from public functions
    for name, obj in globals().items():
        if callable(obj) and obj.__module__ == __package__ + '.functions':
            functions[name] = obj
    return functions