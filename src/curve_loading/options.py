import importlib
from pathlib import Path

def get_curves_loaders() -> dict:
    """Get curves loaders for the CLI.

    Returns:
        dict: Dictionary of curve loaders.
    """
    functions = {}
    for file in (Path(__file__).parent / "curve_loaders").iterdir():
        module = importlib.import_module(f'.curve_loaders.{file.stem}', package=__package__)
        for name, obj in vars(module).items():
            if callable(obj) and obj.__module__ == module.__name__:
                functions[name] = obj
    return functions