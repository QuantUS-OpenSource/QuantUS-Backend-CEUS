import importlib
from pathlib import Path
import inspect
from typing import Dict

def get_quantification_funcs() -> Dict[str, callable]:
    """Get quantification functions for the CLI.
    Returns:
        dict: Dictionary of quantification functions.
    """
    functions = {}
    for file in (Path(__file__).parent / "quantification_plugins").iterdir():
        if not file.stem.startswith("_"):
            module = importlib.import_module(f'.quantification_plugins.{file.stem}', package=__package__)
            for name, obj in vars(module).items():
                if inspect.isfunction(obj) and not name.startswith("_") and obj.__module__ == module.__name__:
                    functions[name] = obj
    return functions