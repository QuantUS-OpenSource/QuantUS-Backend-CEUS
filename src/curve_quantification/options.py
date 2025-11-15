import importlib
import inspect
from typing import Dict

def get_quantification_funcs() -> Dict[str, callable]:
    """Get quantification functions for the CLI.
    Returns:
        dict: Dictionary of quantification functions.
    """
    functions = {}
    module = importlib.import_module(__package__ + '.functions')
    module_file = module.__file__
    defined_funcs = set()
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("_") and inspect.getsourcefile(obj) == module_file:
            defined_funcs.add(name)
    functions = {name: obj for name, obj in inspect.getmembers(module, inspect.isfunction) if name in defined_funcs}

    return functions
