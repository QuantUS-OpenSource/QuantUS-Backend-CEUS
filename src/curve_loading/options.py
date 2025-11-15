from pathlib import Path

from argparse import ArgumentParser

from .functions import *

def get_curves_loaders() -> dict:
    """Get curves loaders for the CLI.

    Returns:
        dict: Dictionary of curve loaders.
    """
    functions = {name: obj for name, obj in globals().items() if callable(obj) and obj.__module__ == __package__ + '.functions'}
    return functions