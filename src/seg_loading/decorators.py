from typing import List

def extensions(*ext_names: List[str]):
    """
    A decorator to specify the supported file extensions for inputs to a function.

    Args:
        ext_names (list): List of supported file extension names.

    Returns:
        function: The decorated function with the specified extensions attached as metadata.
    """
    def decorator(func):
        func.supported_extensions = ext_names
        return func

    return decorator
