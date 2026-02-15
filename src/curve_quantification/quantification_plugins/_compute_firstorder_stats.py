import numpy as np
from scipy.stats import skew, kurtosis, entropy

def _compute_firstorder_stats(curve: np.ndarray, data_dict: dict, name_prefix: str, name_suffix: str = '') -> None:
    """
    Compute first-order statistics for a given curve and store them in the data dictionary.
    
    Args:
        curve (np.ndarray): The curve data to analyze.
        data_dict (dict): Dictionary to store the computed statistics.
        name_prefix (str): Prefix for the keys in the data dictionary.
        name_suffix (str): Suffix for the keys in the data dictionary.
    """
    if len(curve) == 0:
        return

    curve = np.array(curve)
    data_dict[f'{name_prefix}Mean{name_suffix}'] = np.mean(curve)
    data_dict[f'{name_prefix}Std{name_suffix}'] = np.std(curve)
    data_dict[f'{name_prefix}Max{name_suffix}'] = np.max(curve)
    data_dict[f'{name_prefix}Min{name_suffix}'] = np.min(curve)
    data_dict[f'{name_prefix}Median{name_suffix}'] = np.median(curve)
    data_dict[f'{name_prefix}Variance{name_suffix}'] = np.var(curve)
    if np.var(curve) < 1e-10:
        data_dict[f'{name_prefix}Skewness{name_suffix}'] = 0.0
        data_dict[f'{name_prefix}Kurtosis{name_suffix}'] = 3.0  # Normal kurtosis
    else:
        data_dict[f'{name_prefix}Skewness{name_suffix}'] = skew(curve)
        data_dict[f'{name_prefix}Kurtosis{name_suffix}'] = kurtosis(curve)
    data_dict[f'{name_prefix}Range{name_suffix}'] = np.max(curve) - np.min(curve)
    data_dict[f'{name_prefix}InterquartileRange{name_suffix}'] = np.percentile(curve, 75) - np.percentile(curve, 25)
    data_dict[f'{name_prefix}Entropy{name_suffix}'] = entropy(curve)
    data_dict[f'{name_prefix}Energy{name_suffix}'] = np.sum(curve ** 2)