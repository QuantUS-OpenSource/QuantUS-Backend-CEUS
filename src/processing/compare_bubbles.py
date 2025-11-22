from pathlib import Path
from typing import Tuple, Dict, List

import yaml
from tqdm import tqdm

from ..full_workflow import main_dict


# Need scan path and mask path for each analysis run

def organize_bolus_data(bolus_data_path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Organize bolus data from a given path into a dictionary.

    Args:
        bolus_data_path (str): The path to the bolus data file.

    Returns:
        Dict[str, str]: A dictionary with keys as 'bolus_data' and 'bolus_path'.
    """
    bolus_data_path = Path(bolus_data_path)
    scan_paths = {}; mask_paths = {}
    for mouse_folder in bolus_data_path.iterdir():
        if mouse_folder.is_dir() and mouse_folder.name.startswith('m'):
            scan_paths[mouse_folder.name] = []
            mask_paths[mouse_folder.name] = []
            for mask_file in mouse_folder.glob('*Full4DMasked.nii'):
                scan_file = mask_file.parent / mask_file.name.replace('Full4DMasked.nii', 'Full4D.nii')
                scan_paths[mouse_folder.name].append(str(scan_file))
                mask_paths[mouse_folder.name].append(str(mask_file))

    return scan_paths, mask_paths

def organize_molecular_data(molecular_data_path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Organize molecular data from a given path into a dictionary.

    Args:
        molecular_data_path (str): The path to the molecular data file.

    Returns:
        Dict[str, str]: A dictionary with keys as 'molecular_data' and 'molecular_path'.
    """
    molecular_data_path = Path(molecular_data_path)
    scan_paths = {}; mask_paths = {}
    for mouse_folder in molecular_data_path.iterdir():
        if mouse_folder.is_dir() and mouse_folder.name.startswith('m'):
            scan_paths[mouse_folder.name] = []
            mask_paths[mouse_folder.name] = []
            for scan_file in mouse_folder.glob('*.nii.gz'):
                mask_file = scan_file.parent / 'nifti_segmentation_QUANTUS' / f"{scan_file.name[:-7].replace(' ', '_')}_segmentation.nii.gz"
                if mask_file.exists():
                    scan_paths[mouse_folder.name].append(str(scan_file))
                    mask_paths[mouse_folder.name].append(str(mask_file))
                else:
                    print(f"Warning: Mask file {mask_file} does not exist for scan {scan_file}.")

    return scan_paths, mask_paths

def get_overlapping_keys(
    bolus_scan_paths: Dict[str, List[str]],
    molecular_scan_paths: Dict[str, List[str]]
) -> list[str]:
    """
    Get overlapping keys between bolus and molecular scan paths.

    Args:
        bolus_scan_paths (Dict[str, List[str]]): Bolus scan paths.
        molecular_scan_paths (Dict[str, List[str]]): Molecular scan paths.

    Returns:
        set: A set of overlapping keys.
    """
    return list(set(bolus_scan_paths.keys()) & set(molecular_scan_paths.keys()))

def generate_curves(bolus_data_path: str, molecular_data_path: str,
                    pipeline_config_path: str, results_dir: str) -> int:
    """
    Generate curves from bolus and molecular data.

    Args:
        bolus_data_path (str): Path to the bolus data.
        molecular_data_path (str): Path to the molecular data.

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    with open(pipeline_config_path, 'r') as f:
        pipeline_config_dict = yaml.safe_load(f)

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    bolus_scan_paths, bolus_mask_paths = organize_bolus_data(bolus_data_path)
    molecular_scan_paths, molecular_mask_paths = organize_molecular_data(molecular_data_path)

    overlapping_mice = get_overlapping_keys(bolus_scan_paths, molecular_scan_paths)

    for ix, mouse in tqdm(enumerate(overlapping_mice), desc="Processing mice", total=len(overlapping_mice)):
        bolus_scans = bolus_scan_paths[mouse]
        bolus_masks = bolus_mask_paths[mouse]
        if ix > 0:
            for scan, mask in zip(bolus_scans, bolus_masks):
                if not Path(scan).exists() or not Path(mask).exists():
                    print(f"Missing scan or mask for mouse {mouse}: {scan}, {mask}")
                    return 1

                pipeline_config_dict['scan_path'] = scan
                pipeline_config_dict['seg_path'] = mask
                pipeline_config_dict['seg_loader'] = 'load_bolus_mask'

                output_path = Path(results_dir) / f"{mouse}_{Path(scan).stem}_{Path(mask).stem}_bolus_curves.csv"
                pipeline_config_dict['analysis_kwargs']['curves_output_path'] = str(output_path)

                if exit_code := main_dict(pipeline_config_dict):
                    return exit_code

        molecular_scans = molecular_scan_paths[mouse]
        molecular_masks = molecular_mask_paths[mouse]
        for scan, mask in zip(molecular_scans, molecular_masks):
            if not Path(scan).exists() or not Path(mask).exists():
                print(f"Missing scan or mask for mouse {mouse}: {scan}, {mask}")
                return 1

            pipeline_config_dict['scan_path'] = scan
            pipeline_config_dict['seg_path'] = mask
            pipeline_config_dict['seg_loader'] = 'nifti'

            output_path = Path(results_dir) / f"{mouse}_{Path(scan).stem}_{Path(mask).stem}_molecular_curves.csv"
            pipeline_config_dict['analysis_kwargs']['curves_output_path'] = str(output_path)

            if exit_code := main_dict(pipeline_config_dict):
                return exit_code

    return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate curves from bolus and molecular data.")
    parser.add_argument('bolus_data_path', type=str, help='Path to the bolus data directory.')
    parser.add_argument('molecular_data_path', type=str, help='Path to the molecular data directory.')
    parser.add_argument('pipeline_config_path', type=str, help='Path to the pipeline configuration YAML file.')
    parser.add_argument('results_dir', type=str, help='Directory to save the results.')

    args = parser.parse_args()

    exit_code = generate_curves(args.bolus_data_path, args.molecular_data_path, args.pipeline_config_path, args.results_dir)
    exit(exit_code)
