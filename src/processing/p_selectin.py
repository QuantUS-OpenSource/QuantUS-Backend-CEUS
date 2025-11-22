from pathlib import Path
from typing import Tuple, Dict, List

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..full_workflow import main_dict

# Metadata

FEB_MOUSE_NAME_TO_NUM = {
    'C1M1': 1, 'C2M1': 5, 'C2M2': 6, 'C1M2': 2, 'C1M3': 3,
    'C2M3': 7, 'C2M4': 8, 'C3M1': 9, 'C3M3': 11, 'C3M4': 12
}

RADIATION_METADATA = {
    (1, 0, 1, 'L'): ('No', 0), (1, 0, 1, 'R'): ('No', 0),
    (1, 1, 1, 'L'): ('Yes', 0), (1, 1, 1, 'R'): ('Yes', 6),
    (1, 0, 5, 'L'): ('No', 0), (1, 0, 5, 'R'): ('No', 0),
    (1, 1, 5, 'L'): ('Yes', 0), (1, 1, 5, 'R'): ('Yes', 6),
    (1, 0, 6, 'L'): ('No', 0), (1, 0, 6, 'R'): ('No', 0),
    (1, 1, 6, 'L'): ('Yes', 6), (1, 1, 6, 'R'): ('Yes', 0),
    (2, 0, 2, 'L'): ('No', 0), (2, 0, 2, 'R'): ('No', 0),
    (2, 1, 2, 'L'): ('Yes', 6), (2, 1, 2, 'R'): ('Yes', 0),
    (2, 0, 3, 'L'): ('No', 0), (2, 0, 3, 'R'): ('No', 0),
    (2, 1, 3, 'L'): ('No', 0), (2, 1, 3, 'R'): ('No', 0),
    (2, 0, 7, 'L'): ('No', 0), (2, 0, 7, 'R'): ('No', 0),
    (2, 1, 7, 'L'): ('Yes', 0), (2, 1, 7, 'R'): ('Yes', 6),
    (2, 0, 8, 'L'): ('No', 0), (2, 0, 8, 'R'): ('No', 0),
    (2, 1, 8, 'L'): ('Yes', 0), (2, 1, 8, 'R'): ('Yes', 0),
    (2, 0, 9, 'L'): ('No', 0), (2, 0, 9, 'R'): ('No', 0),
    (2, 1, 9, 'L'): ('Yes', 6), (2, 1, 9, 'R'): ('Yes', 0),
    (2, 0, 11, 'L'): ('No', 0), (2, 0, 11, 'R'): ('No', 0),
    (2, 1, 11, 'L'): ('Yes', 0), (2, 1, 11, 'R'): ('Yes', 6),
    (2, 1, 12, 'L'): ('Yes', 0), (2, 1, 12, 'R'): ('Yes', 6),
    (2, 2, 12, 'L'): ('Yes', 0), (2, 2, 12, 'R'): ('Yes', 6),
    (3, 0, 1, 'R'): ('No', 0), (3, 0, 1, 'L'): ('No', 0),
    (3, 1, 1, 'R'): ('Yes', 0), (3, 1, 1, 'L'): ('Yes', 6),
    (3, 0, 4, 'R'): ('No', 0), (3, 0, 4, 'L'): ('No', 0),
    (3, 1, 4, 'R'): ('Yes', 0), (3, 1, 4, 'L'): ('Yes', 6),
    (3, 0, 5, 'R'): ('No', 0), (3, 0, 5, 'L'): ('No', 0),
    (3, 1, 5, 'R'): ('Yes', 0), (3, 1, 5, 'L'): ('Yes', 6),
    (3, 0, 6, 'R'): ('No', 0), (3, 0, 6, 'L'): ('No', 0),
    (3, 1, 6, 'R'): ('Yes', 0), (3, 1, 6, 'L'): ('Yes', 6),
    (3, 0, 8, 'R'): ('No', 0), (3, 0, 8, 'L'): ('No', 0),
    (3, 1, 8, 'R'): ('Yes', 0), (3, 1, 8, 'L'): ('Yes', 6),
    (4, 0, 2, 'R'): ('No', 0), (4, 0, 2, 'L'): ('No', 0),
    (4, 1, 2, 'R'): ('No', 0), (4, 1, 2, 'L'): ('No', 0),
    (4, 2, 2, 'R'): ('Yes', 0), (4, 2, 2, 'L'): ('Yes', 6),
    (4, 0, 3, 'R'): ('No', 0), (4, 0, 3, 'L'): ('No', 0),
    (4, 1, 3, 'R'): ('No', 0), (4, 1, 3, 'L'): ('No', 0),
    (4, 2, 3, 'R'): ('Yes', 0), (4, 2, 3, 'L'): ('Yes', 6),
    (4, 0, 7, 'R'): ('No', 0), (4, 0, 7, 'L'): ('No', 0),
    (4, 1, 7, 'R'): ('No', 0), (4, 1, 7, 'L'): ('No', 0),
    (4, 2, 7, 'R'): ('Yes', 0), (4, 2, 7, 'L'): ('Yes', 6),
    (4, 0, 9, 'R'): ('No', 0), (4, 0, 9, 'L'): ('No', 0),
    (4, 1, 9, 'R'): ('No', 0), (4, 1, 9, 'L'): ('No', 0),
    (4, 2, 9, 'R'): ('Yes', 0), (4, 2, 9, 'L'): ('Yes', 6),
    (4, 0, 10, 'R'): ('No', 0), (4, 0, 10, 'L'): ('No', 0),
    (4, 1, 10, 'R'): ('No', 0), (4, 1, 10, 'L'): ('No', 0),
    (4, 2, 10, 'R'): ('Yes', 0), (4, 2, 10, 'L'): ('Yes', 6)
}

def combine_numerical_results(results_dir: Path) -> None:
    """Combines numerical results from all processed scans into a single CSV file.

    Args:
        results_dir (Path): Path to the directory containing the results of the analysis.
    """
    all_results = []
    for result_file in tqdm(results_dir.glob('**/*_quant_*.csv'), desc="Combining results", total=len(list(results_dir.glob('**/*_quant_*.csv')))):
        if result_file.name == 'combined_quant_results.csv':
            raise ValueError("Cannot combine results with the same name as the output file. Either delete or rename 'combined_results.csv' before running this function.")
        df = pd.read_csv(result_file)

        day = int(result_file.parent.name[len("day_"):])
        batch = int(result_file.parents[1].name.replace(' ', '')[-1])
        mouse = result_file.parents[2].name
        seg_name = "LEFT" if "LEFT" in result_file.name else "RIGHT"
        assert seg_name in result_file.name, f"Segmentation name {seg_name} not found in file name {result_file.name}"
        seg_name = "L" if seg_name == "LEFT" else "R"

        lookup_day = day - 1
        if not mouse.startswith('July'):
            mouse_num = FEB_MOUSE_NAME_TO_NUM[mouse]
        else:
            mouse_num = int(mouse[len('July M'):])
        radiated, dose = RADIATION_METADATA.get((batch, lookup_day, mouse_num, seg_name), (np.nan, np.nan))
        if not type(radiated) is str:
            print(f"Radiation metadata is missing (np.nan) for batch={batch}, day={lookup_day}, mouse={mouse_num}, side={seg_name}")
        try:
            insert_cols = {
            'mouse': mouse,
            'batch': batch,
            'day': day,
            'side': seg_name,
            'radiated': radiated,
            'radiation dose': dose,
            }
        except KeyError as e:
            print(f"Missing metadata for mouse {mouse}: {e}")
            continue

        for col, val in reversed(list(insert_cols.items())):
            df.insert(0, col, val)

        all_results.append(df)

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(results_dir / 'combined_quant_results.csv', index=False)


# Need scan path and mask path for each analysis run

def analyze_segmented_data(scan_path: str, mask_path: str, pipeline_config: Dict, results_dir: str,
                           mouse_name: str, batch: int, day: int) -> int:
    """
    Analyze segmented data using the provided scan and mask paths.

    Args:
        scan_path (str): Path to the scan file.
        mask_path (str): Path to the mask file.
        pipeline_config (Dict): Configuration for the analysis pipeline.
        results_dir (str): Directory to save the results.
        mouse_name (str): Name of the mouse.
        batch (int): Batch number.
        day (int): Day number.

    Returns:
        int: Status code (0 for success, non-zero for failure).
    """
    pipeline_config['scan_path'] = scan_path
    pipeline_config['seg_path'] = mask_path

    output_dir = Path(results_dir)  / mouse_name / f"batch_{batch}"  / f"day_{day}"
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_name = "LEFT" if "LEFT" in mask_path else "RIGHT"
    assert roi_name in mask_path, f"Mask path {mask_path} does not contain expected ROI name {roi_name}"
    pipeline_config['analysis_kwargs']['curves_output_path'] = str(output_dir / f"curves_output_{roi_name}.csv")
    pipeline_config['curve_quant_output_path'] = str(output_dir / f"curve_quant_output_{roi_name}.csv")

    return main_dict(pipeline_config)

def analyze_pselectin_data(feb1_root_path: str, feb2_root_path: str, july_root_path: str, feb_voi_dir: str,
                           july_voi_dir: str, pipeline_config_path: str, results_dir: str) -> int:
    """
    Analyze P-selectin data from February and July datasets.

    Args:
        feb1_root_path (str): Path to the first February dataset.
        feb2_root_path (str): Path to the second February dataset.
        july_root_path (str): Path to the July dataset.
        feb_voi_dir (str): Directory containing February VOI files.
        july_voi_dir (str): Directory containing July VOI files.
        pipeline_config_path (str): Path to the pipeline configuration YAML file.
        results_dir (str): Directory to save the results.

    Returns:
        int: Status code (0 for success, non-zero for failure).
    """
    # Load pipeline configuration
    with open(pipeline_config_path, 'r') as f:
        pipeline_config = yaml.safe_load(f)

    feb1_scanname_to_path = {
        feb_scan_file.name[:-7]: feb_scan_file for feb_scan_file in Path(feb1_root_path).glob('**/*[0-9].nii.gz')
    }
    feb2_scanname_to_path = {
        feb_scan_file.name[:-7]: feb_scan_file for feb_scan_file in Path(feb2_root_path).glob('**/*[0-9].nii.gz')
    }
    july_scanname_to_path = {
        july_scan_file.name[:-7]: july_scan_file for july_scan_file in Path(july_root_path).glob('**/*[0-9][0-9].nii.gz')
    }
    skipped_scanname = ['20190218130746.412', '20190213120637.539',
                        '20190215152929.369'] # spinning issue
    skipped_scanname += ['20190727100541.611'] # washout only. Flash appears to occur after all contrast is gone

    scan_seg_pairs = []
    for feb_seg_file in Path(feb_voi_dir).glob('201902*.nii.gz'):
        scan_name = feb_seg_file.name[:18]
        if scan_name in skipped_scanname:
            print(f"Skipping {scan_name} as it is not a valid scan.")
            continue
        if scan_file := feb1_scanname_to_path.get(scan_name):
            day = int(scan_file.parent.name[11:13])
            mouse_name = scan_file.parent.name[5:9]
            batch = 1
        elif scan_file := feb2_scanname_to_path.get(scan_name):
            day = int(scan_file.parent.name[11:14])
            mouse_name = scan_file.parent.name[5:9]
            batch = 2
        else:
            raise ValueError(f"Scan file not found for {feb_seg_file.name} in February datasets")
        scan_seg_pairs.append((scan_file, feb_seg_file, mouse_name, batch, day))

    ix = 0
    for scan_file, feb_seg_file, mouse_name, batch, day in tqdm(scan_seg_pairs, desc="Analyzing February data"):
        if ix < 4:
            ix += 1
            continue
        print(f"Analyzing {scan_file.name} with mask {feb_seg_file.name} for mouse {mouse_name}, batch {batch}, day {day}")
        if exit_code := analyze_segmented_data(
            scan_path=str(scan_file),
            mask_path=str(feb_seg_file),
            pipeline_config=pipeline_config,
            results_dir=results_dir,
            mouse_name=mouse_name,
            batch=batch,
            day=day
        ):
            return exit_code

    scan_seg_pairs = []
    for july_seg_file in Path(july_voi_dir).glob('201907*.nii.gz'):
        scan_name = july_seg_file.name[:18]
        if scan_name in skipped_scanname:
            print(f"Skipping {scan_name} as it is not a valid scan.")
            continue
        if scan_file := july_scanname_to_path.get(scan_name):
            day = int(scan_file.parent.name.replace(' ', '')[-1])
            try:
                mouse = int(scan_file.parent.name.replace(' ', '')[13:15])
            except ValueError:
                mouse = int(scan_file.parent.name.replace(' ', '')[13])
            batch = int(scan_file.parent.parent.name.replace(' ', '')[13])
            mouse_name = f"July M{mouse}"
        else:
            raise ValueError(f"Scan file not found for {july_seg_file.name} in July dataset")
        scan_seg_pairs.append((scan_file, july_seg_file, mouse_name, batch, day))

    for scan_file, july_seg_file, mouse_name, batch, day in tqdm(scan_seg_pairs, desc="Analyzing July data"):
        if exit_code := analyze_segmented_data(
            scan_path=str(scan_file),
            mask_path=str(july_seg_file),
            pipeline_config=pipeline_config,
            results_dir=results_dir,
            mouse_name=mouse_name,
            batch=batch,
            day=day
        ):
            return exit_code
    combine_numerical_results(Path(results_dir))
    return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze P-selectin data from February and July datasets.")
    parser.add_argument('feb1_root_path', type=str, help="Root path for the first February dataset.")
    parser.add_argument('feb2_root_path', type=str, help="Root path for the second February dataset.")
    parser.add_argument('july_root_path', type=str, help="Root path for the July dataset.")
    parser.add_argument('feb_voi_dir', type=str, help="Directory containing February VOI files.")
    parser.add_argument('july_voi_dir', type=str, help="Directory containing July VOI files.")
    parser.add_argument('pipeline_config_path', type=str,
                        help="Path to the pipeline configuration YAML file.")
    parser.add_argument('results_dir', type=str, help="Directory to save the results.")

    args = parser.parse_args()

    exit_code = analyze_pselectin_data(
        feb1_root_path=args.feb1_root_path,
        feb2_root_path=args.feb2_root_path,
        july_root_path=args.july_root_path,
        feb_voi_dir=args.feb_voi_dir,
        july_voi_dir=args.july_voi_dir,
        pipeline_config_path=args.pipeline_config_path,
        results_dir=args.results_dir
    )
    exit(exit_code)
