from pathlib import Path

import yaml
import pandas as pd
from tqdm import tqdm

from ..full_workflow import main_dict


def analyze_preloaded_curves(curves_dir: str, pipeline_config_path: str, results_dir: str) -> int:
    """
    Analyze preloaded curves from a directory.

    Args:
        curves_dir (str): Directory containing preloaded curves.
        pipeline_config_path (str): Path to the pipeline configuration YAML file.
        results_dir (str): Directory to save the results.

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    with open(pipeline_config_path, 'r') as f:
        pipeline_config_dict = yaml.safe_load(f)

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    for curves_file in Path(curves_dir).glob('*.csv'):
        if not curves_file.is_file():
            print(f"Skipping non-file: {curves_file}")
            continue

        pipeline_config_dict['curves_path'] = str(curves_file)
        pipeline_config_dict['curve_quant_output_path'] = str(Path(results_dir) / f"{curves_file.stem}_quantified_analysis.csv")

        # Run the main workflow
        if exit_code := main_dict(pipeline_config_dict):
            return exit_code

    combine_numerical_results(Path(results_dir))
    return 0

def combine_numerical_results(results_dir: Path) -> None:
    """Combines numerical results from all processed scans into a single CSV file.

    Args:
        results_dir (Path): Path to the directory containing the results of the analysis.
    """
    all_results = []
    for result_file in results_dir.glob('**/*.csv'):
        if result_file.name == 'combined_results.csv':
            raise ValueError("Cannot combine results with the same name as the output file. Either delete or rename 'combined_results.csv' before running this function.")
        df = pd.read_csv(result_file)

        if 'molecular' in result_file.name:
            contrast_type = 'molecular'
        elif 'bolus' in result_file.name:
            contrast_type = 'bolus'
        else:
            print(f"Skipping file {result_file} as it does not contain 'molecular' or 'bolus' in its name.")
            continue
        if 'molecular' in result_file.name and 'bolus' in result_file.name:
            print(f"Skipping file {result_file} as it contains both 'molecular' and 'bolus' in its name.")
            continue

        # Insert columns at the start of the DataFrame
        mouse_name = result_file.name[:4]
        try:
            insert_cols = {
                'mouse': mouse_name,
                'contrast-type': contrast_type,
            }
        except KeyError as e:
            print(f"Missing metadata for mouse {mouse_name}: {e}")
            continue

        for col, val in reversed(list(insert_cols.items())):
            df.insert(0, col, val)

        all_results.append(df)

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(results_dir / 'combined_results.csv', index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate curves from bolus and molecular data.")
    parser.add_argument('curves_dir', type=str, help='Directory containing preloaded curves.')
    parser.add_argument('pipeline_config_path', type=str, help='Path to the pipeline configuration YAML file.')
    parser.add_argument('results_dir', type=str, help='Directory to save the results.')

    args = parser.parse_args()

    exit_code = analyze_preloaded_curves(args.curves_dir, args.pipeline_config_path, args.results_dir)
    exit(exit_code)
