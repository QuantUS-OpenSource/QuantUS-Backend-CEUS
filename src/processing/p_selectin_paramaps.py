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

PATH_MATCHES = {
    'C1M1': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 1/Mouse01D01batch1_02132019/MouseC1M1_d01_02132019/20190213124404.926.nii.gz',
            'start_time': 5,
            'end_time': 30,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 1/Mouse01D02batch1_02152019/MouseC1M1_d02_02152019/20190215152929.369.nii.gz',
            'start_time': -1,
            'end_time': -1,
            'notes': 'Excessive water movement, wash-out only, don\'t recommend for analysis'
        },
    },
    'C2M1': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 1/Mouse01D01batch1_02132019/MouseC2M1_d01_02132019/20190213113020.793.nii.gz',
            'start_time': 5,
            'end_time': 22,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 1/Mouse01D02batch1_02152019/MouseC2M1_d02_02152019/20190215140506.535.nii.gz',
            'start_time': 20,
            'end_time': 75,
        },
    },
    'C2M2': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 1/Mouse01D01batch1_02132019/MouseC2M2_d01_02132019/20190213120637.539.nii.gz',
            'start_time': 0,
            'end_time': 25,
            'notes': 'Excessive water movement, don\'t recommend for analysis'
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 1/Mouse01D02batch1_02152019/MouseC2M2_d02_02152019/20190215144608.915.nii.gz',
            'start_time': 0,
            'end_time': 60,
        },
    },
    'C1M2': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D01_02162019/MouseC1M2_d001_02162019/20190216174022.664.nii.gz',
            'start_time': 0,
            'end_time': 120,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D02_02182019/MouseC1M2_d002_02182019/20190218122221.029.nii.gz',
            'start_time': 0,
            'end_time': 60,
        },
    },
    'C1M3': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D01_02162019/MouseC1M3_d001_02162019/20190216153719.257.nii.gz',
            'start_time': 0,
            'end_time': 60,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D02_02182019/MouseC1M3_d002_02182019/20190218141436.477.nii.gz',
            'start_time': 0,
            'end_time': 50,
        },
    },
    'C2M3': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D01_02162019/MouseC2M3_d001_02162019/20190216160924.918.nii.gz',
            'start_time': 0,
            'end_time': 60,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D02_02182019/MouseC2M3_d002_02182019/20190218124547.315.nii.gz',
            'start_time': 0,
            'end_time': 60,
        },
    },
    'C2M4': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D01_02162019/MouseC2M4_d001_02162019/20190216180639.285.nii.gz',
            'start_time': 0,
            'end_time': 50,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D02_02182019/MouseC2M4_d002_02182019/20190218130746.412.nii.gz',
            'start_time': 0,
            'end_time': 80,
            'notes': 'Transducer moved during scan, don\'t recommend for analysis'
        },
    },
    'C3M1': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D01_02162019/MouseC3M1_d001_02162019/20190216164816.762.nii.gz',
            'start_time': 0,
            'end_time': 50,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D02_02182019/MouseC3M1_d002_02182019/20190218133005.424.nii.gz',
            'start_time': 0,
            'end_time': 50,
            'notes': 'Very poor quality, don\'t recommend for any analysis'
        },
    },
    'C3M3': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D01_02162019/MouseC3M3_d001_02162019/20190216171413.319.nii.gz',
            'start_time': 0,
            'end_time': 50,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D02_02182019/MouseC3M3_d002_02182019/20190218135141.868.nii.gz',
            'start_time': 0,
            'end_time': 55,
        },
    },
    'C3M4': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D01_02162019/MouseC3M4_d001_02162019/20190216183204.449.nii.gz',
            'start_time': 0,
            'end_time': 55,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/Feb 2019 Mice Batch 2/MouseBatch2D02_02182019/MouseC3M4_d002_02182019/20190218143817.014.nii.gz',
            'start_time': 0,
            'end_time': 60,
        },
    },
    'July M1': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 Mice/MouseBatch1D01_07252019/MouseJulyM1_d001_07252019/20190725100957.844.nii.gz',
            'start_time': 0,
            'end_time': 55,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 3 imaging day 2/July2019 mouse 1 imaging day 2/20190727093818.218.nii.gz',
            'start_time': 0,
            'end_time': 45,
        },
    },
    'July M2': {
        'Day 1': {
          'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 4 imaging day 1/July2019 mouse 2 imaging day 1/20190726092440.812.nii.gz',
          'start_time': 0,
          'end_time': 25,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July 2019 batch 4 imaging day 2/July 2019 mouse 2 imaging day 2/20190728085521.429.nii.gz',
            'start_time': 0,
            'end_time': 55,
        },
        'Day 3': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 4 imaging day 3/July2019 mouse 2 imaging day 3/20190730115007.377.nii.gz',
            'start_time': 0,
            'end_time': 50,
        }
    },
    'July M3': {
        'Day 1': {
          'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 4 imaging day 1/July2019 mouse 3 imaging day 1/20190726100254.119.nii.gz',
          'start_time': 0,
          'end_time': 25,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July 2019 batch 4 imaging day 2/July 2019 mouse 3 imaging day 2/20190728092842.768.nii.gz',
            'start_time': 0,
            'end_time': 30
        },
        'Day 3': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 4 imaging day 3/July2019 mouse 3 imaging day 3/20190730095307.661.nii.gz',
            'start_time': 0,
            'end_time': 50,
        }
    },
    'July M4': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July 2019 batch 3 imaging day 1/July2019mouse4imagingday1/20190725123015.493.nii.gz',
            'start_time': -1,
            'end_time': -1,
            'notes': 'Wash-out only, only useful for dTE'
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 3 imaging day 2/July2019 mouse 4 imaging day 2/20190727100541.611.nii.gz',
            'start_time': 0,
            'end_time': 50,
            'notes': 'Severe water movement, poor quality, and mouse movement. Don\'t recommend for any analysis'
        }
    },
    'July M5': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July 2019 batch 3 imaging day 1/July2019mouse5imagingday1/20190725125851.660.nii.gz',
            'start_time': 0,
            'end_time': 50,
            'notes': 'Sudden mouse movements. Perfusion analysis only, dTE isn\'t valid (unless new VOIs are drawn)'
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 3 imaging day 2/July2019 mouse 5 imaging day 2/20190727103247.265.nii.gz',
            'start_time': 0,
            'end_time': 50,
        }
    },
    'July M6': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July 2019 batch 3 imaging day 1/July2019mouse6imagingday1/20190725112516.118.nii.gz',
            'start_time': 0,
            'end_time': 55,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 3 imaging day 2/July2019 mouse 6 imaging day 2/20190727105949.073.nii.gz',
            'start_time': 0,
            'end_time': 50,
            'notes': 'Didn\'t fully capture wash-in. Perfusion analysis suspect, dTE ok'
        }
    },
    'July M7': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 4 imaging day 1/July2019 mouse 7 imaging day 1/20190726103330.200.nii.gz',
            'start_time': 0,
            'end_time': 30,
            'notes': 'Wash-in not fully captured, perfusion analysis suspect, dTE ok'
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July 2019 batch 4 imaging day 2/July 2019 mouse 7 imaging day 2/20190728100733.314.nii.gz',
            'start_time': 0,
            'end_time': 50,
        },
        'Day 3': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 4 imaging day 3/July2019 mouse 7 imaging day 3/20190730101926.649.nii.gz',
            'start_time': 0,
            'end_time': 45,
        }
    },
    'July M8': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July 2019 batch 3 imaging day 1/July2019mouse8imagingday1/20190725115452.271.nii.gz',
            'start_time': 0,
            'end_time': 55,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 3 imaging day 2/July2019 mouse 8 imaging day 2/20190727114054.585.nii.gz',
            'start_time': 0,
            'end_time': 110,
        },
    },
    'July M9': {
        'Day 1': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 4 imaging day 1/July2019 mouse 9 imaging day 1/20190726110721.235.nii.gz',
            'start_time': 0,
            'end_time': 40,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July 2019 batch 4 imaging day 2/July 2019 mouse 9 imaging day 2/20190728104449.936.nii.gz',
            'start_time': 0,
            'end_time': 70,
        },
        'Day 3': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 4 imaging day 3/July2019 mouse 9 imaging day 3/20190730110217.616.nii.gz',
            'start_time': 0,
            'end_time': 50,
        }
    },
    'July M10': {
        'Day 1': {
          'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 4 imaging day 1/July2019 mouse 10 imaging day 1/20190726113502.538.nii.gz',
          'start_time': 0,
          'end_time': 40,
        },
        'Day 2': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July 2019 batch 4 imaging day 2/July 2019 mouse 10 imaging day 2/20190728111152.032.nii.gz',
            'start_time': 0,
            'end_time': 50,
        },
        'Day 3': {
            'file_path': '/Volumes/TOSHIBA EXT/P-Selectin Data/July 2019 US imaging study Mice/July2019 batch 4 imaging day 3/July2019 mouse 10 imaging day 3/20190730112656.658.nii.gz',
            'start_time': 0,
            'end_time': 60,
            'notes': 'Wash-out only, only useful for dTE'
        },
    }
}

# Need scan path and mask path for each analysis run

def analyze_segmented_data(scan_path: str, mask_path: str, pipeline_config: Dict, results_dir: str,
                           mouse_name: str, batch: int, day: int, start_time: int, end_time: int) -> int:
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
    roi_name = "LEFT" if "LEFT" in mask_path else "RIGHT"

    output_dir = Path(results_dir)  / mouse_name / f"batch_{batch}"  / f"day_{day}" / roi_name
    output_dir.mkdir(parents=True, exist_ok=True)

    assert roi_name in mask_path, f"Mask path {mask_path} does not contain expected ROI name {roi_name}"
    pipeline_config['visualization_kwargs']['paramap_folder_path'] = str(output_dir)
    pipeline_config['curve_quant_kwargs']['start_time'] = start_time
    pipeline_config['curve_quant_kwargs']['end_time'] = end_time
    pipeline_config['visualization_kwargs']['start_time'] = start_time
    pipeline_config['visualization_kwargs']['end_time'] = end_time

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

    # Skipped scans for all maps
    skipped_scanname = ['20190218130746.412', '20190218133005.424',
                        '20190727100541.611', '20190215152929.369', '20190213120637.539']

    # Skipped scans not useful for perfusion maps
    # skipped_scanname += ['20190725123015.493', '20190727105949.073',
    #                      '20190726103330.200', '20190730112656.658',]

    # Skipped scans not useful for dTE maps
    skipped_scanname += ['20190725125851.660']

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
            start_time = PATH_MATCHES.get(mouse_name, {}).get(f'Day {day}', {})['start_time']
            end_time = PATH_MATCHES.get(mouse_name, {}).get(f'Day {day}', {})['end_time']
        elif scan_file := feb2_scanname_to_path.get(scan_name):
            day = int(scan_file.parent.name[11:14])
            mouse_name = scan_file.parent.name[5:9]
            batch = 2
            start_time = PATH_MATCHES.get(mouse_name, {}).get(f'Day {day}', {})['start_time']
            end_time = PATH_MATCHES.get(mouse_name, {}).get(f'Day {day}', {})['end_time']
        else:
            raise ValueError(f"Scan file not found for {feb_seg_file.name} in February datasets")
        scan_seg_pairs.append((scan_file, feb_seg_file, mouse_name, batch, day, start_time, end_time))

    ix = 0
    for scan_file, feb_seg_file, mouse_name, batch, day, start_time, end_time in tqdm(scan_seg_pairs, desc="Analyzing February data"):
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
            day=day,
            start_time=start_time,
            end_time=end_time
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
        start_time = PATH_MATCHES.get(mouse_name, {}).get(f'Day {day}', {})['start_time']
        end_time = PATH_MATCHES.get(mouse_name, {}).get(f'Day {day}', {})['end_time']
        scan_seg_pairs.append((scan_file, july_seg_file, mouse_name, batch, day))

    for scan_file, july_seg_file, mouse_name, batch, day in tqdm(scan_seg_pairs, desc="Analyzing July data"):
        if exit_code := analyze_segmented_data(
            scan_path=str(scan_file),
            mask_path=str(july_seg_file),
            pipeline_config=pipeline_config,
            results_dir=results_dir,
            mouse_name=mouse_name,
            batch=batch,
            day=day,
            start_time=start_time,
            end_time=end_time
        ):
            return exit_code

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
