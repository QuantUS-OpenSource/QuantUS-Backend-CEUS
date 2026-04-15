from typing import List
from pathlib import Path

import numpy as np

class UltrasoundImage:
    """
    Class for general ultrasound image data (e.g., B-mode, CEUS, NIfTI).
    """
    def __init__(self, scan_path: str):
        self.scan_name = Path(scan_path).stem
        self.scan_path = scan_path
        self.pixel_data: np.ndarray # image data as a numpy array
        self.pixdim: List[float] # mm
        self.frame_rate: float # Hz
        self.intensities_for_analysis: np.ndarray # linearized intensity values
        self.extras_dict: dict = {} # dictionary for any extra information inputted by plugins

        # Visualization defaults (Moved from CeusSeg)
        self.clahe_clip_limit: float = 1.2
        self.gamma: float = 1.5
        self.width_scale_axial: float = 1.0
        self.width_scale_sagittal: float = 1.0
        self.width_scale_coronal: float = 1.0
        self.use_philips_ceus: bool = False
