import numpy as np
from typing import List

class CeusSeg:
    """
    Class for contrast-enhanced ultrasound image data.
    """

    def __init__(self):
        self.seg_name: str
        self.seg_mask: np.ndarray
        self.pixdim: List[float]  # voxel spacing in mm
        # Visualization defaults
        self.clahe_clip_limit: float = 1.2
        self.gamma: float = 1.5
        self.width_scale_axial: float = 1.0
        self.width_scale_sagittal: float = 1.0
        self.width_scale_coronal: float = 1.0
        self.use_philips_ceus: bool = False
