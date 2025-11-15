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
