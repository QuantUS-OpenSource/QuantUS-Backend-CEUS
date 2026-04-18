from pathlib import Path

import numpy as np
import nibabel as nib

from ..decorators import extensions
from ...data_objs.seg import CeusSeg
from ...data_objs.image import UltrasoundImage

@extensions(".nii", ".nii.gz")
def nifti(image_data: UltrasoundImage, seg_path: str, **kwargs) -> CeusSeg:
    """
    Load ROI/VOI data from a NIfTI file. segmentation mask is used as-is.
    """
    assert seg_path.endswith('.nii.gz') or seg_path.endswith('.nii'), "seg_path must be a NIfTI file"
    
    out = CeusSeg()
    seg = nib.load(seg_path)
    out.seg_mask = np.asarray(seg.dataobj, dtype=np.uint8)

    if out.seg_mask.ndim == 3: # 2D + time
        out.pixdim = seg.header.get_zooms()[:2]
    elif out.seg_mask.ndim == 4: # 3D + time
        out.pixdim = seg.header.get_zooms()[:3]
    
    if seg_path.endswith('.nii.gz'):
        out.seg_name = Path(seg_path).name[:-7]  # Remove '.nii.gz'
    else:
        out.seg_name = Path(seg_path).name[:-4]  # Remove '.nii'

    return out