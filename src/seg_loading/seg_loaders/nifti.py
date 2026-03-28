import json
from pathlib import Path

import numpy as np
import nibabel as nib

from ..decorators import extensions
from ...data_objs.seg import CeusSeg
from ...data_objs.image import UltrasoundImage
from ...seg_preprocessing.motion_compensation_3d import (
    BoundingBox3D, MotionCompensationResult
)

@extensions(".nii", ".nii.gz")
def nifti(image_data: UltrasoundImage, seg_path: str, **kwargs) -> CeusSeg:
    """
    Load ROI/VOI data from a NIfTI file. segmentation mask is used as-is.
    If the file contains a motion_compensation NIfTI extension, it is
    reconstructed into seg_data.motion_compensation.
    """
    assert seg_path.endswith('.nii.gz') or seg_path.endswith('.nii'), "seg_path must be a NIfTI file"

    out = CeusSeg()
    seg = nib.load(seg_path)
    out.seg_mask = np.asarray(seg.dataobj, dtype=np.uint8)

    if out.seg_mask.ndim == 3:
        out.pixdim = seg.header.get_zooms()[:2]
    elif out.seg_mask.ndim == 4:
        out.pixdim = seg.header.get_zooms()[:3]

    if seg_path.endswith('.nii.gz'):
        out.seg_name = Path(seg_path).name[:-7]
    else:
        out.seg_name = Path(seg_path).name[:-4]

    for ext in seg.header.extensions:
        try:
            data = json.loads(ext.get_content())
            if 'motion_compensation' not in data:
                continue
            mc = data['motion_compensation']
            out.motion_compensation = MotionCompensationResult(
                translation_vectors=np.array(mc['translation_vectors'], dtype=np.float32),
                reference_frame=int(mc['reference_frame']),
                correlations=np.array(mc['correlations'], dtype=np.float32),
                reference_bbox=BoundingBox3D(*mc['reference_bbox']),
                tracked_bboxes=[BoundingBox3D(*b) for b in mc['tracked_bboxes']],
            )
            break
        except Exception:
            pass

    return out