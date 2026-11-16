"""
Save a motion-compensated VOI to NIfTI from a script or notebook.

Writes the same format the GUI's "Save VOI" button produces: the reference mask
as the image, with the tracking result attached as a JSON NIfTI extension. The
NIfTI segmentation loader reads that extension back, so reopening the file in
QuantUS restores the full motion compensation rather than a static mask.
"""

import json
from pathlib import Path

import numpy as np
import nibabel as nib


def _bbox_to_list(b):
    return [float(b.x_min), float(b.x_max), float(b.y_min),
            float(b.y_max), float(b.z_min), float(b.z_max)]


def save_mc_voi(seg_data, image_data, out_path):
    """Write seg_data's VOI and motion compensation to out_path (.nii or .nii.gz).

    Args:
        seg_data: CeusSeg with .seg_mask and (optionally) .motion_compensation
        image_data: UltrasoundImage the VOI was drawn on; supplies pixdim and name
        out_path: destination path; '.nii.gz' is appended if no NIfTI suffix given

    Returns:
        The path written, as a str.
    """
    out_path = str(out_path)
    if not (out_path.endswith('.nii.gz') or out_path.endswith('.nii')):
        out_path += '.nii.gz'

    mask = getattr(seg_data, 'seg_mask', None)
    if mask is None:
        raise ValueError("seg_data has no seg_mask to save")

    affine = np.eye(4)
    for i, res in enumerate(image_data.pixdim[:3]):
        affine[i, i] = res

    img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    img.header["descrip"] = image_data.scan_name

    mc = getattr(seg_data, 'motion_compensation', None)
    if mc is not None:
        mc_dict = {
            'translation_vectors': mc.translation_vectors.tolist(),
            'reference_frame': int(mc.reference_frame),
            'correlations': mc.correlations.tolist(),
            'reference_bbox': _bbox_to_list(mc.reference_bbox),
            'tracked_bboxes': [_bbox_to_list(b) for b in mc.tracked_bboxes],
        }
        img.header.extensions.append(nib.nifti1.Nifti1Extension(
            'comment', json.dumps({'motion_compensation': mc_dict}).encode()))

    nib.save(img, out_path)
    return out_path


def save_mc_voi_frames(seg_data, image_data, out_path, frames, valid_volume=None):
    """Write the motion-compensated VOI for specific frames as ordinary 3D masks.

    Unlike save_mc_voi, which stores the reference mask plus the tracking result
    for QuantUS to re-apply, this writes what the VOI actually covers on each
    frame -- the form a plain NIfTI viewer can show.

    Args:
        seg_data: CeusSeg with .seg_mask and .motion_compensation
        image_data: UltrasoundImage supplying pixdim and name
        out_path: base path; the frame index is appended (foo.nii.gz ->
            foo_f0080.nii.gz)
        frames: iterable of frame indices to write
        valid_volume: optional 4D array (same spatial shape, indexed [..., frame])
            whose non-zero voxels mark where the frame has image data. Supply the
            B-mode volume to clip the VOI to the imaged sector; without it only
            VOI leaving the array is dropped.

    Returns:
        List of written paths.
    """
    mc = getattr(seg_data, 'motion_compensation', None)
    if mc is None:
        raise ValueError("seg_data has no motion_compensation result")

    out_path = Path(str(out_path))
    name = out_path.name
    for ext in ('.nii.gz', '.nii'):
        if name.endswith(ext):
            stem, suffix = name[: -len(ext)], ext
            break
    else:
        stem, suffix = name, '.nii.gz'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    affine = np.eye(4)
    for i, res in enumerate(image_data.pixdim[:3]):
        affine[i, i] = res

    mask = seg_data.seg_mask.astype(np.uint8)
    written = []
    for f in frames:
        valid = None if valid_volume is None else valid_volume[..., f] > 0
        shifted = mc.apply_to_mask(mask, f, 0, valid).astype(np.uint8)
        img = nib.Nifti1Image(shifted, affine)
        img.header["descrip"] = image_data.scan_name
        img.header.set_data_dtype(np.uint8)
        path = out_path.parent / f"{stem}_f{f:04d}{suffix}"
        nib.save(img, str(path))
        written.append(path)
    return written
