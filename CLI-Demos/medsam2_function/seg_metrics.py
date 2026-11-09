"""
Overlap / surface-distance metrics (MONAI) used by the 2D temporal evaluation.

Copied from the training repo's medsam2_gt_eval.py (3DMPUS-segmentation/eval), which
also holds the full 3D evaluation harness. Distances are in mm: pass each case's own
voxel `spacing`.
"""

import numpy as np
import torch
from monai.metrics import (
    HausdorffDistanceMetric,
    MeanIoU,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
)


def _as_tensor(mask):
    """(X, Y, Z) mask -> (1, 1, X, Y, Z) float tensor in MONAI's layout."""
    return torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))[None, None]


def _scalar(value):
    """MONAI metric output -> plain float, with inf folded to NaN.

    An empty prediction makes MONAI return inf for ASSD (and NaN for HD95).
    Left as inf it would poison any mean taken over cases, so it becomes NaN
    here and the case is flagged via `empty_pred` instead -- the failure stays
    visible in the table rather than silently dragging an average to infinity.
    """
    value = float(value)
    return float("nan") if np.isinf(value) else value


def iou(pred, gt):
    return _scalar(MeanIoU(include_background=True)(_as_tensor(pred), _as_tensor(gt)).item())


def assd(pred, gt, spacing):
    """Average symmetric surface distance (mm)."""
    metric = SurfaceDistanceMetric(include_background=True, symmetric=True)
    return _scalar(metric(_as_tensor(pred), _as_tensor(gt), spacing=list(spacing)).item())


def hd95(pred, gt, spacing):
    """95th-percentile symmetric Hausdorff distance (mm)."""
    metric = HausdorffDistanceMetric(include_background=True, percentile=95)
    return _scalar(metric(_as_tensor(pred), _as_tensor(gt), spacing=list(spacing)).item())


def nsd(pred, gt, spacing, tolerance_mm):
    """
    Normalized surface Dice at `tolerance_mm` -- the fraction of both surfaces
    lying within tolerance of the other. Reported alongside Dice because Dice
    alone is size-biased, and because NSD answers the clinically relevant
    question "is the boundary within an acceptable margin" directly. The
    tolerance is not optional context: an NSD without its tolerance quoted
    beside it is unreadable.
    """
    metric = SurfaceDiceMetric(class_thresholds=[tolerance_mm], include_background=True)
    return _scalar(metric(_as_tensor(pred), _as_tensor(gt), spacing=list(spacing)).item())
