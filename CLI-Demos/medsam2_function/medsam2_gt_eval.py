"""
Ground-truth evaluation of a fine-tuned MedSAM2 checkpoint on the annotated
single-frame 3D cases under {finetune_root}/processed/.

Companion to MedSam2_3D_GT.ipynb -- the notebook picks checkpoints and draws
figures, the actual case discovery / metric math lives here so it can be
reused and diffed, same split of responsibilities as medsam2_3d_mask.py vs.
MedSam2_inference.ipynb.

Two things this module is deliberate about:

1) **Where the split comes from.** Every image.json in the tree carries
   `recommended_split: "train"`, including cases that were never trained on,
   so that field is stale and is ignored here. The only trustworthy record of
   what the trainer actually saw is which .npz files exist in npz_train/ (it
   reads that folder and nothing else -- see README.md
   Step 3), so the split is derived from those filenames. Cases are graded
   into three tiers:

     "holdout"      -- the *subject* never appears in npz_train. The only tier
                       that supports a generalization claim.
     "seen_subject" -- a different sequence of a subject that was trained on.
                       Weaker: same patient, same lesion, different bolus, so
                       the model has seen this anatomy. Reported separately,
                       never pooled into the headline number.
     "train"        -- this exact case was trained on. Included only as a
                       sanity/fit check; a low score here means training
                       failed, a high score means nothing about generalization.

2) **The score is prompt-conditional.** compute_3d_mask() is a *prompted*
   segmenter: it needs a 3D box, and at inference time in the 4D pipeline that
   box comes from the motion-tracked reference VOI. These extracted frames have
   no tracked bbox, so the box is derived from the GT mask itself. That makes
   every number here conditional on a good box -- it measures "segmentation
   given a correct box", not detection. bbox_from_mask() takes a `jitter_vox`
   argument so the notebook can perturb the box and show how fast the metrics
   decay, which is the honest way to report a GT-seeded prompt.

All scoring is delegated to **MONAI** (`monai.metrics`) rather than hand-rolled,
so the numbers are directly comparable to anything else in the medical-imaging
literature and the implementation is someone else's problem to validate.

Distance metrics are computed in **millimetres**: each case's own
spacing_mm_xyz from image.json is passed to MONAI as `spacing=`. These volumes
are anisotropic (~0.31 x 0.57 x 0.42 mm here), so voxel-space Hausdorff/ASSD
would be meaningless.
"""

import glob
import json
import os
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import torch
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    MeanIoU,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
)

from medsam2_3d_mask import compute_3d_mask
from medsam2_viewer_base import smooth_3d_mask


# ─────────────────────────────────────────────────────────────────────────────
# Case discovery
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Bbox3D:
    """Minimal stand-in for src.seg_preprocessing.motion_compensation_3d.BoundingBox3D.

    compute_3d_mask() only ever reads these six attributes, so defining them
    here keeps this module importable without pulling in the whole motion-
    compensation stack (which needs a loaded 4D CeusSeg to construct).
    """
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int


def find_mask_path(seg_dir):
    """The single GT file in seg_dir, or None. Casing varies across cases
    ("Tumor_segmentation" vs "Tumor_Segmentation"), so this globs rather than
    hardcoding a spelling -- same approach as build_npz_dataset.py."""
    candidates = sorted(glob.glob(os.path.join(seg_dir, "*.nii*")))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"Expected at most 1 segmentation in {seg_dir!r}, found {candidates!r}")
    return candidates[0]


def discover_cases(processed_root, npz_train_dir):
    """
    Every annotated case under processed_root, tagged with its split tier.

    Returns a list of dicts sorted by (tier, case). Cases with no segmentation
    file are skipped entirely -- there is nothing to score them against.
    """
    trained_cases = {
        os.path.basename(p)[: -len(".npz")]
        for p in glob.glob(os.path.join(npz_train_dir, "*.npz"))
    }
    trained_subjects = {c.split("_sequence_")[0] for c in trained_cases}

    cases = []
    for seq_dir in sorted(glob.glob(os.path.join(processed_root, "subject_*", "sequence_*"))):
        subject_id = os.path.basename(os.path.dirname(seq_dir))
        sequence_id = os.path.basename(seq_dir)
        case = f"{subject_id}_{sequence_id}"

        mask_path = find_mask_path(os.path.join(seq_dir, "segmentations"))
        if mask_path is None:
            continue

        if case in trained_cases:
            tier = "train"
        elif subject_id in trained_subjects:
            tier = "seen_subject"
        else:
            tier = "holdout"

        cases.append({
            "case": case,
            "subject_id": subject_id,
            "sequence_id": sequence_id,
            "tier": tier,
            "seq_dir": seq_dir,
            "image_path": os.path.join(seq_dir, "image.nii.gz"),
            "mask_path": mask_path,
            "meta_path": os.path.join(seq_dir, "image.json"),
        })

    tier_order = {"holdout": 0, "seen_subject": 1, "train": 2}
    return sorted(cases, key=lambda c: (tier_order[c["tier"]], c["case"]))


def load_case(case):
    """Volume, GT mask and mm spacing for one case, all in (X, Y, Z)."""
    volume = np.asarray(nib.load(case["image_path"]).dataobj)
    gt = np.asarray(nib.load(case["mask_path"]).dataobj) > 0

    if volume.shape != gt.shape:
        raise ValueError(f"{case['case']}: image {volume.shape} vs mask {gt.shape}")

    with open(case["meta_path"]) as f:
        meta = json.load(f)
    spacing = tuple(float(s) for s in meta["spacing_mm_xyz"])

    return volume, gt, spacing, meta


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def bbox_from_mask(mask, padding_vox=4, jitter_vox=0, rng=None):
    """
    Tight 3D box around `mask`, padded and optionally jittered.

    `jitter_vox` shifts each of the six faces by an independent uniform draw
    from [-jitter_vox, +jitter_vox]. The box is the model's only spatial cue,
    so sweeping this is what separates "the model segments well" from "the box
    was already the answer" -- see the module docstring.
    """
    xs, ys, zs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty GT mask -- nothing to build a box from")

    lo = np.array([xs.min(), ys.min(), zs.min()], dtype=float)
    hi = np.array([xs.max() + 1, ys.max() + 1, zs.max() + 1], dtype=float)

    lo -= padding_vox
    hi += padding_vox

    if jitter_vox:
        rng = rng or np.random.default_rng()
        lo += rng.uniform(-jitter_vox, jitter_vox, size=3)
        hi += rng.uniform(-jitter_vox, jitter_vox, size=3)

    shape = np.array(mask.shape, dtype=float)
    lo = np.clip(lo, 0, shape - 1)
    hi = np.clip(hi, lo + 2, shape)          # keep every extent >= 2 voxels

    return Bbox3D(
        x_min=int(lo[0]), x_max=int(hi[0]),
        y_min=int(lo[1]), y_max=int(hi[1]),
        z_min=int(lo[2]), z_max=int(hi[2]),
    )


def guide_slices_from_mask(mask):
    """(y_mid, x_mid) for compute_3d_mask's coronal/sagittal guide slices --
    the slice carrying the most mask in each direction, mirroring how
    _frame_geometry() picks them off the motion-compensated mask."""
    return (
        int(np.argmax(mask.sum(axis=(0, 2)))),   # y_mid -> coronal guide
        int(np.argmax(mask.sum(axis=(1, 2)))),   # x_mid -> sagittal guide
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics -- MONAI, overlap and surface, the latter in mm
# ─────────────────────────────────────────────────────────────────────────────
#
# MONAI's metrics take [B, C, *spatial] tensors. Every call here is a single
# case with a single foreground channel, so include_background=True throughout
# -- with C=1 that flag means "score channel 0", and setting it False would
# discard the only channel there is and return NaN.
#
# Each surface metric recomputes its own distance transform internally, so
# asking for ASSD, HD95 and NSD does that work three times over. That is the
# price of using the library's implementations instead of sharing one distance
# field between them, and it is small next to SAM2 inference.

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


def dice(pred, gt):
    return _scalar(DiceMetric(include_background=True)(_as_tensor(pred), _as_tensor(gt)).item())


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


def volume_mm3(mask, spacing):
    return float(mask.astype(bool).sum() * np.prod(spacing))


def all_metrics(pred, gt, spacing, tolerance_mm=1.0):
    """Every metric for one case, as a flat dict."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    pred_vol, gt_vol = volume_mm3(pred, spacing), volume_mm3(gt, spacing)
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "assd_mm": assd(pred, gt, spacing),
        "hd95_mm": hd95(pred, gt, spacing),
        f"nsd@{tolerance_mm}mm": nsd(pred, gt, spacing, tolerance_mm),
        "pred_vol_mm3": pred_vol,
        "gt_vol_mm3": gt_vol,
        "vol_ratio": pred_vol / gt_vol if gt_vol else float("nan"),
        "empty_pred": not pred.any(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inference driver
# ─────────────────────────────────────────────────────────────────────────────

def build_predictor(model_cfg, checkpoint, device="cuda"):
    """One SAM2 image predictor. Kept separate from the case loop so a single
    checkpoint is loaded once and reused across every case."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))


def predict_case(image_predictor, volume, gt, padding_vox=4, jitter_vox=0,
                 spacing=None, smooth_sigma_mm=1.0, rng=None):
    """
    Run the adaptive-bbox reconstruction on one case, with the box seeded from
    `gt`. Returns (pred_mask, bbox).

    smooth_sigma_mm mirrors Medsam2AdaptiveBboxMasker's default so the score
    reflects what the 4D pipeline actually produces; set 0 to score the raw
    per-slice stack instead.
    """
    bbox = bbox_from_mask(gt, padding_vox=padding_vox, jitter_vox=jitter_vox, rng=rng)
    y_mid, x_mid = guide_slices_from_mask(gt)

    pred = compute_3d_mask(image_predictor, volume, bbox, y_mid, x_mid)

    if smooth_sigma_mm:
        pred = smooth_3d_mask(pred, spacing_xyz=spacing, sigma_mm=smooth_sigma_mm)

    return pred.astype(bool), bbox


def evaluate_checkpoint(cases, model_cfg, checkpoint, device="cuda",
                        tiers=("holdout",), padding_vox=4, jitter_vox=0,
                        smooth_sigma_mm=1.0, tolerance_mm=1.0, seed=0,
                        label=None, verbose=True):
    """
    Score one checkpoint over every case in `tiers`.

    Returns a list of per-case metric dicts (one row per case), each tagged
    with the checkpoint label and tier so rows from several checkpoints can be
    concatenated into a single DataFrame.
    """
    label = label or os.path.basename(checkpoint)
    selected = [c for c in cases if c["tier"] in tiers]

    if verbose:
        print(f"\n=== {label} — {len(selected)} case(s) from tiers {tiers} ===")

    predictor = build_predictor(model_cfg, checkpoint, device=device)
    rng = np.random.default_rng(seed)
    rows = []

    for case in selected:
        volume, gt, spacing, _ = load_case(case)
        pred, bbox = predict_case(
            predictor, volume, gt,
            padding_vox=padding_vox, jitter_vox=jitter_vox,
            spacing=spacing, smooth_sigma_mm=smooth_sigma_mm, rng=rng,
        )
        row = {
            "checkpoint": label,
            "case": case["case"],
            "subject_id": case["subject_id"],
            "tier": case["tier"],
            **all_metrics(pred, gt, spacing, tolerance_mm=tolerance_mm),
        }
        rows.append(row)

        if verbose:
            print(f"  {case['case']:<34} dice={row['dice']:.3f}  "
                  f"nsd={row[f'nsd@{tolerance_mm}mm']:.3f}  "
                  f"assd={row['assd_mm']:.2f}mm  hd95={row['hd95_mm']:.2f}mm")

    _free(predictor)
    return rows


def _free(predictor):
    """Drop a predictor's GPU memory before the next checkpoint loads."""
    import gc
    import torch

    del predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
