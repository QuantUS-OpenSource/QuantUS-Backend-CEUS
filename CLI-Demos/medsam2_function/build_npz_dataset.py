"""
Convert labeled cases from the MedSAM2 fine-tuning tree
({processed_root}/{subject_id}/{sequence_id}/image.nii.gz +
segmentations/*.nii.gz) into the .npz "video" format MedSAM2's trainer reads
(training.dataset.vos_raw_dataset.NPZRawDataset): one {subject_id}_{sequence_id}.npz
per case, holding 'imgs' (T, H, W) uint8 and 'gts' (T, H, W) integer labels,
where T is the Z (axial slice) dimension — MedSAM2 treats a 3D volume's
slice stack as a "video" and propagates the mask across it, same as at
inference.

Only sequences with a non-empty segmentations/ folder are converted; the rest
are skipped (report printed at the end). Mask filename casing varies across
cases ("Tumor_segmentation.nii.gz" vs "Tumor_Segmentation.nii.gz"), so this
matches case-insensitively rather than hardcoding one spelling.

The train/holdout split is made here, by **subject**, and is applied in the
same pass that writes the files. Splitting by sequence instead would leak: most
subjects have several sequences of the same lesion at a different visit or
bolus, so holding out one sequence while training on another from that patient
means the model has already seen the anatomy.

Doing the split here rather than by moving files afterwards is deliberate.
Re-running this script over the whole processed/ tree with only --output-dir
set writes *every* labeled case into the training folder, silently sweeping the
held-out cases back in -- which is exactly how a holdout gets lost. With
--holdout-subjects, each case is routed on the way out and any stale copy in
the opposite folder is deleted, so re-running is idempotent and safe.

MedSAM2's NPZRawDataset also supports `file_list_txt` / `excluded_videos_list_txt`,
which would let one folder serve both splits. Two folders are kept instead
because medsam2_gt_eval.discover_cases derives its split tiers from which .npz
files exist in npz_train/ -- that folder is the record of what the trainer saw,
and keeping it literal keeps the evaluation honest.

Usage:
    python build_npz_dataset.py \\
        --processed-root /media/ahmed-el-kaffas/20TB-HDD/Yuanshan/3DMPUS/MedSAM2_finetune_data/processed \\
        --output-dir /media/ahmed-el-kaffas/20TB-HDD/Yuanshan/3DMPUS/MedSAM2_finetune_data/npz_train \\
        --holdout-dir /media/ahmed-el-kaffas/20TB-HDD/Yuanshan/3DMPUS/MedSAM2_finetune_data/npz_holdout \\
        --holdout-subjects subject_000003 subject_000006 subject_000008
"""

import argparse
import glob
import json
import os
from datetime import datetime, timezone

import nibabel as nib
import numpy as np


def find_mask_path(seg_dir):
    """Return the single segmentation file in seg_dir, or None if empty."""
    candidates = sorted(
        f for f in glob.glob(os.path.join(seg_dir, "*.nii*"))
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            f"Expected at most 1 segmentation file in {seg_dir!r}, found {candidates!r}"
        )
    return candidates[0]


def convert_case(seq_dir, out_path):
    img_path = os.path.join(seq_dir, "image.nii.gz")
    mask_path = find_mask_path(os.path.join(seq_dir, "segmentations"))
    if mask_path is None:
        return None

    img = np.asarray(nib.load(img_path).dataobj)          # (X, Y, Z) uint8
    mask = np.asarray(nib.load(mask_path).dataobj)         # (X, Y, Z)

    if img.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch in {seq_dir!r}: image {img.shape} vs mask {mask.shape}"
        )

    # Move the axial (Z) axis to the front -> (T, H, W), the "video" layout
    # NPZRawDataset expects (see training/dataset/vos_raw_dataset.py).
    imgs = np.moveaxis(img, -1, 0).astype(np.uint8)
    gts = np.moveaxis(mask, -1, 0)
    gts = (gts > 0).astype(np.uint8)  # binarize: tumor = 1, background = 0

    n_labeled_slices = int(np.any(gts != 0, axis=(1, 2)).sum())
    np.savez_compressed(out_path, imgs=imgs, gts=gts)
    return imgs.shape, n_labeled_slices


def build_dataset(processed_root, output_dir, holdout_dir=None, holdout_subjects=(),
                  train_cases=()):
    """
    Convert every labeled case, routing each to the train or holdout folder.

    Two ways to split, one of which must be chosen:

      holdout_subjects -- hold out every sequence of the named subjects. Use
                          this when defining a new split.
      train_cases      -- name the exact cases to train on; everything else is
                          held out. Use this to reproduce the set an existing
                          checkpoint was trained on, where the split was by
                          case and not by subject.

    Returns (converted, skipped) where converted rows carry the split each case
    landed in.
    """
    holdout_subjects = set(holdout_subjects or ())
    train_cases = set(train_cases or ())
    if holdout_subjects and train_cases:
        raise ValueError("Pass --holdout-subjects or --train-cases, not both")

    os.makedirs(output_dir, exist_ok=True)
    if holdout_subjects or train_cases:
        if not holdout_dir:
            raise ValueError("--holdout-subjects/--train-cases given without --holdout-dir")
        os.makedirs(holdout_dir, exist_ok=True)

    all_cases = {
        os.path.basename(os.path.dirname(d)) + "_" + os.path.basename(d)
        for d in glob.glob(os.path.join(processed_root, "subject_*", "sequence_*"))
    }
    all_subjects = {c.split("_sequence_")[0] for c in all_cases}

    # A mistyped id would silently produce an empty holdout and a training set
    # that quietly contains everything (or the reverse) -- the exact failure
    # this routing exists to prevent, so it is an error rather than a warning.
    unknown = sorted(holdout_subjects - all_subjects)
    if unknown:
        raise ValueError(
            f"--holdout-subjects not found under {processed_root!r}: {unknown}. "
            f"Known subjects: {sorted(all_subjects)}"
        )
    unknown = sorted(train_cases - all_cases)
    if unknown:
        raise ValueError(
            f"--train-cases not found under {processed_root!r}: {unknown}"
        )

    converted, skipped = [], []
    for seq_dir in sorted(glob.glob(os.path.join(processed_root, "subject_*", "sequence_*"))):
        subject_id = os.path.basename(os.path.dirname(seq_dir))
        sequence_id = os.path.basename(seq_dir)
        case_name = f"{subject_id}_{sequence_id}"

        if train_cases:
            split = "train" if case_name in train_cases else "holdout"
        else:
            split = "holdout" if subject_id in holdout_subjects else "train"
        dest = holdout_dir if split == "holdout" else output_dir
        out_path = os.path.join(dest, f"{case_name}.npz")

        result = convert_case(seq_dir, out_path)
        if result is None:
            skipped.append(case_name)
            continue

        # Drop any copy left in the other folder by an earlier run, so a case
        # can never appear in both splits at once.
        other = output_dir if split == "holdout" else holdout_dir
        if other:
            stale = os.path.join(other, f"{case_name}.npz")
            if os.path.isfile(stale):
                os.remove(stale)
                print(f"[{case_name}] removed stale copy from {other}")

        shape, n_labeled = result
        converted.append((case_name, shape, n_labeled, split))
        print(f"[{case_name}] {shape[0]} slices, {n_labeled} labeled, {split} -> {out_path}")

    n_train = sum(1 for c in converted if c[3] == "train")
    n_hold = len(converted) - n_train
    train_subjects = sorted({c[0].split("_sequence_")[0] for c in converted if c[3] == "train"})

    if holdout_subjects or train_cases:
        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "processed_root": os.path.abspath(processed_root),
            "split_by": "case" if train_cases else "subject",
            "train_dir": os.path.abspath(output_dir),
            "holdout_dir": os.path.abspath(holdout_dir),
            "holdout_subjects": sorted(holdout_subjects),
            "requested_train_cases": sorted(train_cases),
            "train_subjects": train_subjects,
            "train_cases": sorted(c[0] for c in converted if c[3] == "train"),
            "holdout_cases": sorted(c[0] for c in converted if c[3] == "holdout"),
        }
        split_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "split.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"\nSplit recorded in {split_path}")

    print()
    print(f"Train:   {n_train} case(s) over {len(train_subjects)} subject(s) -> {output_dir}")
    if holdout_subjects or train_cases:
        hold_subjects = {c[0].split("_sequence_")[0] for c in converted if c[3] == "holdout"}
        print(f"Holdout: {n_hold} case(s) over {len(hold_subjects)} subject(s) -> {holdout_dir}")
        if train_cases:
            missing = sorted(train_cases - {c[0] for c in converted if c[3] == "train"})
            assert not missing, f"requested train cases not converted (unlabeled?): {missing}"
            # With a case-level split, holdout deliberately contains sequences
            # of subjects that ARE in train -- gt_eval tiers those as
            # "seen_subject", which is the point. Only report the overlap.
            both = sorted(set(train_subjects) & hold_subjects)
            if both:
                print(f"         note: {len(both)} subject(s) appear in both splits "
                      f"(other sequences of a trained patient): {both}")
        else:
            overlap = sorted(set(train_subjects) & holdout_subjects)
            assert not overlap, f"subject in both splits: {overlap}"
    else:
        print("Holdout: none requested — every labeled case went to the training folder.")
    print(f"Skipped {len(skipped)} unlabeled case(s): {skipped}")
    return converted, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-root", required=True, help="Path to .../MedSAM2_finetune_data/processed")
    parser.add_argument("--output-dir", required=True, help="Where to write the training {subject}_{sequence}.npz files")
    parser.add_argument("--holdout-dir", default=None,
                        help="Where to write held-out cases (required with --holdout-subjects)")
    parser.add_argument("--holdout-subjects", nargs="*", default=[],
                        help="Subject ids to hold out, e.g. subject_000003 subject_000006. "
                             "Every sequence of a named subject is held out.")
    parser.add_argument("--train-cases", nargs="*", default=[],
                        help="Exact case ids to train on, e.g. subject_000001_sequence_001. "
                             "Everything else is held out. Use to reproduce an existing "
                             "checkpoint's training set. Mutually exclusive with "
                             "--holdout-subjects.")
    args = parser.parse_args()
    build_dataset(args.processed_root, args.output_dir,
                  holdout_dir=args.holdout_dir,
                  holdout_subjects=args.holdout_subjects,
                  train_cases=args.train_cases)
