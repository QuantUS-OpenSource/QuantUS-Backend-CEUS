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

Usage:
    python build_npz_dataset.py \\
        --processed-root /media/ahmed-el-kaffas/20TB-HDD/Yuanshan/3DMPUS/MedSAM2_finetune_data/processed \\
        --output-dir /media/ahmed-el-kaffas/20TB-HDD/Yuanshan/3DMPUS/MedSAM2_finetune_data/npz_train
"""

import argparse
import glob
import os

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


def build_dataset(processed_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    converted, skipped = [], []
    for seq_dir in sorted(glob.glob(os.path.join(processed_root, "subject_*", "sequence_*"))):
        subject_id = os.path.basename(os.path.dirname(seq_dir))
        sequence_id = os.path.basename(seq_dir)
        case_name = f"{subject_id}_{sequence_id}"
        out_path = os.path.join(output_dir, f"{case_name}.npz")

        result = convert_case(seq_dir, out_path)
        if result is None:
            skipped.append(case_name)
            continue
        shape, n_labeled = result
        converted.append((case_name, shape, n_labeled))
        print(f"[{case_name}] {shape[0]} slices, {n_labeled} labeled -> {out_path}")

    print()
    print(f"Converted {len(converted)} case(s) to {output_dir}")
    print(f"Skipped {len(skipped)} unlabeled case(s): {skipped}")
    return converted, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-root", required=True, help="Path to .../MedSAM2_finetune_data/processed")
    parser.add_argument("--output-dir", required=True, help="Where to write {subject}_{sequence}.npz files")
    args = parser.parse_args()
    build_dataset(args.processed_root, args.output_dir)
