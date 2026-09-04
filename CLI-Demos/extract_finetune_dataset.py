"""
Build the MedSAM2 fine-tuning dataset tree from the 3D-MPUS motion-compensation
CSV ("Motion Compensation Comparison 4 patients(MedSAM2).csv").

Which frame to extract from each ~7GB 4D B-mode series is a manual decision
(reviewed case-by-case in QuantUS), so this is a two-phase workflow rather
than a single automated script:

  1) `manifest` — parses the CSV, groups rows into subject_XXX (one per
     patient) / sequence_XXX (one per visit+bolus), locates each case's
     *_BMODE.nii file and its paired *_CEUS.nii file, and writes a manifest
     CSV with an empty `frame_idx` column for you to fill in after reviewing
     each case in QuantUS. Re-running this preserves any frame_idx values
     already filled in at the same output path.

  2) `extract` — for every manifest row with a filled-in frame_idx, extracts
     that one frame from both the B-mode and CEUS volumes (memory-mapped —
     the source files are never fully loaded) and writes:
         {output_root}/processed/{subject_id}/{sequence_id}/image.nii.gz       (B-mode)
         {output_root}/processed/{subject_id}/{sequence_id}/image.json
         {output_root}/processed/{subject_id}/{sequence_id}/image_ceus.nii.gz  (CEUS)
         {output_root}/processed/{subject_id}/{sequence_id}/image_ceus.json
         {output_root}/processed/{subject_id}/{sequence_id}/segmentations/     (empty)

Usage:
    python extract_finetune_dataset.py manifest \\
        --csv "/media/ahmed-el-kaffas/20TB-HDD/Yuanshan/3DMPUS/Motion_compensation_results/Motion Compensation Comparison 4 patients(MedSAM2).csv" \\
        --out manifest.csv

    # ... open manifest.csv, review each case in QuantUS, fill in frame_idx ...

    python extract_finetune_dataset.py extract \\
        --manifest manifest.csv \\
        --output-root /media/ahmed-el-kaffas/20TB-HDD/Yuanshan/3DMPUS/MedSAM2_finetune_data
"""

import argparse
import csv
import glob
import json
import os
import re
from datetime import datetime, timezone

import nibabel as nib
import numpy as np

MANIFEST_FIELDS = [
    "subject_id", "sequence_id", "site", "patient_number", "visit", "bolus",
    "data_dir", "bmode_path", "ceus_path", "n_frames", "frame_idx", "error",
]


def find_bmode_file(data_dir, visit, bolus):
    """
    Locate the *_BMODE.nii file for one (visit, bolus) case.

    The CSV's "Data Dir" column is unreliable beyond the patient level: for
    most patients it's hardcoded to that patient's *first* visit folder (e.g.
    every TJU-P08 row points at ".../TJU-P08/V01", even the V02/V03 rows), so
    any trailing "/V<digits>" is stripped and the row's own `visit` is always
    re-appended rather than trusted from the column.

    Filenames also aren't uniformly derivable from the CSV fields — the site
    prefix, separators, and bolus labels (CE1 / CEUS1 / CE01RUN2 / CEUS-2 ...)
    vary across sites — so this globs instead of formatting a path. The bolus
    token is normally immediately followed by "_" (the start of the
    acquisition timestamp), which is required first so "CE1" can't match a
    file actually named "...CE1RUN2..."; a looser fallback pattern (bolus
    anywhere in the name) catches the handful of cases with an extra suffix
    (e.g. "CE1-2").

    Some patients (e.g. P011, P03) have no per-visit subfolder at all — every
    visit's files sit flat in the patient directory — so both a nested
    "{base_dir}/{visit}/" layout and a flat "{base_dir}/" layout are tried.
    """
    base_dir = re.sub(r"/V\d+/?$", "", data_dir.rstrip("/"))
    candidate_dirs = [os.path.join(base_dir, visit), base_dir]

    for pattern_tail in (f"*{visit}-{bolus}_*BMODE.nii", f"*{visit}-{bolus}*BMODE.nii"):
        for search_dir in candidate_dirs:
            matches = sorted(glob.glob(os.path.join(search_dir, pattern_tail)))
            if len(matches) == 1:
                return matches[0]

    raise FileNotFoundError(
        f"Expected exactly 1 BMODE file for visit={visit!r} bolus={bolus!r} "
        f"searched in {candidate_dirs!r}"
    )


def find_voi_file(data_dir, visit, bolus):
    """
    Locate the *-MC_VOI.nii.gz reference-segmentation file for one (visit,
    bolus) case -- same directory-resolution and strict/loose glob strategy
    as find_bmode_file, since it lives alongside the BMODE/CEUS files and is
    affected by the same site-prefix irregularities (e.g. CSV Site="TJU" vs
    the actual file prefix "TJUH" for TJU-P014 -- the glob patterns below
    don't require site or patient number to match at all, only visit+bolus
    within the already patient-scoped data_dir, which is what makes this
    robust to that kind of mismatch).
    """
    base_dir = re.sub(r"/V\d+/?$", "", data_dir.rstrip("/"))
    candidate_dirs = [os.path.join(base_dir, visit), base_dir]

    for pattern_tail in (f"*{visit}-{bolus}-MC_VOI.nii.gz", f"*{visit}-{bolus}*MC_VOI.nii.gz"):
        for search_dir in candidate_dirs:
            matches = sorted(glob.glob(os.path.join(search_dir, pattern_tail)))
            if len(matches) == 1:
                return matches[0]

    raise FileNotFoundError(
        f"Expected exactly 1 MC_VOI file for visit={visit!r} bolus={bolus!r} "
        f"searched in {candidate_dirs!r}"
    )


def find_ceus_file(bmode_path):
    """
    Locate the *_CEUS.nii file paired with a resolved *_BMODE.nii path.

    B-mode and CEUS are dual-captured together — every case observed so far
    shares the same directory and filename up to the modality suffix (e.g.
    "..._BMODE.nii" / "..._CEUS.nii", same acquisition timestamp), so this
    substitutes the suffix directly rather than globbing again.
    """
    ceus_path = re.sub(r"_BMODE\.nii$", "_CEUS.nii", bmode_path)
    if ceus_path == bmode_path or not os.path.isfile(ceus_path):
        raise FileNotFoundError(
            f"Expected paired CEUS file at {ceus_path!r} (derived from bmode {bmode_path!r})"
        )
    return ceus_path


def build_manifest(csv_path, manifest_out):
    # Preserve any frame_idx values already filled in if a manifest already
    # exists at this path, so re-running `manifest` (e.g. after the CSV
    # changes) doesn't wipe out review work in progress.
    existing_frame_idx = {}
    if os.path.isfile(manifest_out):
        with open(manifest_out, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("frame_idx", "").strip():
                    existing_frame_idx[(r["subject_id"], r["sequence_id"])] = r["frame_idx"]

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # subject_id: one per (site, patient), sorted for determinism
    patients = sorted({(r["Site"], r["Patient Number"]) for r in rows})
    subject_id_for = {patient: f"subject_{i + 1:06d}" for i, patient in enumerate(patients)}

    by_patient = {}
    for r in rows:
        by_patient.setdefault((r["Site"], r["Patient Number"]), []).append(r)

    manifest_rows = []
    for patient, patient_rows in by_patient.items():
        subject_id = subject_id_for[patient]
        # sequence_id: one per (visit, bolus) within a subject, sorted for determinism
        patient_rows_sorted = sorted(patient_rows, key=lambda r: (r["Visit"], r["Bolus"]))
        for seq_i, r in enumerate(patient_rows_sorted):
            data_dir = r["Data Dir"].strip()
            visit = r["Visit"].strip()
            bolus = r["Bolus"].strip()
            sequence_id = f"sequence_{seq_i + 1:03d}"
            errors = []
            bmode_path = ceus_path = n_frames = ""
            try:
                bmode_path = find_bmode_file(data_dir, visit, bolus)
                n_frames = nib.load(bmode_path).shape[-1]   # header only, not a full load
            except Exception as e:
                errors.append(str(e))

            if bmode_path:
                try:
                    ceus_path = find_ceus_file(bmode_path)
                except Exception as e:
                    errors.append(str(e))

            manifest_rows.append({
                "subject_id": subject_id,
                "sequence_id": sequence_id,
                "site": patient[0],
                "patient_number": patient[1],
                "visit": visit,
                "bolus": bolus,
                "data_dir": data_dir,
                "bmode_path": bmode_path,
                "ceus_path": ceus_path,
                "n_frames": n_frames,
                "frame_idx": existing_frame_idx.get((subject_id, sequence_id), ""),
                "error": "; ".join(errors),
            })

    with open(manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_ok = sum(1 for r in manifest_rows if not r["error"])
    print(f"Wrote {len(manifest_rows)} rows to {manifest_out} "
          f"({n_ok} located, {len(manifest_rows) - n_ok} need attention — see the 'error' column).")
    print("Next: review each case in QuantUS and fill in the 0-indexed 'frame_idx' column, "
          "then run the 'extract' command.")
    return manifest_rows


DATASET_NAME = "3DMPUS"
DEFAULT_DIAGNOSTIC_CLASS = "Malignant"       # every current case is malignant
DEFAULT_MODALITY = "3D Ultrasound - Philips X6-1"

ROLE_INFO = {
    "image": {
        "image_stem": "image",
        "image_kind": "3D volume — single timepoint extracted from a 4D B-mode series",
        "source_action": "Extract single frame from 4D B-mode NIfTI volume",
    },
    "ceus": {
        "image_stem": "image_ceus",
        "image_kind": "3D volume — single timepoint extracted from a 4D CEUS (contrast-enhanced) series",
        "source_action": "Extract single frame from 4D CEUS NIfTI volume",
    },
}


def derive_native_subject_id(bmode_path):
    """
    Derive the native (human-readable) subject id from a BMODE filename — the
    prefix before the "-V<digits>-" visit marker, e.g.
    "UCSD-P05-V01-CE1_11.24.01_..._BMODE.nii" -> "UCSD-P05".

    Filenames aren't uniformly prefixed (some omit the site, some have extra
    tokens like "MPUS", patient numbers are sometimes zero-padded differently
    than in the CSV — e.g. "TJU-P6" vs CSV's "P06"), so this reads the id
    straight from the file rather than reconstructing it from CSV columns.
    """
    basename = os.path.basename(bmode_path)
    match = re.match(r"(.+?)-V\d+-", basename)
    if not match:
        raise ValueError(f"Could not derive a native subject id from filename: {basename!r}")
    return match.group(1)


def extract_single_frame(
    source_path,
    frame_idx,
    subject_id,
    sequence_id,
    output_root,
    role="image",
    native_subject_id=None,
    modality=DEFAULT_MODALITY,
    diagnostic_class=DEFAULT_DIAGNOSTIC_CLASS,
    extra_meta=None,
):
    """
    Extract one frame from one 4D NIfTI volume (B-mode via role="image", or
    the co-registered CEUS series via role="ceus") and save it as
    {output_root}/processed/{subject_id}/{sequence_id}/{image_stem}.nii.gz +
    {image_stem}.json, with an empty segmentations/ folder alongside (created
    once, shared by both roles for the same sequence).

    The JSON follows the same field layout as the reference
    Liver.PublicDataProduct.v1 image.json (subject_000066/sequence_001), but
    with values that actually apply to this physically-calibrated 3D volume
    data rather than AUL's 2D JPEG raster corpus — see the field-by-field
    notes below for what changed and why.

    Takes explicit arguments (no manifest row needed) so it can be called
    directly from a notebook for a single case — `extract_case`/`extract_all`
    are a thin wrapper over this for the CSV-driven batch workflow.
    """
    frame_idx = int(frame_idx)
    role_info = ROLE_INFO[role]

    img = nib.load(source_path)                      # lazy — header only, no data read yet
    n_frames = img.shape[-1]
    if not (0 <= frame_idx < n_frames):
        raise ValueError(f"frame_idx={frame_idx} out of range [0, {n_frames}) for {source_path}")

    # Slice only the requested frame off disk — these files are several GB,
    # never materialize the full 4D array.
    frame = np.asarray(img.dataobj[..., frame_idx]).astype(img.get_data_dtype())  # (X, Y, Z)

    out_dir = os.path.join(output_root, "processed", subject_id, sequence_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "segmentations"), exist_ok=True)

    out_img = nib.Nifti1Image(frame, img.affine)
    nib.save(out_img, os.path.join(out_dir, f"{role_info['image_stem']}.nii.gz"))

    # Only the B-mode filename is the canonical naming source — for a "ceus"
    # extraction, the caller (extract_bmode_and_ceus_frame) passes the id
    # already derived from the paired bmode_path rather than the ceus path.
    native_subject_id = native_subject_id or derive_native_subject_id(source_path)
    zooms = img.header.get_zooms()

    meta = {
        # dataset/geometry basics
        "dataset": DATASET_NAME,
        "diagnostic_class": diagnostic_class,
        # AUL's raster JPEGs have no physical calibration ("unit_index_space_nonphysical");
        # these are affine-calibrated NIfTI volumes with real mm spacing instead.
        "geometry_source": "native_nifti_affine_physical_space",
        "image_kind": role_info["image_kind"],
        # "none": this is the raw extracted frame — no denoising/contrast-stretch
        # (e.g. enhance_bmode_noise) or other post-processing has been applied.
        "intensity_normalization": "none",
        # our natural matching key, in place of AUL's "diagnostic class and exact numeric identifier"
        "matching_basis": "site, patient number, visit, and bolus",
        "modality": modality,
        "raster_geometry_policy": "physical_space",
        "recommended_split": "train",
        "resampled": False,
        "resolved_source_paths": [source_path],
        # not yet segmented/reviewed — filled in once a segmentation exists
        "review_record_id": "",
        "reviewed_workbook": "",
        "reviewer_decision": "",
        "reviewer_notes": "",
        "role": role,
        "scan_id": "Ultrasound",
        "schema": "3DMPUS.CEUS_BMODE_finetune.v1",
        "session_id": "",
        "source_action": role_info["source_action"],
        "source_split": "",
        "standardized_sequence_id": sequence_id,
        "standardized_subject_id": subject_id,
        "subject_cohort": "",
        "subject_id": native_subject_id,
        "workbook_source_paths": [],
        # extra provenance not present in the AUL template — useful for a
        # volumetric extraction, so kept alongside rather than dropped.
        "source_frame_index": frame_idx,
        "source_n_frames": n_frames,
        "shape_xyz": list(frame.shape),
        "spacing_mm_xyz": [float(z) for z in zooms[:3]],
        "time_dim_header_spacing": float(zooms[3]) if len(zooms) > 3 else None,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)

    canonical_scan_id = f"{DATASET_NAME}|{native_subject_id}||{meta['scan_id']}"
    meta["canonical_scan_id"] = canonical_scan_id

    with open(os.path.join(out_dir, f"{role_info['image_stem']}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_dir


def extract_bmode_and_ceus_frame(
    bmode_path, ceus_path, frame_idx, subject_id, sequence_id, output_root,
    native_subject_id=None, **kwargs
):
    """Extract the same frame_idx from both the B-mode and co-registered CEUS series."""
    # Derive once from the B-mode filename and reuse for both, so the paired
    # ceus record can't end up with a different id than its bmode sibling.
    native_subject_id = native_subject_id or derive_native_subject_id(bmode_path)
    bmode_dir = extract_single_frame(
        bmode_path, frame_idx, subject_id, sequence_id, output_root,
        role="image", native_subject_id=native_subject_id, **kwargs
    )
    ceus_dir = extract_single_frame(
        ceus_path, frame_idx, subject_id, sequence_id, output_root,
        role="ceus", native_subject_id=native_subject_id, **kwargs
    )
    assert bmode_dir == ceus_dir
    return bmode_dir


def extract_case(row, output_root):
    extra_meta = {
        "site": row["site"],
        "patient_number": row["patient_number"],
        "visit": row["visit"],
        "bolus": row["bolus"],
    }
    return extract_bmode_and_ceus_frame(
        bmode_path=row["bmode_path"],
        ceus_path=row["ceus_path"],
        frame_idx=row["frame_idx"],
        subject_id=row["subject_id"],
        sequence_id=row["sequence_id"],
        output_root=output_root,
        native_subject_id=derive_native_subject_id(row["bmode_path"]),
        extra_meta=extra_meta,
    )


def extract_all(manifest_path, output_root):
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    done, skipped, failed = 0, 0, 0
    for row in rows:
        if not row.get("frame_idx", "").strip():
            skipped += 1
            continue
        if not row.get("ceus_path", "").strip():
            print(f"[{row['subject_id']}/{row['sequence_id']}] SKIPPED — no ceus_path in manifest "
                  f"(re-run the 'manifest' command to re-resolve it)")
            failed += 1
            continue
        out_dir = extract_case(row, output_root)
        print(f"[{row['subject_id']}/{row['sequence_id']}] {row['site']}-{row['patient_number']} "
              f"{row['visit']} {row['bolus']} frame {row['frame_idx']} "
              f"-> {out_dir} (image.* + image_ceus.*)")
        done += 1

    print(f"Extracted {done} case(s), skipped {skipped} (no frame_idx set yet), "
          f"{failed} failed (missing ceus_path).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("manifest", help="Build manifest.csv from the source CSV")
    p_manifest.add_argument("--csv", required=True)
    p_manifest.add_argument("--out", default="manifest.csv")

    p_extract = sub.add_parser("extract", help="Extract frames listed in manifest.csv")
    p_extract.add_argument("--manifest", required=True)
    p_extract.add_argument("--output-root", required=True)

    args = parser.parse_args()
    if args.command == "manifest":
        build_manifest(args.csv, args.out)
    elif args.command == "extract":
        extract_all(args.manifest, args.output_root)
