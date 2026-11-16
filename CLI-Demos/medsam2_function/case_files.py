"""
Locate the B-mode / CEUS / VOI files of one case from the motion-compensation
CSV, and derive a case's native subject id from its B-mode filename.

These four helpers are what the MedSAM2 *inference* notebooks and
medsam2_2d_temporal_roi.py need. They are copied from the training repo's
extract_finetune_dataset.py (3DMPUS-segmentation/data), where dataset extraction
now lives.
"""

import glob
import os
import re


def find_bmode_file(data_dir, visit, bolus):
    """
    Locate the *_BMODE.nii file for one (visit, bolus) case.

    The CSV's "Data Dir" column is unreliable beyond the patient level: for
    most patients it's hardcoded to that patient's *first* visit folder (e.g.
    every row of a patient points at ".../SITE-P01/V01", even the V02/V03 rows), so
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

    Some patients have no per-visit subfolder at all — every
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
    affected by the same site-prefix irregularities (e.g. CSV Site="SITE" vs
    the actual file prefix "SITEH" -- the glob patterns below
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


def derive_native_subject_id(bmode_path):
    """
    Derive the native (human-readable) subject id from a BMODE filename — the
    prefix before the "-V<digits>-" visit marker, e.g.
    "SITE-P05-V01-CE1_11.24.01_..._BMODE.nii" -> "SITE-P05".

    Filenames aren't uniformly prefixed (some omit the site, some have extra
    tokens like "MPUS", patient numbers are sometimes zero-padded differently
    than in the CSV — e.g. "SITE-P6" vs CSV's "P06"), so this reads the id
    straight from the file rather than reconstructing it from CSV columns.
    """
    basename = os.path.basename(bmode_path)
    match = re.match(r"(.+?)-V\d+-", basename)
    if not match:
        raise ValueError(f"Could not derive a native subject id from filename: {basename!r}")
    return match.group(1)
