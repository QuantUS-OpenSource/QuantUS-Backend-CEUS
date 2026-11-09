"""
Pick the cross-validation checkpoint that never saw a given patient.

The MedSAM2 model was fine-tuned with 5-fold, leave-patients-out cross-validation.
Fold k's checkpoint was trained without that fold's held-out patients, so
segmenting a patient with the checkpoint of the fold that held it out gives an
honest result. Inference on any patient in the CV set therefore goes through
`checkpoint_for(site, patient)`.

Layout (under AI_inference/MedSAM2/, or $MEDSAM2_FOLD_DIR):

    checkpoints/fold_1.pt ... fold_5.pt   the checkpoint chosen for each fold
    fold_assignment.json                  {"SITE-P01": 1, ...}, patient -> fold
                                          (patient ids: git-ignored, see below)

`fold_assignment.json` is generated from the training repo's `manifest.csv` and
`kfold_split.json`:

    python fold_checkpoints.py --manifest .../manifest.csv --kfold-split .../kfold/kfold_split.json
"""

import argparse
import csv
import json
import os
from pathlib import Path

# .../QuantUS/engines/ceus/CLI-Demos/medsam2_function/fold_checkpoints.py -> .../QuantUS
_QUANTUS_ROOT = Path(__file__).resolve().parents[4]


def fold_dir():
    return Path(os.environ.get("MEDSAM2_FOLD_DIR", _QUANTUS_ROOT / "AI_inference" / "MedSAM2"))


def _key(site, patient):
    return f"{str(site).strip()}-{str(patient).strip()}"


def load_fold_assignment(path=None):
    """{"SITE-P01": fold_number, ...}"""
    path = Path(path) if path else fold_dir() / "fold_assignment.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No fold assignment at {path}. Generate it with:\n"
            f"  python fold_checkpoints.py --manifest <manifest.csv> --kfold-split <kfold_split.json>"
        )
    with open(path, encoding="utf-8") as f:
        return {k: int(v) for k, v in json.load(f).items()}


def fold_for(site, patient, assignment=None):
    assignment = assignment or load_fold_assignment()
    key = _key(site, patient)
    if key not in assignment:
        raise KeyError(
            f"{key} is not in the cross-validation set, so no fold held it out. "
            f"Use a checkpoint trained on all patients (set MEDSAM2_CHECKPOINT) for new data."
        )
    return assignment[key]


def checkpoint_for(site, patient, assignment=None, verbose=True):
    """Path of the checkpoint of the fold that held out (site, patient)."""
    fold = fold_for(site, patient, assignment)
    path = fold_dir() / "checkpoints" / f"fold_{fold}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{_key(site, patient)} is in fold {fold}, but {path} is missing.")
    if verbose:
        print(f"{_key(site, patient)}: fold {fold} (held out) -> {path.name}")
    return str(path)


def build_fold_assignment(manifest_csv, kfold_split_json, out_json=None):
    """Write {site-patient: fold} from the training repo's manifest and kfold_split.json."""
    with open(kfold_split_json, encoding="utf-8") as f:
        folds = json.load(f)["folds"]
    fold_of_subject = {s: fold["fold"] for fold in folds for s in fold["holdout_subjects"]}

    assignment = {}
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["subject_id"] in fold_of_subject:
                assignment[_key(row["site"], row["patient_number"])] = fold_of_subject[row["subject_id"]]

    out_json = Path(out_json) if out_json else fold_dir() / "fold_assignment.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(assignment.items())), f, indent=2)
    print(f"{len(assignment)} patients across {len(set(assignment.values()))} folds -> {out_json}")
    return assignment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--kfold-split", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    build_fold_assignment(args.manifest, args.kfold_split, args.out)
