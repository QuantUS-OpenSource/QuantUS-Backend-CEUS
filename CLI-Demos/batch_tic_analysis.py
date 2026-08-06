#!/usr/bin/env python3
"""
Batch TIC (Time-Intensity Curve) analysis for CEUS mp4 cines + NIfTI ROI segmentations.

Runs the exact same pipeline as tic_demo.ipynb (mp4 scan loading -> nifti ROI loading ->
TIC curve extraction -> lognormal-fit curve quantification) over every case in a parent
directory, where each case lives in its own subdirectory:

    parent_dir/
        case_1/
            case_1.mp4
            case_1 - ROI.nii.gz
        case_2/
            case_2.mp4
            case_2 - ROI.nii.gz
        ...

For each processed subdirectory this writes, inside that same subdirectory:
    sample.csv       - raw per-frame TIC curve
    parameters.csv   - AUC, PE, TP, MTT from the lognormal fit
    TIC_curve.png    - plot of the raw TIC curve overlaid with its lognormal fit

Each case is processed in its own subprocess (see --process-one) so that the large
in-memory video arrays for one case are fully released by the OS before the next
case starts. This avoids Jupyter/kernel crashes on large mp4 files.

Usage:
    # interactively pick which subdirectories to process
    python batch_tic_analysis.py /path/to/parent_dir

    # process everything, no prompts
    python batch_tic_analysis.py /path/to/parent_dir --subdirs all

    # process specific cases by name
    python batch_tic_analysis.py /path/to/parent_dir --subdirs "case_1,case_3"

    # process specific cases by 1-based index (as shown by --list), incl. ranges
    python batch_tic_analysis.py /path/to/parent_dir --subdirs "1,3-5"

    # just see what would be processed
    python batch_tic_analysis.py /path/to/parent_dir --list
"""

import argparse
import gc
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the `src` package (engines/ceus/src) importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.entrypoints import scan_loading_step, seg_loading_step, analysis_step

# Exit codes used by --process-one so the orchestrating process (main()) can tell
# apart a genuine failure from a skip without parsing stdout.
EXIT_OK = 0
EXIT_SKIPPED = 20
EXIT_FAILED = 21


# ── Lognormal fitting (identical to tic_demo.ipynb) ────────────────────────────
def bolus_lognormal(x, auc, mu, sigma, t0):
    with np.errstate(divide="ignore", invalid="ignore"):
        shifted = x - t0
        result = (auc / (shifted * sigma * np.sqrt(2 * np.pi))) * \
                 np.exp(-((np.log(shifted) - mu) ** 2) / (2 * sigma ** 2))
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def fit_lognormal_curve(time, curve):
    """Returns (auc, pe, tp, mtt, t0, mu, sigma, pe_loc, baseline).
    pe and auc are amplitude above baseline. baseline = amin of input curve."""
    from scipy.optimize import curve_fit

    curve = np.array(curve, dtype=float)
    baseline = float(np.amin(curve))
    curve = curve - baseline  # shift so minimum == 0

    if np.amax(curve) <= 0:
        print("    Curve is constant, cannot normalize.")
        return tuple(np.nan for _ in range(9))
    normalizer = np.amax(curve)
    curve = curve / normalizer  # normalize to 0-1

    auc_guess = np.sum(curve) * (time[1] - time[0])
    mu_guess = np.log(np.argmax(curve) + 1e-8)
    sigma_guess = 0.5
    t0_guess = time[np.argmax(curve)] * 0.15

    mu_max = np.log(time[-1]) if time[-1] > 0 else 10.0
    auc_max = (np.sum(curve) * (time[1] - time[0])) * 10.0
    auc_guess = min(auc_guess, auc_max)
    mu_guess = min(mu_guess, mu_max)

    try:
        params, _ = curve_fit(
            bolus_lognormal, time, curve,
            p0=(auc_guess, mu_guess, sigma_guess, t0_guess),
            bounds=([0., 0., 0.01, 0.], [auc_max, mu_max, 5.0, time[-1]]),
            method="trf", maxfev=10000)
    except Exception as e:
        print(f"    Error fitting curve: {e}")
        return tuple(np.nan for _ in range(9))

    auc, mu, sigma, t0 = params
    auc = auc * normalizer  # amplitude above baseline
    mtt = np.exp(mu + sigma ** 2 / 2)
    tp = np.exp(mu - sigma ** 2)

    fitted_curve = bolus_lognormal(time, *params)  # 0-1 normalized space
    pe = float(np.max(fitted_curve)) * normalizer  # amplitude above baseline
    pe_loc = int(np.argmax(fitted_curve))
    return auc, pe, tp, mtt, t0, mu, sigma, pe_loc, baseline


def reconstruct_fitted_curve(time, outcome):
    auc, pe, tp, mtt, t0, mu, sigma, pe_loc, baseline = outcome
    if np.isnan(auc):
        return np.full_like(time, np.nan, dtype=float)
    return bolus_lognormal(time, auc, mu, sigma, t0) + baseline


# ── Case discovery ──────────────────────────────────────────────────────────
def discover_subdirs(parent_dir: Path) -> list[Path]:
    return sorted((d for d in parent_dir.iterdir() if d.is_dir()), key=lambda p: p.name.lower())


def find_case_files(subdir: Path):
    """Find the mp4 scan and '... - ROI.nii.gz' segmentation inside a case subdirectory."""
    mp4_candidates = sorted(subdir.glob("*.mp4"))
    roi_candidates = sorted(subdir.glob("* - ROI.nii.gz"))
    if not roi_candidates:
        # fall back to any nifti containing "ROI" in the name, then any nifti at all
        roi_candidates = sorted(subdir.glob("*ROI*.nii.gz")) or sorted(subdir.glob("*.nii.gz"))

    warnings = []
    if len(mp4_candidates) > 1:
        warnings.append(f"multiple .mp4 files found, using '{mp4_candidates[0].name}'")
    if len(roi_candidates) > 1:
        warnings.append(f"multiple ROI files found, using '{roi_candidates[0].name}'")

    mp4_path = mp4_candidates[0] if mp4_candidates else None
    roi_path = roi_candidates[0] if roi_candidates else None
    return mp4_path, roi_path, warnings


def parse_selection(selection: str, all_subdirs: list[Path]) -> list[Path]:
    """Resolve a selection string ('all', comma-separated names, and/or 1-based
    indices/ranges like '1,3-5') into a list of subdirectories."""
    selection = selection.strip()
    if selection.lower() == "all":
        return list(all_subdirs)

    names_by_lower = {d.name.lower(): d for d in all_subdirs}
    chosen: list[Path] = []
    for token in selection.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token and token.replace("-", "").isdigit():
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            for i in range(start, end + 1):
                if 1 <= i <= len(all_subdirs):
                    chosen.append(all_subdirs[i - 1])
                else:
                    print(f"[WARN] Index {i} out of range (1-{len(all_subdirs)}), skipping.")
        elif token.isdigit():
            i = int(token)
            if 1 <= i <= len(all_subdirs):
                chosen.append(all_subdirs[i - 1])
            else:
                print(f"[WARN] Index {i} out of range (1-{len(all_subdirs)}), skipping.")
        elif token.lower() in names_by_lower:
            chosen.append(names_by_lower[token.lower()])
        else:
            print(f"[WARN] No subdirectory matching '{token}', skipping.")

    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for d in chosen:
        if d not in seen:
            seen.add(d)
            deduped.append(d)
    return deduped


def prompt_for_selection(all_subdirs: list[Path]) -> list[Path]:
    print(f"\nFound {len(all_subdirs)} subdirectories:")
    for i, d in enumerate(all_subdirs, start=1):
        print(f"  [{i}] {d.name}")
    print("\nEnter which to process: 'all', a comma-separated list of names/indices,")
    print("or index ranges (e.g. '1,3-5'). Leave blank to cancel.")
    selection = input("> ").strip()
    if not selection:
        return []
    return parse_selection(selection, all_subdirs)


# ── Per-case processing ─────────────────────────────────────────────────────
def process_case(subdir: Path, scan_loader_kwargs: dict, seg_loader_kwargs: dict) -> str:
    """Returns 'ok', 'skipped', or 'failed'."""
    mp4_path, roi_path, warnings = find_case_files(subdir)
    for w in warnings:
        print(f"  [WARN] {w}")

    if mp4_path is None:
        print(f"  [SKIP] No .mp4 file found in '{subdir.name}'.")
        return "skipped"
    if roi_path is None:
        print(f"  [SKIP] No '... - ROI.nii.gz' file found in '{subdir.name}'.")
        return "skipped"

    print(f"  Scan: {mp4_path.name}")
    print(f"  ROI:  {roi_path.name}")

    image_data = seg_data = analysis_obj = None
    try:
        image_data = scan_loading_step("mp4", str(mp4_path), **scan_loader_kwargs)
        seg_data = seg_loading_step("nifti", image_data, str(roi_path), str(mp4_path), **seg_loader_kwargs)

        roi_pixels = int(np.sum(seg_data.seg_mask > 0))
        if roi_pixels == 0:
            print("  [WARN] ROI mask has 0 pixels selected — TIC curve will be empty/NaN. "
                  "Check that the segmentation's shape/orientation matches the video frames.")

        sample_csv = subdir / "sample.csv"
        analysis_obj = analysis_step(
            "curves", image_data, seg_data, ["tic"],
            curves_output_path=str(sample_csv),
        )
        print(f"  Saved raw TIC curve -> {sample_csv.name}  "
              f"({len(analysis_obj.curves[0].get('TIC', []))} frames, {roi_pixels} ROI pixels)")

        time_arr = np.asarray(analysis_obj.time_arr, dtype=float)
        tic_curve = np.asarray(analysis_obj.curves[0]["TIC"], dtype=float)

        fit = fit_lognormal_curve(time_arr, tic_curve)
        fitted_curve = reconstruct_fitted_curve(time_arr, fit)
        auc, pe, tp, mtt = fit[0], fit[1], fit[2], fit[3]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_arr, tic_curve, label="TIC")
        ax.plot(time_arr, fitted_curve, label="Lognormal fit")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Average Intensity")
        ax.set_title(subdir.name)
        ax.legend()
        fig.tight_layout()
        png_path = subdir / "TIC_curve.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  Saved TIC plot -> {png_path.name}")

        params_csv = subdir / "parameters.csv"
        pd.DataFrame([{
            "Scan Name": image_data.scan_name,
            "Segmentation Name": seg_data.seg_name,
            "AUC": auc,
            "PE": pe,
            "TP": tp,
            "MTT": mtt,
        }]).to_csv(params_csv, index=False)
        print(f"  Saved parameters -> {params_csv.name}")

        if np.isnan(auc):
            print("  [WARN] Lognormal fit failed (curve is constant or fit did not converge); "
                  "AUC/PE/TP/MTT are NaN.")
        print(f"  AUC={auc:.4g}  PE={pe:.4g}  TP={tp:.4g}  MTT={mtt:.4g}")

        return "ok"
    except Exception:
        print(f"  [FAILED] {subdir.name}:")
        traceback.print_exc()
        return "failed"
    finally:
        # Video arrays can be gigabytes in size; drop references and force collection
        # before the next case is (potentially) processed in this same process.
        del image_data, seg_data, analysis_obj
        gc.collect()


def run_case_subprocess(parent_dir: Path, case_name: str,
                         scan_loader_kwargs: dict | None = None,
                         seg_loader_kwargs: dict | None = None) -> str:
    """Run a single case in a fresh subprocess so its (potentially many-GB) video
    arrays are fully reclaimed by the OS before the next case starts. Prefer this
    over calling process_case() directly in a loop -- e.g. from a notebook -- when
    processing more than one case, to avoid the host process/kernel running out of
    memory. Streams the subprocess's stdout/stderr live. Returns 'ok', 'skipped',
    or 'failed'.
    """
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        str(parent_dir), "--process-one", case_name,
        "--scan-loader-kwargs", json.dumps(scan_loader_kwargs or {"transpose": False}),
        "--seg-loader-kwargs", json.dumps(seg_loader_kwargs or {}),
    ]
    result = subprocess.run(cmd)
    if result.returncode == EXIT_OK:
        return "ok"
    if result.returncode == EXIT_SKIPPED:
        return "skipped"
    return "failed"


def main():
    parser = argparse.ArgumentParser(
        description="Batch TIC analysis over CEUS mp4 + NIfTI ROI case subdirectories.")
    parser.add_argument("parent_dir", type=str,
                        help="Directory containing one subdirectory per case.")
    parser.add_argument("--subdirs", type=str, default=None,
                        help="Which subdirectories to process: 'all', comma-separated names, "
                             "and/or 1-based indices/ranges (e.g. '1,3-5'). "
                             "If omitted, you'll be prompted interactively.")
    parser.add_argument("--list", action="store_true",
                        help="List discovered subdirectories with their indices and exit.")
    parser.add_argument("--scan-loader-kwargs", type=str, default='{"transpose": false}',
                        help="JSON kwargs passed to the mp4 scan loader.")
    parser.add_argument("--seg-loader-kwargs", type=str, default="{}",
                        help="JSON kwargs passed to the nifti segmentation loader.")
    parser.add_argument("--process-one", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    parent_dir = Path(args.parent_dir).expanduser().resolve()
    if not parent_dir.is_dir():
        print(f"Error: '{parent_dir}' is not a directory.")
        return 1

    all_subdirs = discover_subdirs(parent_dir)
    if not all_subdirs:
        print(f"No subdirectories found in '{parent_dir}'.")
        return 1

    scan_loader_kwargs = json.loads(args.scan_loader_kwargs)
    seg_loader_kwargs = json.loads(args.seg_loader_kwargs)

    if args.process_one is not None:
        # Internal: invoked by run_case_subprocess()/main()'s own loop to process
        # exactly one case in this (disposable) process, then exit.
        match = next((d for d in all_subdirs if d.name == args.process_one), None)
        if match is None:
            print(f"Error: subdirectory '{args.process_one}' not found under '{parent_dir}'.")
            return EXIT_FAILED
        status = process_case(match, scan_loader_kwargs, seg_loader_kwargs)
        return {"ok": EXIT_OK, "skipped": EXIT_SKIPPED, "failed": EXIT_FAILED}[status]

    if args.list:
        print(f"Subdirectories in '{parent_dir}':")
        for i, d in enumerate(all_subdirs, start=1):
            mp4_path, roi_path, _ = find_case_files(d)
            status = "ready" if mp4_path and roi_path else "incomplete"
            print(f"  [{i}] {d.name}  ({status})")
        return 0

    if args.subdirs:
        selected = parse_selection(args.subdirs, all_subdirs)
    else:
        selected = prompt_for_selection(all_subdirs)

    if not selected:
        print("No subdirectories selected. Nothing to do.")
        return 0

    print(f"\nProcessing {len(selected)} of {len(all_subdirs)} subdirectories "
          f"(each in its own subprocess):")
    for d in selected:
        print(f"  - {d.name}")

    results = {}
    for d in selected:
        print(f"\n=== {d.name} ===")
        results[d.name] = run_case_subprocess(parent_dir, d.name, scan_loader_kwargs, seg_loader_kwargs)

    ok = [n for n, r in results.items() if r == "ok"]
    skipped = [n for n, r in results.items() if r == "skipped"]
    failed = [n for n, r in results.items() if r == "failed"]

    print("\n=== Summary ===")
    print(f"  Succeeded: {len(ok)}")
    print(f"  Skipped:   {len(skipped)}  {skipped if skipped else ''}")
    print(f"  Failed:    {len(failed)}  {failed if failed else ''}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())