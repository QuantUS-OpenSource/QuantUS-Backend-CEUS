#!/usr/bin/env python3
"""
Batch runner for the 2D motion compensation workflow used in QuantUS notebooks.

This script mirrors the logic from `TwoD_CEUS_test/2D_Motion_compensation.ipynb`
and allows you to process multiple patient folders in one go. For every folder
that contains a CEUS cine (`*.mp4`) and one or more segmentations (`*.nii.gz`),
the script:

1. Runs the curve analysis and curve quantification workflow.
2. Saves each segmentation's outputs inside `<timepoint>/Results/` with segmentation-prefixed filenames.
3. Generates numerical paramaps (saved as `.npy`) via the visualization step.
4. Exports selected paramaps (as specified via --paramaps) as PNG overlays inside `<timepoint>/Results/paramaps/`.
5. Appends summary statistics to `<root>/Results/paramap_summary.csv`.

Key switches you can add:   
    --paramaps to choose which paramaps you want PNGs of 
    --recursive if patient/timepoint folders are nested more than one level deep.
    --frame-index 350 to change the background frame for the PNG overlays.
    --use-gamma --gamma 1.2 to enable gamma-corrected overlays (default is 1.5; script has gamma has false by default).
    --overwrite to regenerate outputs even when a Results/<seg>/curve_quantification.csv already exists.
    --log-level DEBUG if you want extra logging.
    --export-raw-arrays to get raw arrays

Example of how to run the script:
    python3 TwoD_CEUS_test/batch_motion_compensation.py \
        "/Volumes/ExternalDrive/ctDNA Data" \
        --paramaps PE_full_TIC MTT_full_TIC \
        --use-gamma \
"""

from __future__ import annotations

import argparse
import json
import sys
import logging
from datetime import datetime
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Use a headless backend before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.entrypoints import (  # noqa: E402
    analysis_step,
    curve_quantification_step,
    scan_loading_step,
    seg_loading_step,
    visualization_step,
)


LOGGER = logging.getLogger("batch_motion_compensation")


# Default parameter names written by the curve quantification plugin.
PARAM_COLUMNS = [
    "AUC_full_TIC",
    "PE_full_TIC",
    "TP_full_TIC",
    "MTT_full_TIC",
    "T0_full_TIC",
    "Mu_full_TIC",
    "Sigma_full_TIC",
    "PE_Ix_full_TIC",
]

# Exclude display metadata for paramaps where downstream consumers don't need it.
EXCLUDED_DISPLAY_STATS = {"PE_full_TIC"}

# Paramaps to export as PNG overlays. Keys correspond to column names in the
# quantification CSV / paramap names returned by the visualization step.
DEFAULT_PARAMAP_EXPORT_SETTINGS = {
    "AUC_full_TIC": {"vmin": None, "vmax": None},
    "PE_full_TIC": {"vmin": 0.0, "vmax": 1.0},
    "TP_full_TIC": {"vmin": None, "vmax": None},
    "MTT_full_TIC": {"vmin": 0.0, "vmax": 300.0},
    "T0_full_TIC": {"vmin": None, "vmax": None},
    "Mu_full_TIC": {"vmin": None, "vmax": None},
    "Sigma_full_TIC": {"vmin": None, "vmax": None},
    "PE_Ix_full_TIC": {"vmin": None, "vmax": None},
}
PARAMAP_CHOICES = tuple(DEFAULT_PARAMAP_EXPORT_SETTINGS) + ("all",)


@dataclass(frozen=True)
class CaseTask:
    case_dir: Path
    scan_path: Path
    seg_path: Path


def _strip_nii_suffix(path: Path) -> str:
    """Return the base filename without .nii or .nii.gz endings."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _seg_matches_scan(seg_path: Path, scan_path: Path) -> bool:
    """Heuristic to decide whether a segmentation was drawn for a given scan."""
    seg_base = _strip_nii_suffix(seg_path).lower()
    scan_base = scan_path.stem.lower()

    if scan_base in seg_base or seg_base in scan_base:
        return True

    scan_tokens = [token for token in scan_base.replace("-", "_").split("_") if token]
    return any(token in seg_base for token in scan_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory containing patient/timepoint folders (e.g., 'ctDNA Data').",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=250,
        help="Frame index used for the grayscale background in exported PNG overlays.",
    )
    parser.add_argument(
        "--use-gamma",
        action="store_true",
        help="Apply gamma correction to paramap overlays before saving PNGs.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.5,
        help="Gamma value used when --use-gamma is enabled.",
    )
    parser.add_argument(
        "--scan-loader-kwargs",
        type=json.loads,
        default=json.dumps({"transpose": False}),
        help="JSON string with keyword arguments passed to the scan loader.",
    )
    parser.add_argument(
        "--seg-loader-kwargs",
        type=json.loads,
        default="{}",
        help="JSON string with keyword arguments passed to the segmentation loader.",
    )
    parser.add_argument(
        "--summary-name",
        default="paramap_summary.csv",
        help="Filename for the aggregated summary CSV stored under <root>/Results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run cases even if curve quantification output already exists.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for mp4/nii.gz pairs inside the root directory.",
    )
    parser.add_argument(
        "--paramaps",
        nargs="+",
        choices=PARAMAP_CHOICES,
        required=True,
        help=(
            "Paramap names to export as PNG overlays (required). Accepts: "
            + ", ".join(DEFAULT_PARAMAP_EXPORT_SETTINGS)
            + ", all."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--export-raw-arrays",
        action="store_true",
        help="Save full-resolution image/segmentation arrays alongside paramaps (disabled by default to avoid large files).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def discover_cases(root_dir: Path, recursive: bool = False) -> List[CaseTask]:
    """Discover all scan/segmentation combinations that should be processed."""

    def in_results(path: Path) -> bool:
        return any(part.lower() == "results" for part in path.parts)

    def is_excluded_dir(path: Path) -> bool:
        return path.name.lower() in {"results", "statistical analysis"}

    case_dirs: List[Path] = []
    if recursive:
        case_dirs = sorted({p.parent for p in root_dir.rglob("*.mp4") if not in_results(p)})
    else:
        case_dirs_set = set()
        for child in root_dir.iterdir():
            if child.is_dir() and not is_excluded_dir(child):
                case_dirs_set.add(child)
        if any(root_dir.glob("*.mp4")):
            case_dirs_set.add(root_dir)
        case_dirs = sorted(case_dirs_set)

    case_dirs = sorted(case_dirs)

    tasks: List[CaseTask] = []
    for case_dir in case_dirs:
        if recursive:
            mp4_candidates = sorted(p for p in case_dir.rglob("*.mp4") if not in_results(p))
        else:
            mp4_candidates = sorted(case_dir.glob("*.mp4"))
            if not mp4_candidates:
                mp4_candidates = sorted(
                    p for p in case_dir.glob("**/*.mp4") if not in_results(p)
                )

        if not mp4_candidates:
            LOGGER.debug("No mp4 found in %s; skipping.", case_dir)
            continue

        if recursive:
            seg_candidates = sorted(p for p in case_dir.rglob("*.nii.gz") if not in_results(p))
        else:
            seg_candidates = sorted(case_dir.glob("*.nii.gz"))
            if not seg_candidates:
                seg_candidates = sorted(
                    p for p in case_dir.glob("**/*.nii.gz") if not in_results(p)
                )
        if not seg_candidates:
            LOGGER.warning("No segmentation (.nii.gz) found for case '%s'. Skipping.", case_dir)
            continue

        for scan_path in mp4_candidates:
            same_dir_segs = [seg for seg in seg_candidates if seg.parent == scan_path.parent]
            if same_dir_segs:
                matching_segs = same_dir_segs
            else:
                matching_segs = [seg for seg in seg_candidates if _seg_matches_scan(seg, scan_path)]
            if not matching_segs:
                # If no obvious match, process every segmentation with this scan.
                matching_segs = seg_candidates
            for seg_path in matching_segs:
                tasks.append(CaseTask(case_dir=case_dir, scan_path=scan_path, seg_path=seg_path))

    return tasks


def maybe_fix_segmentation_orientation(seg_mask: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    """Transpose the segmentation mask if it looks misaligned with the image data."""
    if seg_mask.ndim != 3 or len(image_shape) < 3:
        return seg_mask

    # Image shape is (frames, height, width) for 2D + time data.
    frames, height, width = image_shape[:3]

    # Notebook manually transposed (2, 1, 0). Detect that scenario.
    if seg_mask.shape == (width, height, frames):
        LOGGER.debug("Transposing segmentation from (W, H, T) to (T, H, W).")
        return np.transpose(seg_mask, (2, 1, 0))

    if seg_mask.shape == (height, width, frames):
        return seg_mask

    # Fallback: return original mask.
    return seg_mask


def ensure_segmentation_frame_count(seg_mask: np.ndarray, num_frames: int) -> np.ndarray:
    """Pad or truncate the segmentation mask so its frame axis matches the scan."""
    if seg_mask.ndim != 3:
        return seg_mask

    current_frames = seg_mask.shape[0]
    if current_frames == num_frames:
        return seg_mask

    if current_frames > num_frames:
        LOGGER.warning(
            "Segmentation has %d frames; truncating to match scan frames (%d).",
            current_frames,
            num_frames,
        )
        return seg_mask[:num_frames]

    LOGGER.warning(
        "Segmentation has %d frames; padding last frame to reach scan frame count (%d).",
        current_frames,
        num_frames,
    )
    if current_frames == 0:
        pad_frame = np.zeros((seg_mask.shape[1], seg_mask.shape[2]), dtype=seg_mask.dtype)
        seg_mask = pad_frame[np.newaxis, ...]
        current_frames = 1

    pad_count = num_frames - current_frames
    pad_slice = seg_mask[-1][np.newaxis, ...]
    pad = np.repeat(pad_slice, pad_count, axis=0)
    return np.concatenate([seg_mask, pad], axis=0)


def process_case(
    case_dir: Path,
    scan_path: Path,
    seg_path: Path,
    results_dir: Path,
    frame_index: int,
    use_gamma: bool,
    gamma: float,
    scan_loader_kwargs: Dict,
    seg_loader_kwargs: Dict,
    paramap_settings: Dict[str, Dict[str, float]],
    export_raw_arrays: bool,
) -> Dict[str, float]:
    """Execute the QuantUS workflow for a single case and return summary metrics."""
    try:
        subdir = seg_path.parent.relative_to(case_dir)
    except ValueError:
        subdir = Path(".")

    LOGGER.info(
        "Processing case: %s | subdir: %s | scan: %s | segmentation: %s",
        case_dir,
        subdir,
        scan_path.name,
        seg_path.name,
    )
    seg_base = _strip_nii_suffix(seg_path)

    image_data = scan_loading_step("mp4", str(scan_path), **scan_loader_kwargs)
    seg_data = seg_loading_step("nifti", image_data, str(seg_path), str(scan_path), **seg_loader_kwargs)

    if seg_data.seg_mask.ndim == 3:
        LOGGER.debug("Transposing segmentation axes to match motion-compensated frame layout.")
        seg_data.seg_mask = seg_data.seg_mask.transpose(2, 1, 0)

    seg_data.seg_mask = maybe_fix_segmentation_orientation(seg_data.seg_mask, image_data.pixel_data.shape)
    seg_data.seg_mask = ensure_segmentation_frame_count(
        seg_data.seg_mask, image_data.intensities_for_analysis.shape[0]
    )

    analysis_kwargs = {
        "ax_vox_ovrlp": 50.0,
        "sag_vox_ovrlp": 50.0,
        "cor_vox_ovrlp": 50.0,
        "ax_vox_len": 30.0,
        "sag_vox_len": 30.0,
        "cor_vox_len": 30.0,
        "curves_output_path": "",
    }
    analysis_type = "curves_paramap"
    analysis_funcs = ["tic"]
    analysis_obj = analysis_step(analysis_type, image_data, seg_data, analysis_funcs, **analysis_kwargs)

    results_dir.mkdir(parents=True, exist_ok=True)
    curve_csv_path = results_dir / f"{seg_base}_curve_quantification.csv"
    quant_funcs = ["lognormal_fit_full"]
    curve_quant_kwargs = {"curves_to_fit": ["TIC"]}
    with contextlib.redirect_stdout(io.StringIO()):
        curve_quant = curve_quantification_step(
            analysis_obj, quant_funcs, str(curve_csv_path), **curve_quant_kwargs
        )

    paramap_dir = results_dir / "paramaps"
    vis_kwargs = {
        "paramap_folder_path": str(paramap_dir),
        "hide_all_visualizations": False,
        "export_raw_arrays": export_raw_arrays,
    }
    vis_obj = visualization_step(curve_quant, "paramap", [], [], **vis_kwargs)

    # Rename paramap npy files to include segmentation context (avoids collisions without extra folders).
    for param_name in getattr(vis_obj, "paramap_names", []):
        src_npy = paramap_dir / f"{param_name}_numerical.npy"
        dest_npy = paramap_dir / f"{seg_base}_{param_name}_numerical.npy"
        if src_npy.exists():
            if dest_npy.exists():
                dest_npy.unlink()
            src_npy.rename(dest_npy)

    exported_stats = export_paramap_pngs(
        image_data.pixel_data,
        vis_obj,
        paramap_dir,
        frame_index=frame_index,
        use_gamma=use_gamma,
        gamma=gamma,
        paramap_settings=paramap_settings,
        name_prefix=seg_base,
    )

    summary_metrics = compute_summary_metrics(curve_csv_path, exported_stats)
    ordered_row = {
        "scan_name": scan_path.name,
        "segmentation_name": seg_path.name,
        **summary_metrics,
    }
    return ordered_row


def export_paramap_pngs(
    pixel_data: np.ndarray,
    vis_obj,
    paramap_dir: Path,
    frame_index: int,
    use_gamma: bool,
    gamma: float,
    paramap_settings: Dict[str, Dict[str, float]],
    name_prefix: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Export selected paramaps as PNG overlays and return display stats."""
    paramap_dir.mkdir(parents=True, exist_ok=True)

    frame_index = int(np.clip(frame_index, 0, pixel_data.shape[0] - 1))
    base_frame = pixel_data[frame_index]

    paramap_lookup = {
        name: np.array(paramap, copy=True)
        for name, paramap in zip(vis_obj.paramap_names, vis_obj.numerical_paramaps)
    }

    display_stats: Dict[str, Dict[str, float]] = {}

    prefix = f"{name_prefix}_" if name_prefix else ""

    for param_name, settings in paramap_settings.items():
        if param_name not in paramap_lookup:
            LOGGER.warning("Paramap '%s' not found in visualization output; skipping PNG export.", param_name)
            continue

        disp_map = paramap_lookup[param_name]

        if use_gamma:
            finite_mask = np.isfinite(disp_map)
            disp_map = disp_map.copy()
            disp_map[finite_mask] = disp_map[finite_mask] ** (1.0 / gamma)

        vmin = settings.get("vmin")
        vmax = settings.get("vmax")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(base_frame, cmap="gray")
        heat = ax.imshow(disp_map, cmap="jet", vmin=vmin, vmax=vmax, alpha=0.6)
        ax.set_title(f"{param_name} | frame {frame_index}")
        ax.axis("off")
        cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(param_name, rotation=90)

        png_path = paramap_dir / f"{prefix}{param_name}.png"
        fig.savefig(png_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

        display_stats[param_name] = {
            "png_path": str(png_path),
            "frame_index": frame_index,
            "vmin": float(vmin) if vmin is not None else float(np.nanmin(disp_map)),
            "vmax": float(vmax) if vmax is not None else float(np.nanmax(disp_map)),
        }

    return display_stats


def compute_summary_metrics(
    curve_csv_path: Path, display_stats: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Compute average metrics from the curve quantification CSV."""
    df = pd.read_csv(curve_csv_path)

    summary: Dict[str, float] = {}
    for column in PARAM_COLUMNS:
        if column in df.columns:
            summary[f"mean_{column}"] = float(df[column].mean())
        else:
            LOGGER.debug("Column '%s' not present in %s.", column, curve_csv_path)

    # Optionally track display metadata for reference.
    for param_name, stats in display_stats.items():
        if param_name in EXCLUDED_DISPLAY_STATS:
            continue
        summary[f"{param_name}_frame_index"] = stats["frame_index"]
        summary[f"{param_name}_vmin"] = stats["vmin"]
        summary[f"{param_name}_vmax"] = stats["vmax"]

    summary["summary_generated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    return summary


def append_summary_row(root_dir: Path, summary_name: str, row: Dict[str, float]) -> None:
    root_results = root_dir / "Results"
    root_results.mkdir(parents=True, exist_ok=True)
    summary_path = root_results / summary_name

    row_df = pd.DataFrame([row])
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        updated = pd.concat([existing, row_df], ignore_index=True)
    else:
        updated = row_df

    dedupe_keys = [col for col in ("scan_name", "segmentation_name") if col in updated.columns]
    if len(dedupe_keys) == 2:
        updated = updated.drop_duplicates(subset=dedupe_keys, keep="last")

    first_columns = [col for col in ("scan_name", "segmentation_name") if col in updated.columns]
    remaining_columns = [col for col in updated.columns if col not in first_columns]
    column_order = first_columns + remaining_columns
    updated = updated[column_order]

    updated.to_csv(summary_path, index=False)
    LOGGER.info("Appended summary metrics to %s.", summary_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    root_dir = args.root_dir.expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory '{root_dir}' does not exist.")

    if isinstance(args.scan_loader_kwargs, str):
        scan_loader_kwargs = json.loads(args.scan_loader_kwargs)
    else:
        scan_loader_kwargs = args.scan_loader_kwargs

    if isinstance(args.seg_loader_kwargs, str):
        seg_loader_kwargs = json.loads(args.seg_loader_kwargs)
    else:
        seg_loader_kwargs = args.seg_loader_kwargs

    if "all" in args.paramaps:
        selected_paramaps = list(DEFAULT_PARAMAP_EXPORT_SETTINGS)
    else:
        selected_paramaps = list(dict.fromkeys(args.paramaps))

    paramap_settings = {
        name: dict(DEFAULT_PARAMAP_EXPORT_SETTINGS[name]) for name in selected_paramaps
    }

    tasks = discover_cases(root_dir, recursive=args.recursive)
    if not tasks:
        LOGGER.error("No mp4/nifti pairs were found under %s.", root_dir)
        return

    for task in tasks:
        case_dir = task.case_dir
        scan_path = task.scan_path
        seg_path = task.seg_path

        seg_base = _strip_nii_suffix(seg_path)
        timepoint_dir = seg_path.parent
        results_dir = timepoint_dir / "Results"
        curve_csv = results_dir / f"{seg_base}_curve_quantification.csv"
        if curve_csv.exists() and not args.overwrite:
            LOGGER.info(
                "Skipping %s (curve quantification already exists). Use --overwrite to re-run.",
                results_dir,
            )
            continue

        try:
            row = process_case(
                case_dir=case_dir,
                scan_path=scan_path,
                seg_path=seg_path,
                results_dir=results_dir,
                frame_index=args.frame_index,
                use_gamma=args.use_gamma,
                gamma=args.gamma,
                scan_loader_kwargs=scan_loader_kwargs,
                seg_loader_kwargs=seg_loader_kwargs,
                paramap_settings=paramap_settings,
                export_raw_arrays=args.export_raw_arrays,
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Failed to process %s: %s", case_dir, exc)
            continue

        append_summary_row(root_dir, args.summary_name, row)


if __name__ == "__main__":
    main()
