#!/usr/bin/env python3
"""
Batch Canon BIN to NIfTI Converter

Process multiple Canon .bin files organized in patient/timepoint folders.
Automatically pairs CHI and Fund files and creates both RAW and normalized outputs.

Directory structure:
    raw_mc_ctdna/
    ├── p14/
    │   ├── pre/
    │   │   └── raw/
    │   │       ├── *_CHI.bin
    │   │       └── *_Fund.bin
    │   ├── wk6/
    │   └── wk12/
    └── p16/
        └── ...
"""

import struct
import numpy as np
import nibabel as nib
import os
from pathlib import Path
from typing import Tuple, Dict, List
import re


def extract_canon_bin(bin_path: str, output_nifti_path: str = None,
                     normalize_chi: bool = True, verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Extract Canon .bin file to NIfTI format.

    Args:
        bin_path: Path to Canon .bin file
        output_nifti_path: Optional path for NIfTI output
        normalize_chi: If True, normalize CHI (CEUS) data to 0-255
        verbose: Print extraction progress

    Returns:
        volume_data: 3D numpy array (axial, lateral, frames)
        metadata: Dictionary with header information
    """
    if verbose:
        print(f"  Reading: {Path(bin_path).name}")

    # Read header (20 bytes: 5 integers)
    with open(bin_path, 'rb') as f:
        hdr_info = struct.unpack('i', f.read(4))[0]
        axial_size = struct.unpack('i', f.read(4))[0]
        lateral_size = struct.unpack('i', f.read(4))[0]
        frame_num = struct.unpack('i', f.read(4))[0]
        data_type = struct.unpack('i', f.read(4))[0]

    # Determine data type: 1=CHI (float64), 2=Fund (uint16)
    data_type_str = "CHI (CEUS)" if data_type == 1 else "Fund (B-mode)"
    read_dtype = np.float64 if data_type == 1 else np.uint16

    if verbose:
        print(f"    Dimensions: {axial_size} x {lateral_size} x {frame_num} frames")
        print(f"    Type: {data_type_str}")

    # Read frame data
    num_per_frame = axial_size * lateral_size
    volume_data = np.zeros((axial_size, lateral_size, frame_num), dtype=np.float64)

    frames_read = 0
    with open(bin_path, 'rb') as f:
        f.seek(20)  # Skip header
        for i in range(frame_num):
            frame = np.fromfile(f, dtype=read_dtype, count=num_per_frame)
            if frame.size != num_per_frame:
                break

            # Convert to float64 if needed
            if data_type == 2:
                frame = frame.astype(np.float64)

            # Reshape to (axial, lateral)
            frame = frame.reshape((axial_size, lateral_size))
            volume_data[:, :, i] = frame
            frames_read = i + 1

    if verbose:
        print(f"    Raw range: {volume_data.min():.6f} to {volume_data.max():.6f}")

    # Apply standard rotation (GUI will handle final orientation)
    volume_data = np.rot90(volume_data, k=3, axes=(0, 1))
    volume_data = volume_data.transpose(1, 0, 2)

    # Normalize CHI data using percentile-based scaling
    if data_type == 1 and normalize_chi:
        p_low = np.percentile(volume_data, 20)
        p_high = np.percentile(volume_data, 95)

        if verbose:
            print(f"    Normalizing: p20={p_low:.6f}, p95={p_high:.6f}")

        volume_data = np.clip(volume_data, p_low, p_high)
        volume_data = ((volume_data - p_low) / (p_high - p_low) * 255)

    if verbose:
        print(f"    Extracted {frames_read} frames")

    # Save to NIfTI if path provided
    if output_nifti_path:
        affine = np.eye(4)
        nii = nib.Nifti1Image(volume_data, affine)
        nii.header['pixdim'] = [1.0, 0.0993, 0.0725, 0.1307, 1.0, 0.0, 0.0, 0.0]
        nib.save(nii, output_nifti_path)
        if verbose:
            print(f"    Saved: {Path(output_nifti_path).name}")

    metadata = {
        'shape': volume_data.shape,
        'data_type_str': data_type_str,
        'frames_read': frames_read
    }

    return volume_data, metadata


def extract_timestamp(filename: str) -> str:
    """
    Extract timestamp portion from Canon filename.

    Canon filenames typically look like: R20240116132619009RawLinear_CHI.bin
    This extracts the date/time portion: R20240116132619

    Args:
        filename: Name of the .bin file

    Returns:
        Timestamp string (or full filename if pattern not found)
    """
    # Match pattern like R20240116132619009
    match = re.match(r'(R\d{14})', filename)
    if match:
        return match.group(1)

    # Fallback: return everything before _CHI or _Fund
    return re.sub(r'_(CHI|Fund).*$', '', filename)


def find_best_fund_match(chi_file: Path, fund_files: List[Path],
                         time_threshold: int = 60) -> Path:
    """
    Find the best matching Fund file for a given CHI file.

    Uses timestamp similarity to match CHI and Fund files that were
    acquired close together in time.

    Args:
        chi_file: CHI file to match
        fund_files: List of available Fund files
        time_threshold: Maximum time difference in seconds (default: 60)

    Returns:
        Best matching Fund file, or None if no good match found
    """
    chi_timestamp = extract_timestamp(chi_file.name)

    # Try to extract time in seconds from timestamp (RYYYYMMDDHHMMSS)
    try:
        chi_time_str = chi_timestamp[9:15]  # HHMMSS portion
        chi_seconds = (int(chi_time_str[0:2]) * 3600 +
                      int(chi_time_str[2:4]) * 60 +
                      int(chi_time_str[4:6]))
    except (ValueError, IndexError):
        # If we can't parse time, fall back to string matching
        chi_seconds = None

    best_match = None
    best_diff = float('inf')

    for fund_file in fund_files:
        fund_timestamp = extract_timestamp(fund_file.name)

        # First check: date portion must match (first 9 chars: RYYYYMMDD)
        if chi_timestamp[:9] != fund_timestamp[:9]:
            continue

        # If we have valid timestamps, compare by time difference
        if chi_seconds is not None:
            try:
                fund_time_str = fund_timestamp[9:15]
                fund_seconds = (int(fund_time_str[0:2]) * 3600 +
                              int(fund_time_str[2:4]) * 60 +
                              int(fund_time_str[4:6]))

                time_diff = abs(chi_seconds - fund_seconds)

                if time_diff < best_diff and time_diff <= time_threshold:
                    best_diff = time_diff
                    best_match = fund_file
            except (ValueError, IndexError):
                continue
        else:
            # Fallback: use string similarity
            if fund_timestamp == chi_timestamp:
                return fund_file

    return best_match


def get_frame_count(bin_path: Path) -> int:
    """
    Quickly read frame count from Canon .bin file header.

    Args:
        bin_path: Path to .bin file

    Returns:
        Number of frames, or 0 if read fails
    """
    try:
        with open(bin_path, 'rb') as f:
            f.seek(8)  # Skip hdr_info and axial_size
            frame_num = struct.unpack('i', f.read(4))[0]
            return frame_num
    except Exception:
        return 0


def pair_chi_fund_files(raw_folder: Path, matching_strategy: str = "timestamp",
                       keep_only_max_frames: bool = False) -> List[Tuple[Path, Path, str]]:
    """
    Find and pair CHI and Fund .bin files in a raw folder.

    Pairing strategies:
    - "timestamp": Match by timestamp similarity (recommended)
    - "index": Sort alphabetically and match by position

    Args:
        raw_folder: Path to raw folder containing .bin files
        matching_strategy: Strategy to use for pairing ("timestamp" or "index")
        keep_only_max_frames: If True, only keep the pair with the most frames

    Returns:
        List of (chi_path, fund_path, match_quality) tuples
        match_quality: "exact", "close", or "index"
    """
    # Find all CHI and Fund files
    chi_files = sorted(raw_folder.glob("*CHI.bin"))
    fund_files = sorted(raw_folder.glob("*Fund.bin"))

    if len(chi_files) == 0 or len(fund_files) == 0:
        return []

    pairs = []

    if matching_strategy == "timestamp":
        # Smart timestamp-based matching
        used_fund_files = set()

        for chi_file in chi_files:
            best_match = find_best_fund_match(chi_file, fund_files)

            if best_match and best_match not in used_fund_files:
                # Determine match quality
                chi_ts = extract_timestamp(chi_file.name)
                fund_ts = extract_timestamp(best_match.name)

                if chi_ts == fund_ts:
                    quality = "exact"
                else:
                    quality = "close"

                pairs.append((chi_file, best_match, quality))
                used_fund_files.add(best_match)
            else:
                print(f"  WARNING: No good Fund match found for {chi_file.name}")

        if len(pairs) < len(chi_files):
            print(f"  WARNING: Only matched {len(pairs)}/{len(chi_files)} CHI files")

    else:  # Index-based matching
        if len(chi_files) != len(fund_files):
            print(f"  WARNING: Unequal number of CHI ({len(chi_files)}) and Fund ({len(fund_files)}) files")
            print(f"  Will pair first {min(len(chi_files), len(fund_files))} files")

        # Pair by index
        for chi_file, fund_file in zip(chi_files, fund_files):
            pairs.append((chi_file, fund_file, "index"))

    # Filter to keep only the pair with maximum frames
    if keep_only_max_frames and len(pairs) > 1:
        print(f"  Filtering {len(pairs)} pairs to keep only max frames...")

        max_frames = 0
        best_pair = None

        for chi_path, fund_path, quality in pairs:
            chi_frames = get_frame_count(chi_path)
            fund_frames = get_frame_count(fund_path)
            avg_frames = (chi_frames + fund_frames) / 2

            print(f"    {chi_path.name}: {chi_frames} frames")
            print(f"    {fund_path.name}: {fund_frames} frames")

            if avg_frames > max_frames:
                max_frames = avg_frames
                best_pair = (chi_path, fund_path, quality)

        if best_pair:
            print(f"  Selected pair with {int(max_frames)} avg frames")
            pairs = [best_pair]
        else:
            print(f"  WARNING: Could not determine best pair, keeping all")

    return pairs


def extract_base_name(chi_path: Path, fund_path: Path) -> str:
    """
    Extract a common base name from CHI and Fund file pair.

    Uses the date/timestamp portion before the _CHI or _Fund suffix.
    Falls back to using just the CHI filename.

    Args:
        chi_path: Path to CHI .bin file
        fund_path: Path to Fund .bin file

    Returns:
        Base name for output files
    """
    chi_name = chi_path.stem  # Remove .bin extension
    fund_name = fund_path.stem

    # Try to extract common prefix (everything before _CHI or _Fund)
    chi_base = re.sub(r'_CHI$', '', chi_name)
    fund_base = re.sub(r'_Fund$', '', fund_name)

    # Use CHI base if they match closely, otherwise use full CHI name
    if chi_base[:10] == fund_base[:10]:  # At least date portion matches
        return chi_base
    else:
        return chi_name.replace('_CHI', '')


def process_timepoint(timepoint_folder: Path, output_folder_name: str = "nifti_output",
                      matching_strategy: str = "timestamp", skip_existing: bool = True,
                      keep_only_max_frames: bool = False, verbose: bool = True) -> Dict[str, int]:
    """
    Process all .bin files in a timepoint folder (e.g., pre, wk6, wk12).

    Args:
        timepoint_folder: Path to timepoint folder (e.g., .../p14/wk12/)
        output_folder_name: Name of output folder to create in timepoint
        matching_strategy: How to pair CHI/Fund files ("timestamp" or "index")
        skip_existing: If True, skip pairs where output already exists
        keep_only_max_frames: If True, only process the pair with most frames
        verbose: Print processing details

    Returns:
        Dictionary with processing statistics
    """
    raw_folder = timepoint_folder / "raw"

    if not raw_folder.exists():
        if verbose:
            print(f"  No 'raw' folder found in {timepoint_folder}")
        return {"processed": 0, "skipped": 0, "failed": 0}

    # Find CHI/Fund pairs
    pairs = pair_chi_fund_files(raw_folder, matching_strategy=matching_strategy,
                                keep_only_max_frames=keep_only_max_frames)

    if len(pairs) == 0:
        if verbose:
            print(f"  No CHI/Fund pairs found in {raw_folder}")
        return {"processed": 0, "skipped": 0, "failed": 0}

    # Create output directory
    output_dir = timepoint_folder / output_folder_name
    output_dir.mkdir(exist_ok=True)

    if verbose:
        print(f"\n  Found {len(pairs)} CHI/Fund pair(s) in {raw_folder.name}")
        print(f"  Output directory: {output_dir}")

    stats = {"processed": 0, "skipped": 0, "failed": 0}

    # Process each pair
    for idx, (chi_path, fund_path, match_quality) in enumerate(pairs, 1):
        base_name = extract_base_name(chi_path, fund_path)

        # Check if outputs already exist
        chi_raw_output = output_dir / f"{base_name}_CHI_RAW.nii"
        chi_norm_output = output_dir / f"{base_name}_CHI_normalized_GUI.nii"
        fund_raw_output = output_dir / f"{base_name}_Fund_RAW.nii"
        fund_norm_output = output_dir / f"{base_name}_Fund_normalized_GUI.nii"

        all_outputs_exist = all([
            chi_raw_output.exists(),
            chi_norm_output.exists(),
            fund_raw_output.exists(),
            fund_norm_output.exists()
        ])

        if skip_existing and all_outputs_exist:
            if verbose:
                print(f"\n  [{idx}/{len(pairs)}] SKIPPING (already exists): {base_name}")
                print(f"    Match: {match_quality}")
            stats["skipped"] += 1
            continue

        if verbose:
            print(f"\n  [{idx}/{len(pairs)}] Processing pair:")
            print(f"    CHI:  {chi_path.name}")
            print(f"    Fund: {fund_path.name}")
            print(f"    Base: {base_name}")
            print(f"    Match: {match_quality}")

        try:
            # Extract CHI - RAW version
            chi_data, _ = extract_canon_bin(chi_path, str(chi_raw_output),
                                            normalize_chi=False, verbose=verbose)

            # Extract CHI - Normalized version for GUI
            chi_p20 = np.percentile(chi_data, 20)
            chi_p95 = np.percentile(chi_data, 95)
            chi_normalized = np.clip(chi_data, chi_p20, chi_p95)
            chi_normalized = ((chi_normalized - chi_p20) / (chi_p95 - chi_p20) * 255)

            nii = nib.Nifti1Image(chi_normalized, np.eye(4))
            nii.header['pixdim'] = [1.0, 0.0993, 0.0725, 0.1307, 1.0, 0.0, 0.0, 0.0]
            nib.save(nii, str(chi_norm_output))
            if verbose:
                print(f"    Saved: {chi_norm_output.name}")

            # Extract Fund - RAW version
            fund_data, _ = extract_canon_bin(fund_path, str(fund_raw_output),
                                             normalize_chi=False, verbose=verbose)

            # Extract Fund - Normalized version for GUI
            fund_normalized = ((fund_data - fund_data.min()) /
                              (fund_data.max() - fund_data.min()) * 255)

            nii = nib.Nifti1Image(fund_normalized, np.eye(4))
            nii.header['pixdim'] = [1.0, 0.0993, 0.0725, 0.1307, 1.0, 0.0, 0.0, 0.0]
            nib.save(nii, str(fund_norm_output))
            if verbose:
                print(f"    Saved: {fund_norm_output.name}")

            stats["processed"] += 1

        except Exception as e:
            print(f"    ERROR: Failed to process pair: {e}")
            stats["failed"] += 1

    return stats


def batch_process(root_folder: str, output_folder_name: str = "nifti_output",
                 patient_pattern: str = "p*", timepoint_pattern: str = "*",
                 matching_strategy: str = "timestamp", skip_existing: bool = True,
                 keep_only_max_frames: bool = False, verbose: bool = True):
    """
    Batch process Canon .bin files organized in patient/timepoint structure.

    Args:
        root_folder: Root directory (e.g., raw_mc_ctdna)
        output_folder_name: Name of output folder to create in each timepoint
        patient_pattern: Glob pattern for patient folders (default: "p*")
        timepoint_pattern: Glob pattern for timepoint folders (default: "*" for all)
        matching_strategy: How to pair CHI/Fund files ("timestamp" or "index")
        skip_existing: If True, skip pairs where output already exists
        keep_only_max_frames: If True, only process the pair with most frames per timepoint
        verbose: Print processing details
    """
    root_path = Path(root_folder)

    if not root_path.exists():
        print(f"ERROR: Root folder does not exist: {root_folder}")
        return

    print(f"=" * 80)
    print(f"Batch Canon BIN Converter")
    print(f"=" * 80)
    print(f"Root folder: {root_path}")
    print(f"Patient pattern: {patient_pattern}")
    print(f"Timepoint pattern: {timepoint_pattern}")
    print(f"Output folder name: {output_folder_name}")
    print(f"Matching strategy: {matching_strategy}")
    print(f"Skip existing: {skip_existing}")
    print(f"Keep only max frames: {keep_only_max_frames}")
    print(f"=" * 80)

    # Find all patient folders
    patient_folders = sorted(root_path.glob(patient_pattern))

    if len(patient_folders) == 0:
        print(f"\nNo patient folders found matching pattern '{patient_pattern}'")
        return

    print(f"\nFound {len(patient_folders)} patient folder(s)")

    # Overall statistics
    total_stats = {"processed": 0, "skipped": 0, "failed": 0}

    # Process each patient
    for patient_folder in patient_folders:
        if not patient_folder.is_dir():
            continue

        print(f"\n{'=' * 80}")
        print(f"Patient: {patient_folder.name}")
        print(f"{'=' * 80}")

        # Find timepoint folders
        timepoint_folders = sorted(patient_folder.glob(timepoint_pattern))
        timepoint_folders = [tp for tp in timepoint_folders if tp.is_dir()]

        if len(timepoint_folders) == 0:
            print(f"  No timepoint folders found")
            continue

        # Process each timepoint
        for timepoint_folder in timepoint_folders:
            print(f"\nTimepoint: {timepoint_folder.name}")
            print("-" * 40)

            stats = process_timepoint(timepoint_folder,
                                     output_folder_name,
                                     matching_strategy,
                                     skip_existing,
                                     keep_only_max_frames,
                                     verbose)

            # Accumulate statistics
            for key in total_stats:
                total_stats[key] += stats[key]

    print(f"\n{'=' * 80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Processed: {total_stats['processed']} pair(s)")
    print(f"Skipped:   {total_stats['skipped']} pair(s)")
    print(f"Failed:    {total_stats['failed']} pair(s)")
    print(f"Total:     {sum(total_stats.values())} pair(s)")
    print(f"={'=' * 80}")


if __name__ == "__main__":
    # === CONFIGURATION ===
    # Edit these settings to match your data structure

    ROOT_FOLDER = "/Users/samantha/Desktop/ultrasound lab stuff/raw_mc_ctdna"  # Update this path
    OUTPUT_FOLDER_NAME = "converted_nifti"     # Folder name to create in each timepoint
    PATIENT_PATTERN = "p*"                  # Match all patient folders starting with 'p'
    TIMEPOINT_PATTERN = "*"                 # Match all timepoint folders (pre, wk6, wk12, etc.)

    # Pairing strategy: "timestamp" (recommended) or "index"
    # - "timestamp": Matches CHI/Fund by acquisition time (handles unequal counts)
    # - "index": Matches by alphabetical position (requires equal counts)
    MATCHING_STRATEGY = "timestamp"

    # Skip files that already exist (useful for re-running after interruption)
    SKIP_EXISTING = True

    # Keep only the pair with the highest frame count per timepoint
    # Useful when you have multiple acquisitions and want the longest/best one
    KEEP_ONLY_MAX_FRAMES = True

    VERBOSE = True                          # Print detailed progress

    # Example: Process only specific timepoints
    # TIMEPOINT_PATTERN = "wk*"  # Only week timepoints
    # TIMEPOINT_PATTERN = "wk12"  # Only week 12

    # Example: Process specific patients
    # PATIENT_PATTERN = "p14"  # Only patient 14

    # Example: Force reprocessing all files
    # SKIP_EXISTING = False

    # Run batch processing
    batch_process(
        root_folder=ROOT_FOLDER,
        output_folder_name=OUTPUT_FOLDER_NAME,
        patient_pattern=PATIENT_PATTERN,
        timepoint_pattern=TIMEPOINT_PATTERN,
        matching_strategy=MATCHING_STRATEGY,
        skip_existing=SKIP_EXISTING,
        keep_only_max_frames=KEEP_ONLY_MAX_FRAMES,
        verbose=VERBOSE
    )
