#!/usr/bin/env python3
"""
Canon BIN to NIfTI Batch Converter with Session Combining

Converts multiple Canon .bin files from a single session into combined NIFTI files.
- Combines all CHI files into one NIFTI (sorted by timestamp)
- Combines all Fund files into one NIFTI (sorted by timestamp)
- Creates both RAW and normalized versions
"""

import struct
import numpy as np
import nibabel as nib
import os
import re
import csv
from pathlib import Path
from typing import Tuple, Dict, List
from datetime import datetime


def extract_timestamp_from_filename(filename: str) -> str:
    """Extract timestamp from Canon filename (e.g., R20240325153313721RawLinear_CHI.bin -> 20240325153313721)"""
    match = re.search(r'R(\d+)RawLinear', filename)
    if match:
        return match.group(1)
    return filename  # Fallback to filename if pattern not found


def parse_canon_timestamp(timestamp_str: str) -> float:
    """
    Parse Canon timestamp string to seconds since epoch.

    Format: R20240325153313721 -> YYYYMMDDHHMMSS + milliseconds

    Args:
        timestamp_str: Timestamp string (e.g., '20240325153313721')

    Returns:
        Seconds since epoch (float)
    """
    try:
        # Parse: YYYYMMDDHHMMSS + milliseconds
        year = int(timestamp_str[0:4])
        month = int(timestamp_str[4:6])
        day = int(timestamp_str[6:8])
        hour = int(timestamp_str[8:10])
        minute = int(timestamp_str[10:12])
        second = int(timestamp_str[12:14])
        # Remaining digits are subseconds (milliseconds)
        subsec = int(timestamp_str[14:]) if len(timestamp_str) > 14 else 0

        dt = datetime(year, month, day, hour, minute, second)
        epoch_seconds = dt.timestamp()

        # Add subseconds (assuming milliseconds, adjust if needed)
        subsec_fraction = subsec / 1000.0

        return epoch_seconds + subsec_fraction
    except (ValueError, IndexError):
        return 0.0


def extract_canon_bin(bin_path: str, verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Extract Canon .bin file and return the volume data.

    Args:
        bin_path: Path to Canon .bin file
        verbose: Print extraction progress

    Returns:
        volume_data: 3D numpy array (axial, lateral, frames)
        metadata: Dictionary with header information
    """
    if verbose:
        print(f"  Reading: {Path(bin_path).name}")

    # Read header (20 bytes: 5 integers)
    try:
        with open(bin_path, 'rb') as f:
            hdr_info = struct.unpack('i', f.read(4))[0]
            axial_size = struct.unpack('i', f.read(4))[0]
            lateral_size = struct.unpack('i', f.read(4))[0]
            frame_num = struct.unpack('i', f.read(4))[0]
            data_type = struct.unpack('i', f.read(4))[0]
    except struct.error as e:
        if verbose:
            print(f"    ⚠️  ERROR: Corrupted or incomplete file - {e}")
        return None, None

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
        print(f"    Frames: {frames_read}")

    # Apply standard rotation (GUI will handle final orientation)
    volume_data = np.rot90(volume_data, k=3, axes=(0, 1))
    volume_data = volume_data.transpose(1, 0, 2)

    metadata = {
        'shape': volume_data.shape,
        'data_type_str': data_type_str,
        'data_type': data_type,
        'frames_read': frames_read
    }

    return volume_data, metadata


def combine_and_convert_session(raw_dir: str, output_dir: str, session_name: str = None,
                                fps: float = 8.5, verbose: bool = True):
    """
    Combine all CHI and Fund .bin files from a session into single NIFTI files.

    Args:
        raw_dir: Directory containing .bin files (e.g., /path/to/p14/wk24/raw)
        output_dir: Directory to save combined NIFTI files
        session_name: Optional custom name prefix (default: auto-detect from path)
        fps: Frames per second for timing calculations (default: 8.5)
        verbose: Print progress information
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Auto-detect session name from path if not provided (e.g., p14_wk24)
    if session_name is None:
        # Extract patient and visit from path (e.g., .../p14/wk24/raw -> p14_wk24)
        parts = raw_path.parts
        if len(parts) >= 2:
            session_name = f"{parts[-2]}_{parts[-3]}" if parts[-1] == 'raw' else parts[-1]
        else:
            session_name = "session"

    if verbose:
        print(f"\n{'='*80}")
        print(f"Processing session: {session_name}")
        print(f"Input directory: {raw_dir}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}\n")

    # Find all .bin files
    all_chi_files = sorted(raw_path.glob("*CHI.bin"), key=lambda x: extract_timestamp_from_filename(x.name))
    all_fund_files = sorted(raw_path.glob("*Fund.bin"), key=lambda x: extract_timestamp_from_filename(x.name))

    if verbose:
        print(f"Found {len(all_chi_files)} CHI files and {len(all_fund_files)} Fund files")

    if len(all_chi_files) == 0 and len(all_fund_files) == 0:
        print("⚠️  No .bin files found in directory!")
        return

    # Pair CHI and Fund files by timestamp proximity to ensure matched frames
    chi_files = []
    fund_files = []
    used_fund_files = set()

    for chi_file in all_chi_files:
        chi_timestamp_str = extract_timestamp_from_filename(chi_file.name)
        chi_timestamp = parse_canon_timestamp(chi_timestamp_str)

        # Find matching Fund file with closest timestamp (within 60 seconds)
        best_match = None
        best_diff = float('inf')

        for fund_file in all_fund_files:
            if fund_file in used_fund_files:
                continue

            fund_timestamp_str = extract_timestamp_from_filename(fund_file.name)
            fund_timestamp = parse_canon_timestamp(fund_timestamp_str)

            # Check if same date (YYYYMMDD)
            if chi_timestamp_str[:8] != fund_timestamp_str[:8]:
                continue

            time_diff = abs(chi_timestamp - fund_timestamp)

            if time_diff < best_diff and time_diff <= 60:  # Within 60 seconds
                best_diff = time_diff
                best_match = fund_file

        if best_match:
            chi_files.append(chi_file)
            fund_files.append(best_match)
            used_fund_files.add(best_match)
        else:
            if verbose:
                print(f"  ⚠️  No matching Fund file for CHI: {chi_file.name}")

    if verbose:
        print(f"Matched {len(chi_files)} CHI/Fund pairs (within 60 sec)")

    if len(chi_files) == 0:
        print("⚠️  No matched CHI/Fund pairs found!")
        return

    # Process CHI files
    if len(chi_files) > 0:
        if verbose:
            print(f"\n--- Processing CHI files ({len(chi_files)} files) ---")
            print(f"  Using FPS: {fps}")

        chi_volumes = []
        chi_file_info = []  # Track file info for timing

        for chi_file in chi_files:
            volume, meta = extract_canon_bin(str(chi_file), verbose=verbose)
            if volume is None:
                if verbose:
                    print(f"    Skipping corrupted file")
                continue
            chi_volumes.append(volume)
            chi_file_info.append({
                'filename': chi_file.name,
                'frames': meta['frames_read'],
                'timestamp': extract_timestamp_from_filename(chi_file.name)
            })

        # Concatenate along frame axis (axis 2)
        combined_chi = np.concatenate(chi_volumes, axis=2)
        if verbose:
            print(f"\n  Combined CHI shape: {combined_chi.shape}")
            print(f"  Total frames: {combined_chi.shape[2]}")

        # Generate timing file
        timing_csv_path = output_path / f"{session_name}_CHI_timing.csv"
        with open(timing_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'source_file', 'time_seconds', 'acquisition_timestamp'])

            cumulative_time = 0.0
            frame_idx = 0
            first_timestamp = parse_canon_timestamp(chi_file_info[0]['timestamp'])

            for file_info in chi_file_info:
                file_timestamp = parse_canon_timestamp(file_info['timestamp'])
                # Gap from first file
                gap_from_start = file_timestamp - first_timestamp if file_timestamp > 0 else 0.0

                for local_frame in range(file_info['frames']):
                    # Time within this file
                    frame_time_in_file = local_frame / fps
                    # Total time = gap from start + time within file
                    total_time = gap_from_start + frame_time_in_file

                    writer.writerow([
                        frame_idx,
                        file_info['filename'],
                        f"{total_time:.3f}",
                        file_info['timestamp']
                    ])
                    frame_idx += 1

        if verbose:
            print(f"  ✓ Saved timing: {timing_csv_path.name}")

        # Save RAW version (float64, original values)
        chi_raw_path = output_path / f"{session_name}_CHI_RAW.nii"
        affine = np.eye(4)
        nii = nib.Nifti1Image(combined_chi, affine)
        nii.header['pixdim'] = [1.0, 0.0993, 0.0725, 0.1307, 1.0, 0.0, 0.0, 0.0]
        nib.save(nii, str(chi_raw_path))
        if verbose:
            print(f"  ✓ Saved RAW: {chi_raw_path.name}")

        # Save normalized version for GUI (uint8, 0-255, percentile-based)
        chi_p20 = np.percentile(combined_chi, 20)
        chi_p95 = np.percentile(combined_chi, 95)
        chi_normalized = np.clip(combined_chi, chi_p20, chi_p95)
        chi_normalized = ((chi_normalized - chi_p20) / (chi_p95 - chi_p20) * 255).astype(np.uint8)

        chi_norm_path = output_path / f"{session_name}_CHI_normalized_GUI.nii"
        nii_norm = nib.Nifti1Image(chi_normalized, affine)
        nii_norm.header['pixdim'] = [1.0, 0.0993, 0.0725, 0.1307, 1.0, 0.0, 0.0, 0.0]
        nib.save(nii_norm, str(chi_norm_path))
        if verbose:
            print(f"  ✓ Saved normalized: {chi_norm_path.name}")
            print(f"    (p20={chi_p20:.6f}, p95={chi_p95:.6f} -> 0-255)")

    # Process Fund files
    if len(fund_files) > 0:
        if verbose:
            print(f"\n--- Processing Fund files ({len(fund_files)} files) ---")
            print(f"  Using FPS: {fps}")

        fund_volumes = []
        fund_file_info = []  # Track file info for timing

        for fund_file in fund_files:
            volume, meta = extract_canon_bin(str(fund_file), verbose=verbose)
            if volume is None:
                if verbose:
                    print(f"    Skipping corrupted file")
                continue
            fund_volumes.append(volume)
            fund_file_info.append({
                'filename': fund_file.name,
                'frames': meta['frames_read'],
                'timestamp': extract_timestamp_from_filename(fund_file.name)
            })

        # Concatenate along frame axis (axis 2)
        combined_fund = np.concatenate(fund_volumes, axis=2)
        if verbose:
            print(f"\n  Combined Fund shape: {combined_fund.shape}")
            print(f"  Total frames: {combined_fund.shape[2]}")

        # Generate timing file
        timing_csv_path = output_path / f"{session_name}_Fund_timing.csv"
        with open(timing_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'source_file', 'time_seconds', 'acquisition_timestamp'])

            cumulative_time = 0.0
            frame_idx = 0
            first_timestamp = parse_canon_timestamp(fund_file_info[0]['timestamp'])

            for file_info in fund_file_info:
                file_timestamp = parse_canon_timestamp(file_info['timestamp'])
                # Gap from first file
                gap_from_start = file_timestamp - first_timestamp if file_timestamp > 0 else 0.0

                for local_frame in range(file_info['frames']):
                    # Time within this file
                    frame_time_in_file = local_frame / fps
                    # Total time = gap from start + time within file
                    total_time = gap_from_start + frame_time_in_file

                    writer.writerow([
                        frame_idx,
                        file_info['filename'],
                        f"{total_time:.3f}",
                        file_info['timestamp']
                    ])
                    frame_idx += 1

        if verbose:
            print(f"  ✓ Saved timing: {timing_csv_path.name}")

        # Save RAW version (float64, original values)
        fund_raw_path = output_path / f"{session_name}_Fund_RAW.nii"
        affine = np.eye(4)
        nii = nib.Nifti1Image(combined_fund, affine)
        nii.header['pixdim'] = [1.0, 0.0993, 0.0725, 0.1307, 1.0, 0.0, 0.0, 0.0]
        nib.save(nii, str(fund_raw_path))
        if verbose:
            print(f"  ✓ Saved RAW: {fund_raw_path.name}")

        # Save normalized version for GUI (uint8, 0-255, min-max scaling)
        fund_normalized = ((combined_fund - combined_fund.min()) / (combined_fund.max() - combined_fund.min()) * 255).astype(np.uint8)

        fund_norm_path = output_path / f"{session_name}_Fund_normalized_GUI.nii"
        nii_norm = nib.Nifti1Image(fund_normalized, affine)
        nii_norm.header['pixdim'] = [1.0, 0.0993, 0.0725, 0.1307, 1.0, 0.0, 0.0, 0.0]
        nib.save(nii_norm, str(fund_norm_path))
        if verbose:
            print(f"  ✓ Saved normalized: {fund_norm_path.name}")
            print(f"    ({combined_fund.min():.2f}-{combined_fund.max():.2f} -> 0-255)")

    if verbose:
        print(f"\n{'='*80}")
        print(f"✓ Session {session_name} complete!")
        print(f"{'='*80}\n")


def batch_process_patient(patient_dir: str, fps_map: Dict = None, default_fps: float = 8.5,
                         output_base_dir: str = None, verbose: bool = True):
    """
    Process all visits for a patient.

    Args:
        patient_dir: Patient directory containing visit folders (e.g., /path/to/p14)
        fps_map: Dictionary mapping patient->visit->fps (e.g., {"p14": {"wk24": 8}})
        default_fps: Default FPS if not found in fps_map
        output_base_dir: Base output directory (default: same as patient_dir)
        verbose: Print progress information
    """
    patient_path = Path(patient_dir)

    if output_base_dir is None:
        output_base_dir = str(patient_path)

    patient_name = patient_path.name

    if verbose:
        print(f"\n{'#'*80}")
        print(f"# BATCH PROCESSING PATIENT: {patient_name}")
        print(f"{'#'*80}")

    # Find all visit folders (folders containing a 'raw' subfolder)
    visit_folders = [f for f in patient_path.iterdir() if f.is_dir() and (f / 'raw').exists()]

    if len(visit_folders) == 0:
        print(f"⚠️  No visit folders with 'raw' subdirectories found in {patient_dir}")
        return

    if verbose:
        print(f"\nFound {len(visit_folders)} visit folders: {[f.name for f in visit_folders]}")

    for visit_folder in sorted(visit_folders):
        visit_name = visit_folder.name
        raw_folder = visit_folder / 'raw'
        output_folder = visit_folder / 'combined_nifti'

        session_name = f"{patient_name}_{visit_name}"

        # Get FPS for this visit
        fps = default_fps
        if fps_map and patient_name in fps_map and visit_name in fps_map[patient_name]:
            fps = fps_map[patient_name][visit_name]

        combine_and_convert_session(
            raw_dir=str(raw_folder),
            output_dir=str(output_folder),
            session_name=session_name,
            fps=fps,
            verbose=verbose
        )

    if verbose:
        print(f"\n{'#'*80}")
        print(f"# ✓ PATIENT {patient_name} COMPLETE - {len(visit_folders)} visits processed")
        print(f"{'#'*80}\n")


def batch_process_all_patients(root_folder: str, fps_map: Dict = None, default_fps: float = 8.5,
                              patient_pattern: str = "p*", verbose: bool = True):
    """
    Process all patients in a root folder.

    Args:
        root_folder: Root directory containing patient folders
        fps_map: Dictionary mapping patient->visit->fps
        default_fps: Default FPS if not found in fps_map
        patient_pattern: Glob pattern for patient folders
        verbose: Print progress information
    """
    root_path = Path(root_folder)

    if not root_path.exists():
        print(f"ERROR: Root folder does not exist: {root_folder}")
        return

    # Find all patient folders
    patient_folders = sorted(root_path.glob(patient_pattern))

    if len(patient_folders) == 0:
        print(f"No patient folders found matching pattern '{patient_pattern}' in {root_folder}")
        return

    if verbose:
        print(f"\n{'#'*80}")
        print(f"# BATCH PROCESSING ALL PATIENTS")
        print(f"# Root: {root_folder}")
        print(f"# Found {len(patient_folders)} patients: {[p.name for p in patient_folders]}")
        print(f"# Default FPS: {default_fps}")
        print(f"{'#'*80}\n")

    for patient_folder in patient_folders:
        if patient_folder.is_dir():
            batch_process_patient(str(patient_folder), fps_map=fps_map,
                                default_fps=default_fps, verbose=verbose)

    if verbose:
        print(f"\n{'#'*80}")
        print(f"# ✓ ALL PATIENTS COMPLETE")
        print(f"{'#'*80}\n")


if __name__ == "__main__":
    # === CONFIGURATION ===
    # Edit these settings to match your data structure

    # Path to folder containing patient folders (p5, p8, p14, p16, etc.)
    ROOT_FOLDER = "/Users/samantha/Desktop/ultrasound lab stuff/raw_ctdna"

    # Pattern to match patient folders (default: "p*" matches all patients)
    PATIENT_PATTERN = "p*"

    # Default FPS for all scans (used if not specified in FPS_MAP)
    DEFAULT_FPS = 8.5

    # FPS configuration for specific patient/visit combinations
    # Edit this to match your actual frame rates from DICOM data
    # Format: {"patient_id": {"visit_id": fps_value}}
    FPS_MAP = {
        "p5": {
            "pre": 8,
            "wk6": 8,
            "wk12": 8,
        },
        "p8": {
            "pre": 9,
            "wk6": 8,
            "wk12": 8,
            "wk24": 8,
        },
        "p14": {
            "pre": 8,
            "wk6": 8,
            "wk12": 9,
            "wk24": 8,
        },
        "p16": {
            "pre": 8,
            "wk6": 9,
            "wk12": 8,
            "wk24": 9,
        },
    }

    # Print detailed progress
    VERBOSE = True

    # === EXAMPLES ===
    # Process only specific patients:
    # PATIENT_PATTERN = "p14"  # Only patient 14
    # PATIENT_PATTERN = "p{14,16}"  # Only patients 14 and 16

    # Example FPS configurations:
    # FPS_MAP = {
    #     "p14": {"wk24": 9.0, "wk12": 8.0},  # p14 has different FPS
    #     "p16": {"pre": 8.5}  # p16 pre has specific FPS, others use default
    # }

    # === RUN BATCH PROCESSING ===
    batch_process_all_patients(
        root_folder=ROOT_FOLDER,
        fps_map=FPS_MAP,
        default_fps=DEFAULT_FPS,
        patient_pattern=PATIENT_PATTERN,
        verbose=VERBOSE
    )
