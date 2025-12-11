"""
Apply motion compensation displacement vectors from small ROI to large ROI.

This script extracts displacement vectors from a small motion-corrected ROI
(using B-mode data for tracking) and applies them to a larger ROI that covers
the full lesion.

Usage:
    python apply_mc_to_large_roi.py \
        --bmode /path/to/bmode.nii \
        --small-roi /path/to/small_mc_roi.nii.gz \
        --large-roi /path/to/large_roi.nii.gz \
        --output /path/to/output_mc_roi.nii.gz
"""

import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import shift as scipy_shift
from pathlib import Path
import sys

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.seg_preprocessing.motion_compensation import BoundingBox3D, MotionCompensation3D


def load_nifti(path):
    """Load NIfTI file and return data array."""
    return nib.load(path).get_fdata()


def save_nifti(data, path, reference_nifti=None):
    """Save data as NIfTI file."""
    if reference_nifti is not None:
        # Use reference header/affine
        ref_img = nib.load(reference_nifti)
        img = nib.Nifti1Image(data.astype(np.uint8), ref_img.affine, ref_img.header)
    else:
        img = nib.Nifti1Image(data.astype(np.uint8), np.eye(4))
    nib.save(img, path)


def apply_mc_to_large_roi(
    bmode_path: str,
    small_roi_path: str,
    large_roi_path: str,
    output_path: str,
    reference_frame: int = 0,
    search_range: tuple = (0, 10, 10),
    correlation_threshold: float = 0.3,
    padding: int = 5,
    verbose: bool = True
):
    """
    Apply motion compensation from small ROI tracking to large ROI.

    Args:
        bmode_path: Path to B-mode NIfTI file
        small_roi_path: Path to small motion-corrected ROI
        large_roi_path: Path to large (full lesion) ROI
        output_path: Path to save motion-compensated large ROI
        reference_frame: Reference frame for motion tracking
        search_range: (z, y, x) search range for motion tracking
        correlation_threshold: Minimum correlation to accept a shift
        padding: Padding around small ROI for tracking
        verbose: Print progress information
    """

    if verbose:
        print("=" * 70)
        print("Motion Compensation: Apply Small ROI Displacement to Large ROI")
        print("=" * 70)

    # Load data
    if verbose:
        print("\n1. Loading data...")
    bmode_data = load_nifti(bmode_path)
    small_roi_seg = load_nifti(small_roi_path)
    large_roi_seg = load_nifti(large_roi_path)

    if verbose:
        print(f"   B-mode shape: {bmode_data.shape}")
        print(f"   Small ROI shape: {small_roi_seg.shape}")
        print(f"   Large ROI shape: {large_roi_seg.shape}")

    # Reshape B-mode to (T, Z, Y, X) format
    if verbose:
        print("\n2. Preparing B-mode data...")
    if bmode_data.shape[2] > bmode_data.shape[0]:  # Time is on last axis
        bmode_data = bmode_data.transpose(2, 0, 1)  # (T, Y, X)
        bmode_data = bmode_data[:, np.newaxis, :, :]  # (T, 1, Y, X)

    if verbose:
        print(f"   B-mode reshaped to: {bmode_data.shape}")

    # Get bounding box from small ROI
    if verbose:
        print("\n3. Extracting tracking region from small ROI...")
    small_bbox = BoundingBox3D.from_mask(
        small_roi_seg[:, :, reference_frame][np.newaxis, :, :],
        padding=padding
    )

    if verbose:
        print(f"   Bounding box: z=[{small_bbox.z_min}, {small_bbox.z_max}), "
              f"y=[{small_bbox.y_min}, {small_bbox.y_max}), "
              f"x=[{small_bbox.x_min}, {small_bbox.x_max})")
        print(f"   Center: {small_bbox.center}")

    # Initialize motion compensation
    if verbose:
        print("\n4. Computing displacement vectors from B-mode...")
    mc = MotionCompensation3D(
        bmode_volume=bmode_data,
        reference_frame=reference_frame,
        search_range=search_range,
        correlation_threshold=correlation_threshold,
        use_reference_bbox=True
    )

    # Compute displacement vectors
    shifts = mc.compute_shifts(small_bbox, verbose=verbose)

    if verbose:
        print(f"\n   Computed {shifts.shape[0]} displacement vectors")
        print(f"   Displacement statistics:")
        print(f"     Max Y displacement: {np.max(np.abs(shifts[:, 1])):.2f} pixels")
        print(f"     Max X displacement: {np.max(np.abs(shifts[:, 2])):.2f} pixels")
        print(f"     Mean displacement magnitude: {np.mean(np.linalg.norm(shifts[:, 1:], axis=1)):.2f} pixels")

    # Quality metrics
    metrics = mc.get_quality_metrics()
    if verbose:
        print(f"\n   Motion tracking quality:")
        print(f"     Mean correlation: {metrics['mean_correlation']:.3f}")
        print(f"     Min correlation: {metrics['min_correlation']:.3f}")
        print(f"     Frames below threshold: {int(metrics['frames_below_threshold'])}")

    # Apply shifts to large ROI
    if verbose:
        print("\n5. Applying displacement vectors to large ROI...")
    n_frames = large_roi_seg.shape[2]
    large_roi_compensated = np.zeros_like(large_roi_seg)

    for t in range(n_frames):
        # Use only Y and X shifts (ignore Z for 2D data)
        shift_vector = -shifts[t, 1:]  # (dy, dx) - negative to align back to reference
        large_roi_compensated[:, :, t] = scipy_shift(
            large_roi_seg[:, :, t],
            shift_vector,
            order=0,  # Nearest neighbor for binary mask
            mode='constant',
            cval=0
        )

    if verbose:
        print(f"   Motion-compensated large ROI shape: {large_roi_compensated.shape}")

    # Save result
    if verbose:
        print(f"\n6. Saving motion-compensated large ROI...")
    save_nifti(large_roi_compensated, output_path, reference_nifti=large_roi_path)

    if verbose:
        print(f"   Saved to: {output_path}")
        print("\n" + "=" * 70)
        print("Motion compensation completed successfully!")
        print("=" * 70)

    return large_roi_compensated, shifts, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Apply motion compensation displacement vectors from small ROI to large ROI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python apply_mc_to_large_roi.py \\
        --bmode data/bmode.nii \\
        --small-roi data/small_mc_roi.nii.gz \\
        --large-roi data/large_roi.nii.gz \\
        --output data/large_roi_mc.nii.gz

    # With custom parameters
    python apply_mc_to_large_roi.py \\
        --bmode data/bmode.nii \\
        --small-roi data/small_mc_roi.nii.gz \\
        --large-roi data/large_roi.nii.gz \\
        --output data/large_roi_mc.nii.gz \\
        --reference-frame 10 \\
        --search-range 0 15 15 \\
        --correlation-threshold 0.4
        """
    )

    # Required arguments
    parser.add_argument('--bmode', required=True, help='Path to B-mode NIfTI file')
    parser.add_argument('--small-roi', required=True, help='Path to small motion-corrected ROI')
    parser.add_argument('--large-roi', required=True, help='Path to large (full lesion) ROI')
    parser.add_argument('--output', required=True, help='Path to save motion-compensated large ROI')

    # Optional arguments
    parser.add_argument('--reference-frame', type=int, default=0,
                        help='Reference frame for motion tracking (default: 0)')
    parser.add_argument('--search-range', nargs=3, type=int, default=[0, 10, 10],
                        metavar=('Z', 'Y', 'X'),
                        help='Search range for motion tracking in pixels (default: 0 10 10)')
    parser.add_argument('--correlation-threshold', type=float, default=0.3,
                        help='Minimum correlation to accept a shift (default: 0.3)')
    parser.add_argument('--padding', type=int, default=5,
                        help='Padding around small ROI for tracking (default: 5)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Run motion compensation
    apply_mc_to_large_roi(
        bmode_path=args.bmode,
        small_roi_path=args.small_roi,
        large_roi_path=args.large_roi,
        output_path=args.output,
        reference_frame=args.reference_frame,
        search_range=tuple(args.search_range),
        correlation_threshold=args.correlation_threshold,
        padding=args.padding,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
