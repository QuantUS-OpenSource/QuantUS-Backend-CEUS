"""
Generate T0 map using only frames with correlation scores above threshold.
This ensures we only use frames where motion correction was reliable.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Correlation threshold - only use frames with correlation >= this value
CORRELATION_THRESHOLD = 0.6

# T0 map parameters
OTSU_THRESHOLD = 0.0025  # Adjust based on your Otsu analysis
START_FRAME = 110
END_FRAME = 485
MIN_CONSECUTIVE_FRAMES = 3

# Paths - UPDATE THESE
CEUS_PATH = '/Users/samantha/Desktop/ultrasound lab stuff/raw_ctdna/p14/wk12/combined_nifti/p14_wk12_CHI_RAW.nii'
MC_ROI_PATH = '/Users/samantha/Desktop/ultrasound lab stuff/raw_mc_ctdna/p14/wk12/p14_wk12_raw_mc_roi.nii.gz'
CORRELATION_SCORES_PATH = '/Users/samantha/Desktop/ultrasound lab stuff/raw_mc_ctdna/p14/wk12/correlation_scores.npy'  # You'll need to save this

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*70)
print("T0 MAP GENERATION WITH CORRELATION FILTERING")
print("="*70)

# Load CEUS data
print(f"\nLoading CEUS data from: {CEUS_PATH}")
ceus_data = nib.load(CEUS_PATH).get_fdata()
print(f"  Shape: {ceus_data.shape}")

# Load motion-corrected ROI
print(f"\nLoading MC ROI from: {MC_ROI_PATH}")
mc_roi = nib.load(MC_ROI_PATH).get_fdata()

# Process ROI if needed (handle side-by-side format)
if mc_roi.shape[0] == 1048:
    print("  Processing combined ROI format...")
    mid = 524
    left_half = mc_roi[:mid, :, :]
    right_half = mc_roi[mid:, :, :]

    if np.sum(left_half > 0) > np.sum(right_half > 0):
        mc_roi = left_half
    else:
        mc_roi = right_half

    if mc_roi.shape != ceus_data.shape:
        mc_roi = mc_roi.transpose(1, 0, 2)

print(f"  ROI shape: {mc_roi.shape}")

# Load correlation scores
print(f"\nLoading correlation scores from: {CORRELATION_SCORES_PATH}")
try:
    correlation_scores = np.load(CORRELATION_SCORES_PATH)
    print(f"  Loaded {len(correlation_scores)} correlation scores")
except FileNotFoundError:
    print(f"  ERROR: File not found!")
    print(f"  Please save your correlation_scores array first using:")
    print(f"  np.save('{CORRELATION_SCORES_PATH}', correlation_scores)")
    exit(1)

# ============================================================================
# FILTER FRAMES BY CORRELATION
# ============================================================================

print(f"\n" + "="*70)
print("FILTERING FRAMES BY CORRELATION THRESHOLD")
print("="*70)

good_frames = np.where(correlation_scores >= CORRELATION_THRESHOLD)[0]
bad_frames = np.where(correlation_scores < CORRELATION_THRESHOLD)[0]

print(f"\nCorrelation threshold: {CORRELATION_THRESHOLD}")
print(f"  Good frames: {len(good_frames)}/{len(correlation_scores)} ({100*len(good_frames)/len(correlation_scores):.1f}%)")
print(f"  Bad frames: {len(bad_frames)}/{len(correlation_scores)} ({100*len(bad_frames)/len(correlation_scores):.1f}%)")

# Create a mask for good frames within our T0 search range
frames_in_range = np.arange(START_FRAME, END_FRAME)
good_frames_in_range = np.intersect1d(good_frames, frames_in_range)

print(f"\nFrames in T0 search range [{START_FRAME}, {END_FRAME}]: {len(frames_in_range)}")
print(f"  Good frames in range: {len(good_frames_in_range)} ({100*len(good_frames_in_range)/len(frames_in_range):.1f}%)")

# ============================================================================
# GENERATE T0 MAP (GOOD FRAMES ONLY)
# ============================================================================

print(f"\n" + "="*70)
print("GENERATING T0 MAP")
print("="*70)

def generate_t0_map_filtered(
    pixel_data,
    seg_mask,
    good_frames,
    threshold,
    start_frame,
    end_frame,
    min_consecutive_frames=3
):
    """
    Generate T0 map using only frames with good correlation scores.

    Parameters
    ----------
    pixel_data : np.ndarray
        Shape (height, width, time)
    seg_mask : np.ndarray
        Shape (height, width, time) or (height, width)
    good_frames : np.ndarray
        Array of frame indices with good correlation
    threshold : float
        Intensity threshold for enhancement detection
    start_frame : int
        Start of search window
    end_frame : int
        End of search window
    min_consecutive_frames : int
        Minimum consecutive frames above threshold

    Returns
    -------
    t0_map : np.ndarray
        T0 map (height, width) with frame indices of first enhancement
    """
    height, width = pixel_data.shape[0], pixel_data.shape[1]
    t0_map = np.zeros((height, width), dtype=np.int32)

    # Get 2D ROI mask
    if seg_mask.ndim == 3:
        roi_2d = np.any(seg_mask > 0, axis=2)
    else:
        roi_2d = seg_mask > 0

    # Filter good frames to be within search range
    frames_to_check = good_frames[(good_frames >= start_frame) & (good_frames < end_frame)]
    frames_to_check = np.sort(frames_to_check)  # Ensure sorted order

    print(f"  Checking {len(frames_to_check)} good frames between {start_frame} and {end_frame}")
    print(f"  Threshold: {threshold}")
    print(f"  Min consecutive frames: {min_consecutive_frames}")

    # For each pixel in ROI
    roi_pixels = np.where(roi_2d)
    total_pixels = len(roi_pixels[0])

    print(f"  Processing {total_pixels} ROI pixels...")

    for idx, (y, x) in enumerate(zip(roi_pixels[0], roi_pixels[1])):
        if idx % 10000 == 0 and idx > 0:
            print(f"    Processed {idx}/{total_pixels} pixels ({100*idx/total_pixels:.1f}%)")

        # Get time series for this pixel (only good frames)
        time_series = pixel_data[y, x, frames_to_check]

        # Find first sustained enhancement
        consecutive_count = 0
        t0_frame = None

        for i, frame_idx in enumerate(frames_to_check):
            if time_series[i] > threshold:
                consecutive_count += 1
                if consecutive_count >= min_consecutive_frames and t0_frame is None:
                    # Found first sustained enhancement
                    t0_frame = frame_idx
                    break
            else:
                consecutive_count = 0

        if t0_frame is not None:
            # Store as time-to-end (higher = earlier arrival)
            t0_map[y, x] = end_frame - t0_frame

    return t0_map


# Generate T0 map
print(f"\nGenerating T0 map using good frames only...")
t0_map = generate_t0_map_filtered(
    pixel_data=ceus_data,
    seg_mask=mc_roi,
    good_frames=good_frames,
    threshold=OTSU_THRESHOLD,
    start_frame=START_FRAME,
    end_frame=END_FRAME,
    min_consecutive_frames=MIN_CONSECUTIVE_FRAMES
)

# Create masked version (NaN for non-enhanced pixels)
if mc_roi.ndim == 3:
    roi_2d = np.any(mc_roi > 0, axis=2)
else:
    roi_2d = mc_roi > 0

t0_map_masked = t0_map.copy().astype(float)
t0_map_masked[~roi_2d] = np.nan
t0_map_masked[t0_map == 0] = np.nan

enhanced_pixels = np.sum(t0_map > 0)
total_roi_pixels = np.sum(roi_2d)

print(f"\n✓ T0 map generated!")
print(f"  Enhanced pixels: {enhanced_pixels}/{total_roi_pixels} ({100*enhanced_pixels/total_roi_pixels:.1f}%)")
print(f"  T0 range: {np.nanmin(t0_map_masked):.0f} to {np.nanmax(t0_map_masked):.0f}")

# ============================================================================
# VISUALIZE T0 MAP
# ============================================================================

print(f"\n" + "="*70)
print("VISUALIZING T0 MAP")
print("="*70)

# Get background frame
background_frame = ceus_data[:, :, 250]

# Normalize for display
vmin = np.percentile(ceus_data, 0.1)
vmax = np.percentile(ceus_data, 99.9)
background_display = np.clip(background_frame, vmin, vmax)
background_display = ((background_display - vmin) / (vmax - vmin) * 255).astype(np.uint8)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Full T0 map
axes[0].imshow(background_display, cmap='gray')
im1 = axes[0].imshow(t0_map_masked, alpha=0.7, cmap='jet')
axes[0].set_title(f'T0 Map - Good Frames Only\nCorrelation >= {CORRELATION_THRESHOLD}', fontsize=14)
axes[0].axis('off')
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='T0 (higher = earlier)')

# Tertile split
if enhanced_pixels > 0:
    t0_values = t0_map_masked[~np.isnan(t0_map_masked)]
    tertile_33 = np.percentile(t0_values, 33.33)
    tertile_67 = np.percentile(t0_values, 66.67)

    # Create tertile masks
    early_map = t0_map_masked.copy()
    early_map[t0_map_masked < tertile_67] = np.nan

    middle_map = t0_map_masked.copy()
    middle_map[(t0_map_masked < tertile_33) | (t0_map_masked >= tertile_67)] = np.nan

    late_map = t0_map_masked.copy()
    late_map[t0_map_masked >= tertile_33] = np.nan

    # Plot
    axes[1].imshow(background_display, cmap='gray')
    axes[1].imshow(early_map, alpha=0.7, cmap='Reds')
    axes[1].imshow(middle_map, alpha=0.7, cmap='Greens')
    axes[1].imshow(late_map, alpha=0.7, cmap='Blues')
    axes[1].set_title('Tertile Split\nRed=Early, Green=Middle, Blue=Late', fontsize=14)
    axes[1].axis('off')

plt.tight_layout()
plt.savefig('/Users/samantha/Desktop/t0_map_correlation_filtered.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to: /Users/samantha/Desktop/t0_map_correlation_filtered.png")

plt.show()

# ============================================================================
# SAVE T0 MAP
# ============================================================================

output_path = '/Users/samantha/Desktop/ultrasound lab stuff/raw_mc_ctdna/p14/wk12/t0_map_corr_filtered.nii.gz'
nib.save(nib.Nifti1Image(t0_map.astype(np.float32), np.eye(4)), output_path)
print(f"✓ Saved T0 map to: {output_path}")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
