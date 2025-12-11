# ============================================================================
# GENERATE T0 MAP USING ONLY FRAMES WITH GOOD CORRELATION (>= 0.6)
# ============================================================================

print("="*70)
print("T0 MAP GENERATION WITH CORRELATION FILTERING")
print("="*70)

# Set correlation threshold
correlation_threshold = 0.6

# Filter frames by correlation
good_frames = np.where(correlation_scores >= correlation_threshold)[0]
bad_frames = np.where(correlation_scores < correlation_threshold)[0]

print(f"\nCorrelation threshold: {correlation_threshold}")
print(f"  Good frames: {len(good_frames)}/{len(correlation_scores)} ({100*len(good_frames)/len(correlation_scores):.1f}%)")
print(f"  Bad frames: {len(bad_frames)}/{len(correlation_scores)} ({100*len(bad_frames)/len(correlation_scores):.1f}%)")

# T0 parameters
otsu_threshold_value = 0.0025  # Use your Otsu threshold from earlier
start_frame = 110
end_frame = 485
min_consecutive_frames = 3

# Filter good frames to those in our search range
frames_in_range = np.arange(start_frame, end_frame)
good_frames_in_range = np.intersect1d(good_frames, frames_in_range)

print(f"\nT0 search parameters:")
print(f"  Frame range: [{start_frame}, {end_frame}]")
print(f"  Total frames in range: {len(frames_in_range)}")
print(f"  Good frames in range: {len(good_frames_in_range)} ({100*len(good_frames_in_range)/len(frames_in_range):.1f}%)")
print(f"  Threshold: {otsu_threshold_value}")
print(f"  Min consecutive frames: {min_consecutive_frames}")


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

    print(f"\n  Processing {len(frames_to_check)} good frames...")

    # For each pixel in ROI
    roi_pixels = np.where(roi_2d)
    total_pixels = len(roi_pixels[0])

    print(f"  Processing {total_pixels} ROI pixels...")

    for idx, (y, x) in enumerate(zip(roi_pixels[0], roi_pixels[1])):
        if idx % 10000 == 0 and idx > 0:
            print(f"    {idx}/{total_pixels} pixels ({100*idx/total_pixels:.1f}%)")

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


# Generate T0 map using good frames only
print(f"\nGenerating T0 map...")
t0_map_filtered = generate_t0_map_filtered(
    pixel_data=ceus_data,
    seg_mask=large_roi_compensated,
    good_frames=good_frames,
    threshold=otsu_threshold_value,
    start_frame=start_frame,
    end_frame=end_frame,
    min_consecutive_frames=min_consecutive_frames
)

# Create masked version
if large_roi_compensated.ndim == 3:
    roi_2d = np.any(large_roi_compensated > 0, axis=2)
else:
    roi_2d = large_roi_compensated > 0

t0_map_masked_filtered = t0_map_filtered.copy().astype(float)
t0_map_masked_filtered[~roi_2d] = np.nan
t0_map_masked_filtered[t0_map_filtered == 0] = np.nan

enhanced_pixels = np.sum(t0_map_filtered > 0)
total_roi_pixels = np.sum(roi_2d)

print(f"\n✓ T0 map generated!")
print(f"  Enhanced pixels: {enhanced_pixels}/{total_roi_pixels} ({100*enhanced_pixels/total_roi_pixels:.1f}%)")
if enhanced_pixels > 0:
    print(f"  T0 range: {np.nanmin(t0_map_masked_filtered):.0f} to {np.nanmax(t0_map_masked_filtered):.0f}")

# ============================================================================
# VISUALIZE
# ============================================================================

# Get background frame
background_frame = ceus_data[:, :, 250]

# Normalize for display
vmin = np.percentile(ceus_data, 0.1)
vmax = np.percentile(ceus_data, 99.9)
background_display = np.clip(background_frame, vmin, vmax)
background_display = ((background_display - vmin) / (vmax - vmin) * 255).astype(np.uint8)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Full T0 map
axes[0].imshow(background_display, cmap='gray')
im1 = axes[0].imshow(t0_map_masked_filtered, alpha=0.7, cmap='jet')
axes[0].set_title(f'T0 Map - Correlation Filtered (>= {correlation_threshold})\n{len(good_frames_in_range)} good frames', fontsize=14, fontweight='bold')
axes[0].axis('off')
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='T0 (higher = earlier)')

# Tertile split
if enhanced_pixels > 0:
    t0_values = t0_map_masked_filtered[~np.isnan(t0_map_masked_filtered)]
    tertile_33 = np.percentile(t0_values, 33.33)
    tertile_67 = np.percentile(t0_values, 66.67)

    # Convert to frame numbers for display
    frame_max = end_frame - t0_values.min()
    frame_67 = end_frame - tertile_67
    frame_33 = end_frame - tertile_33
    frame_min = end_frame - t0_values.max()

    # Create tertile masks
    early_map = t0_map_masked_filtered.copy()
    early_map[t0_map_masked_filtered < tertile_67] = np.nan

    middle_map = t0_map_masked_filtered.copy()
    middle_map[(t0_map_masked_filtered < tertile_33) | (t0_map_masked_filtered >= tertile_67)] = np.nan

    late_map = t0_map_masked_filtered.copy()
    late_map[t0_map_masked_filtered >= tertile_33] = np.nan

    # Plot
    axes[1].imshow(background_display, cmap='gray')
    axes[1].imshow(early_map, alpha=0.7, cmap='Reds')
    axes[1].imshow(middle_map, alpha=0.7, cmap='Greens')
    axes[1].imshow(late_map, alpha=0.7, cmap='Blues')
    axes[1].set_title(f'Tertile Split\nRed=Early ({frame_min:.0f}-{frame_67:.0f})\nGreen=Middle ({frame_67:.0f}-{frame_33:.0f})\nBlue=Late ({frame_33:.0f}-{frame_max:.0f})',
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')

plt.tight_layout()
plt.show()

print("="*70)
