from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage
from ...data_objs.seg import CeusSeg
from ..motion_compensation_3d import (MotionCompensation3D, BoundingBox3D,
                                      MotionCompensationResult, compute_sector_mask)
import numpy as np

@required_kwargs('bmode_image_data','search_margin','padding','reference_frame')
def motion_compensation_3d(image_data: UltrasoundImage, seg_data: CeusSeg, **kwargs) -> CeusSeg:
    """
    Apply 3D motion compensation using ILSA tracking.
    
    MEMORY EFFICIENT: Stores only translation vectors (~10 KB) instead of full 4D mask (36 GB).
    Motion compensation is applied on-demand when needed for analysis.
    
    Kwargs:
        bmode_image_data (UltrasoundImage): B-mode data for motion tracking [REQUIRED]
        reference_frame (int): Reference frame index (default: 0)
        search_margin (Tuple[int,int,int]): Per-axis (X, Y, Z) search margin in
                                    voxels (default: (5, 5, 5)). This is the largest
                                    per-frame displacement findable on each axis; raise
                                    the Z entry for large out-of-plane motion.
        padding (int): Padding around bounding box (default: 5)
        shift_order (int): Interpolation order for shifting (default: 0 for nearest neighbor)
        precompute_mc_mask (bool): If True, create full 4D mc_seg_mask (uses ~36 GB).
                                    If False (default), only store vectors (uses ~10 KB)
        n_splits (Tuple[int,int,int]): ROI grid for block-wise tracking (default: (3, 3, 2)).
                                    Large ROIs are tiled into this many sub-blocks per axis;
                                    each is tracked independently and blocks whose track
                                    reaches the image edge, or whose correlation is too low,
                                    are dropped before averaging.
        min_correlation (float): Minimum mean correlation for a sub-block to be kept (default: 0.5)
        use_gpu (bool): Run the 3D correlation on the GPU via CuPy (default: True).
                                    Falls back to CPU if CuPy/CUDA is unavailable.
        use_reference_only (bool): If True, every frame is matched against the
                                    reference-frame template. If False (default), ILSA also
                                    tries the neighbouring frame's template and keeps whichever
                                    correlates better, which tracks large cumulative motion and
                                    gradual appearance change far better.

    Returns:
        CeusSeg: Segmentation with motion compensation info stored
    """
    # Extract kwargs
    bmode_image_data = kwargs['bmode_image_data']
    reference_frame = kwargs.get('reference_frame', 0)
    padding = kwargs.get('padding', 5)
    shift_order = kwargs.get('shift_order', 0)  # 0=nearest neighbor for masks
    precompute_mc_mask = kwargs.get('precompute_mc_mask', False)  # Default: memory efficient
    n_splits = kwargs.get('n_splits', (3, 3, 2))
    min_correlation = kwargs.get('min_correlation', 0.5)
    use_gpu = kwargs.get('use_gpu', True)
    search_margin = kwargs.get('search_margin', (5, 5, 5))
    use_reference_only = kwargs.get('use_reference_only', False)
    
    # Validate inputs
    if not isinstance(bmode_image_data, UltrasoundImage):
        raise TypeError("bmode_image_data must be an UltrasoundImage object")
    
    bmode_shape = bmode_image_data.pixel_data.shape
    if bmode_image_data.pixel_data.ndim != 4:
        raise ValueError(f"B-mode data must be 4D (X,Y,Z,T), got shape {bmode_shape}")
    
    reference_mask = seg_data.seg_mask
    # seg_mask should be (X,Y,Z) - single frame
    seg_mask_shape = reference_mask.shape
    if reference_mask.ndim != 3:
        raise ValueError(f"Segmentation mask must be 3D (X, Y, Z), got shape {seg_mask_shape}")

    print("\n" + "="*60)
    print("3D Motion Compensation with ILSA Tracking (Memory Efficient)")
    print("="*60)
    
    # Step 1: Extract bounding box from segmentation
    print("\nStep 1: Extracting bounding box from segmentation...")
    try:
        reference_bbox = BoundingBox3D.from_mask(reference_mask, padding=padding)
        print(f"  Bounding box: Z=[{reference_bbox.z_min}, {reference_bbox.z_max}], "
              f"Y=[{reference_bbox.y_min}, {reference_bbox.y_max}], "
              f"X=[{reference_bbox.x_min}, {reference_bbox.x_max}]")
        print(f"  Center: {reference_bbox.center}")
    except ValueError as e:
        print(f"Error: {e}")
        return seg_data
    
    # Step 2: Track motion using forward and backward correlation
    print("\nStep 2: Tracking motion using forward and backward correlation...")
    print(f"  Reference frame: {reference_frame}")
    print(f"  Search margin: {tuple(search_margin)} voxels (X, Y, Z)")

    mc = MotionCompensation3D(
        search_margin=search_margin,
        use_reference_only=use_reference_only,
        n_splits=n_splits,
        use_gpu=use_gpu
    )
    
    # Track motion - volumes are (X,Y,Z) - (Lateral, Depth, Elevational)
    tracked_bboxes, correlations, translations = mc.track_motion_blockwise_3d(
        bmode_image_data.pixel_data,
        reference_frame,
        reference_bbox,
        min_correlation=min_correlation
    )
    
    # Step 3: Calculate translation vectors (memory efficient!)
    print("\nStep 3: Computing translation vectors...")
    n_frames = bmode_shape[-1]
    
    # Store translation vectors instead of full mask. These come straight from
    # the tracker's own shifts, not from differencing tracked_bboxes' centers:
    # those boxes are clipped to the volume, which truncates a large ROI on one
    # side only and damps the motion it appears to have undergone.
    translation_vectors = np.zeros((n_frames, 3), dtype=np.float32)
    
    for frame_idx in range(n_frames):
        # (X, Y, Z) = (lateral, depth, elevational), matching the volume axes
        translation_vectors[frame_idx] = translations[frame_idx]
        
        if frame_idx % 10 == 0 or frame_idx == n_frames - 1:
            print(f"  Frame {frame_idx}: shift=({translation_vectors[frame_idx, 0]:.1f}, "
                  f"{translation_vectors[frame_idx, 1]:.1f}, {translation_vectors[frame_idx, 2]:.1f}), "
                  f"corr={correlations[frame_idx]:.3f}")
    
    # Create MotionCompensationResult object
    mc_result = MotionCompensationResult(
        translation_vectors=translation_vectors,
        reference_frame=reference_frame,
        correlations=np.array(correlations, dtype=np.float32),
        reference_bbox=reference_bbox,
        tracked_bboxes=tracked_bboxes,
        sector_mask=compute_sector_mask(bmode_image_data.pixel_data, reference_frame)
    )
    
    # Store motion compensation result in seg_data
    seg_data.motion_compensation = mc_result
    seg_data.use_mc = True

    # Store motion info in extras_dict
    image_data.extras_dict['motion_compensation'] = {
        'applied': True,
        'reference_frame': reference_frame,
        'mean_correlation': float(np.mean(correlations)),
        'min_correlation': float(np.min(correlations)),
        'bboxes': [
            {
                'z_min': bbox.z_min, 'z_max': bbox.z_max,
                'y_min': bbox.y_min, 'y_max': bbox.y_max,
                'x_min': bbox.x_min, 'x_max': bbox.x_max,
                'center': bbox.center
            } for bbox in tracked_bboxes
        ],
        'correlations': [float(c) for c in correlations]
    }
    
    print("\n" + "="*60)
    print("Motion Compensation Complete!")
    print(f"  Mean correlation: {np.mean(correlations):.3f}")
    print("="*60 + "\n")
    
    return seg_data