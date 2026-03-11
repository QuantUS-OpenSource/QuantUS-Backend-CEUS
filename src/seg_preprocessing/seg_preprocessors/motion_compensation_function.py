from ..decorators import required_kwargs
from ...data_objs.image import UltrasoundImage
from ...data_objs.seg import CeusSeg
from ..motion_compensation_3d import MotionCompensation3D, BoundingBox3D, MotionCompensationResult
import numpy as np

@required_kwargs('bmode_image_data','search_margin_ratio','padding')
def motion_compensation_3d(image_data: UltrasoundImage, seg_data: CeusSeg, **kwargs) -> CeusSeg:
    """
    Apply 3D motion compensation using ILSA tracking.
    
    MEMORY EFFICIENT: Stores only translation vectors (~10 KB) instead of full 4D mask (36 GB).
    Motion compensation is applied on-demand when needed for analysis.
    
    Kwargs:
        bmode_image_data (UltrasoundImage): B-mode data for motion tracking [REQUIRED]
        reference_frame (int): Reference frame index (default: 0)
        search_margin_ratio (float): Search margin ratio (default: 0.5/30)
        padding (int): Padding around bounding box (default: 5)
        shift_order (int): Interpolation order for shifting (default: 0 for nearest neighbor)
        precompute_mc_mask (bool): If True, create full 4D mc_seg_mask (uses ~36 GB). 
                                    If False (default), only store vectors (uses ~10 KB)
    
    Returns:
        CeusSeg: Segmentation with motion compensation info stored
    """
    # Extract kwargs
    bmode_image_data = kwargs['bmode_image_data']
    reference_frame = kwargs.get('reference_frame', 0)
    search_margin_ratio = kwargs.get('search_margin_ratio', 0.5 / 25)
    padding = kwargs.get('padding', 5)
    shift_order = kwargs.get('shift_order', 0)  # 0=nearest neighbor for masks
    precompute_mc_mask = kwargs.get('precompute_mc_mask', False)  # Default: memory efficient
    
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
    print(f"  Search margin ratio: {search_margin_ratio}")

    mc = MotionCompensation3D(
        search_margin_ratio=search_margin_ratio,
        use_reference_only = True
    )
    
    # Track motion - volumes are (X,Y,Z) - (Lateral, Depth, Elevational)
    tracked_bboxes, correlations = mc.track_motion_ilsa_3d(
        bmode_image_data.pixel_data,
        reference_frame,
        reference_bbox
    )
    
    # Step 3: Calculate translation vectors (memory efficient!)
    print("\nStep 3: Computing translation vectors...")
    n_frames = bmode_shape[-1]
    
    # Store translation vectors instead of full mask
    translation_vectors = np.zeros((n_frames, 3), dtype=np.float32)
    ref_center = reference_bbox.center
    
    for frame_idx in range(n_frames):
        bbox = tracked_bboxes[frame_idx]
        curr_center = bbox.center
        
        # Calculate shift from reference (these are the motion compensation vectors)
        translation_vectors[frame_idx, 0] = curr_center[0] - ref_center[0]  # dz
        translation_vectors[frame_idx, 1] = curr_center[1] - ref_center[1]  # dy
        translation_vectors[frame_idx, 2] = curr_center[2] - ref_center[2]  # dx
        
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
        tracked_bboxes=tracked_bboxes
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