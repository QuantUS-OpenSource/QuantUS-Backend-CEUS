"""
3D T0 Map Visualization using Napari
Interactive 3D viewer for parametric maps
"""

import numpy as np
import napari
from typing import Optional, List

# ============================================================================
# NAPARI 3D VISUALIZATION
# ============================================================================

def view_T0_in_napari(vis_obj, param_name='T0_full_TIC', show_all_params=False):
    """
    View T0 parametric map in interactive 3D using Napari
    
    Args:
        vis_obj: Visualization object from visualization_step
        param_name: Parameter to display (default: 'T0_full_TIC')
        show_all_params: If True, load all available parameters as separate layers
    
    Returns:
        viewer: Napari viewer object
    
    Usage:
        viewer = view_T0_in_napari(vis_obj)
        napari.run()  # Start the interactive viewer
    """
    
    # Helper function to convert object array to float
    def convert_to_float_array(obj_array):
        float_array = np.full(obj_array.shape, np.nan, dtype=float)
        for idx in np.ndindex(obj_array.shape):
            val = obj_array[idx]
            if val is not None and val != 'None':
                try:
                    float_array[idx] = float(val)
                except (ValueError, TypeError):
                    pass
        return float_array
    
    # Create napari viewer
    viewer = napari.Viewer()
    
    # Get segmentation mask for reference
    seg_mask = vis_obj.quants_obj.analysis_objs.seg_data.seg_mask
    
    # Check if 3D
    if len(seg_mask.shape) != 3:
        print("Warning: This is 2D data. Napari works best with 3D volumes.")
        print("Consider using plot_T0_from_vis_obj() instead for 2D data.")
    
    # Add segmentation as a label layer
    viewer.add_labels(seg_mask.astype(int), 
                     name='Segmentation',
                     opacity=0.3)
    
    # Determine which parameters to show
    if show_all_params:
        params_to_show = vis_obj.paramap_names
    else:
        params_to_show = [param_name]
    
    # Add each parameter as a separate image layer
    for param in params_to_show:
        try:
            # Find the parameter index
            idx = vis_obj.paramap_names.index(param)
            numerical_map = vis_obj.numerical_paramaps[idx]
            
            # Convert to float array
            param_volume = convert_to_float_array(numerical_map)
            
            # Get valid value range for colormap
            valid_vals = param_volume[~np.isnan(param_volume)]
            if len(valid_vals) == 0:
                print(f"No valid values for {param}, skipping...")
                continue
            
            vmin, vmax = np.percentile(valid_vals, [5, 95])
            
            # Choose colormap based on parameter
            if 'T0' in param or 'TP' in param:
                colormap = 'turbo'  # Good for time parameters
            elif 'PE' in param or 'AUC' in param:
                colormap = 'viridis'  # Good for intensity
            elif 'MTT' in param:
                colormap = 'plasma'
            else:
                colormap = 'inferno'
            
            # Add as image layer
            viewer.add_image(
                param_volume,
                name=param,
                colormap=colormap,
                contrast_limits=[vmin, vmax],
                opacity=0.8,
                visible=(param == param_name)  # Only show the requested param initially
            )
            
            print(f"Added {param}: range=[{vmin:.2f}, {vmax:.2f}], mean={np.mean(valid_vals):.2f}")
            
        except ValueError:
            print(f"Parameter {param} not found in vis_obj.paramap_names")
            continue
    
    # Set up the viewer
    viewer.dims.axis_labels = ['Sagittal (Y)', 'Coronal (Z)', 'Axial (X)']
    viewer.camera.angles = (0, 0, 90)  # Initial viewing angle
    viewer.camera.zoom = 2.0
    
    print("\n=== Napari Controls ===")
    print("- Scroll: Navigate through slices")
    print("- Click and drag: Rotate 3D view")
    print("- Shift + scroll: Zoom")
    print("- Layer list (left): Toggle visibility, adjust opacity")
    print("- 3D view button (bottom left): Switch to 3D rendering")
    print("- Colormap controls (top): Adjust contrast limits")
    
    return viewer


def view_T0_with_CEUS(vis_obj, image_data, param_name='T0_full_TIC', 
                      time_point=None, show_motion_compensated=True):
    """
    View T0 map overlaid on CEUS image data
    
    Args:
        vis_obj: Visualization object
        image_data: UltrasoundImage object from scan_loading_step
        param_name: Parameter to display
        time_point: Which time frame to show (None = middle frame)
        show_motion_compensated: If True, show motion-compensated segmentation
    
    Returns:
        viewer: Napari viewer object
    """
    
    # Helper function
    def convert_to_float_array(obj_array):
        float_array = np.full(obj_array.shape, np.nan, dtype=float)
        for idx in np.ndindex(obj_array.shape):
            val = obj_array[idx]
            if val is not None and val != 'None':
                try:
                    float_array[idx] = float(val)
                except (ValueError, TypeError):
                    pass
        return float_array
    
    viewer = napari.Viewer()
    
    # Get CEUS volume
    ceus_volume = image_data.intensities_for_analysis
    
    if time_point is None:
        time_point = ceus_volume.shape[-1] // 2  # Middle frame
    
    # Add CEUS image (single time point)
    if len(ceus_volume.shape) == 4:  # 3D + time
        ceus_frame = ceus_volume[:, :, :, time_point]
    else:  # 2D + time
        ceus_frame = ceus_volume[time_point]
    
    viewer.add_image(
        ceus_frame,
        name=f'CEUS (t={time_point})',
        colormap='gray',
        opacity=0.6
    )
    
    # Add motion-compensated segmentation if available
    seg_data = vis_obj.quants_obj.analysis_objs.seg_data
    if show_motion_compensated and hasattr(seg_data, 'use_mc') and seg_data.use_mc:
        # Apply motion compensation for this time point
        mc_mask = seg_data.motion_compensation.apply_to_mask(
            seg_data.seg_mask, time_point, order=0
        )
        viewer.add_labels(
            mc_mask.astype(int),
            name=f'MC Segmentation (t={time_point})',
            opacity=0.3
        )
    else:
        # Static segmentation
        viewer.add_labels(
            seg_data.seg_mask.astype(int),
            name='Segmentation',
            opacity=0.3
        )
    
    # Add T0 map
    try:
        idx = vis_obj.paramap_names.index(param_name)
        numerical_map = vis_obj.numerical_paramaps[idx]
        param_volume = convert_to_float_array(numerical_map)
        
        valid_vals = param_volume[~np.isnan(param_volume)]
        if len(valid_vals) > 0:
            vmin, vmax = np.percentile(valid_vals, [5, 95])
            
            viewer.add_image(
                param_volume,
                name=param_name,
                colormap='turbo',
                contrast_limits=[vmin, vmax],
                opacity=0.7
            )
    except ValueError:
        print(f"Parameter {param_name} not found")
    
    viewer.dims.axis_labels = ['Sagittal (Y)', 'Coronal (Z)', 'Axial (X)']
    viewer.title = f'{param_name} Map with CEUS Background'
    
    return viewer


def view_heatmap(
    t0_map: np.ndarray,
    image_data,
    time_point: Optional[int] = None,
    seg_mask: Optional[np.ndarray] = None,
    colormap: str = 'turbo',
    t0_opacity: float = 0.7,
    percentile_range: tuple = (2, 98),
):
    """
    Display a T0 heatmap overlaid on CEUS image data in Napari.

    This is a lightweight viewer that works directly with a T0 numpy array
    and an UltrasoundImage, without requiring the full visualization pipeline.
    """
    viewer = napari.Viewer()
    ceus = image_data.intensities_for_analysis

    # ------------------------------------------------------------------
    # Background CEUS frame
    # ------------------------------------------------------------------
    is_3d = ceus.ndim == 4  # (sag, cor, ax, time)

    if time_point is None:
        time_point = ceus.shape[-1] // 2 if is_3d else ceus.shape[0] // 2

    if is_3d:
        bg_frame = ceus[:, :, :, time_point]
    else:
        bg_frame = ceus[time_point]

    viewer.add_image(
        bg_frame,
        name=f'CEUS (t={time_point})',
        colormap='gray',
        opacity=0.5,
    )

    # ------------------------------------------------------------------
    # Optional segmentation outline
    # ------------------------------------------------------------------
    if seg_mask is not None:
        mask_to_show = np.asarray(seg_mask)
        # If per-frame mask, take the union for display
        if mask_to_show.ndim == 4:
            mask_to_show = np.any(mask_to_show > 0, axis=3).astype(int)
        else:
            mask_to_show = (mask_to_show > 0).astype(int)

        viewer.add_labels(
            mask_to_show,
            name='VOI',
            opacity=0.2,
        )

    # ------------------------------------------------------------------
    # T0 heatmap overlay
    # ------------------------------------------------------------------
    # Keep zero-valued voxels as 0 so they render as dark pixels.
    # Contrast limits start at 0 so the colormap maps 0 → darkest colour.
    t0_display = t0_map.astype(np.float64).copy()

    valid = t0_display[t0_display > 0]
    if len(valid) > 0:
        _, vmax = np.percentile(valid, list(percentile_range))
        viewer.add_image(
            t0_display,
            name='T0 Map',
            colormap=colormap,
            contrast_limits=[0, vmax],
            opacity=t0_opacity,
        )
        print(f"T0 Map: contrast=[0, {vmax:.1f}], "
              f"mean(activated)={np.mean(valid):.1f}, "
              f"activated voxels={len(valid)}")
    else:
        print("Warning: T0 map has no activated voxels to display.")

    # ------------------------------------------------------------------
    # Viewer setup
    # ------------------------------------------------------------------
    if is_3d:
        viewer.dims.axis_labels = ['Sagittal (Y)', 'Coronal (Z)', 'Axial (X)']
    viewer.camera.zoom = 2.0
    viewer.title = 'T0 Heatmap'

    return viewer