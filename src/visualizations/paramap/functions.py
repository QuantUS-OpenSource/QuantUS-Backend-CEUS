import numpy as np
from .framework import CurveQuantifications

def plot_tics(quants_obj: CurveQuantifications, paramap_folder_path: str, **kwargs):
    """Plot time-intensity curves for all voxels in the segmentation mask.

    Args:
        quans_obj (Quantifications): The Quantifications object containing the analysis data.
        paramap_folder_path (str): The folder path to save the plots.
        **kwargs: Additional keyword arguments for customization.
    """
    if kwargs.get('hide_all_visualizations', False):
        return
    
    time = quants_obj.analysis_objs.time_arr
    start_time = kwargs.get('start_time', 0)
    end_time = kwargs.get('end_time', time[-1]+1)

    curves = quants_obj.analysis_objs.curves
    assert 'TIC' in curves[0].keys(), "TIC curves not found in the analysis object."
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Create output directory if it doesn't exist
    output_dir = Path(paramap_folder_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot TICs for each voxel in the segmentation mask
    plt.figure()
    tics = []
    
    start_frame = int(np.searchsorted(quants_obj.analysis_objs.time_arr, start_time, side='left'))
    end_frame = int(np.searchsorted(quants_obj.analysis_objs.time_arr, end_time, side='right'))
    for curve in curves:
        tic = curve['TIC']
        plt.plot(time[start_frame:end_frame], tic[start_frame:end_frame], alpha=0.3)
        tics.append(tic)
    
    tics = np.array(tics)
    av_tic = np.nanmean(tics, axis=0)
    plt.plot(time[start_frame:end_frame], av_tic[start_frame:end_frame], color='red', linewidth=2, label='Average TIC')
    
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity')
    plt.title('All TICs')
    plt.savefig(output_dir / f'tics.png')
    plt.close()
