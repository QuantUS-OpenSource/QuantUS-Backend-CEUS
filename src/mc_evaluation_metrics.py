"""
MODIFIED: 3D Motion Compensation Evaluation Metrics
- Generates MC masks on-the-fly to save memory
- Uses updated lognormal fitting function
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import pandas as pd
from skimage import metrics
from tqdm import tqdm

# ============================================================================
# VOI B-mode Similarity Metrics (MEMORY EFFICIENT)
# ============================================================================

def compute_roi_similarity_metrics(volume1, volume2, ref_mask, mask):
    """
    Compute similarity metrics between two volumes within a masked region.
    
    This measures how similar the B-mode intensities are within the VOI,
    which reflects whether the VOI is tracking the same tissue.
    
    Args:
        volume1: Reference volume (z, y, x)
        volume2: Current volume (z, y, x)
        ref_mask: Binary mask defining VOI in reference volume (z, y, x)
        mask: Binary mask defining VOI (z, y, x)
        
    Returns:
        dict: Similarity metrics
    """
    # Extract intensities within mask
    roi1 = volume1[ref_mask > 0]
    roi2 = volume2[mask > 0]
    
    if len(roi1) == 0 or len(roi2) == 0:
        return {
            'correlation': 0.0,
            'ssim': 0.0,
            'mse': np.inf,
            'mae': np.inf
        }
    
    # Ensure same length for comparison
    min_len = min(len(roi1), len(roi2))
    roi1 = roi1[:min_len]
    roi2 = roi2[:min_len]
    
    # 1. Pearson Correlation
    if len(roi1) > 1:
        correlation, _ = pearsonr(roi1.flatten(), roi2.flatten())
    else:
        correlation = 0.0
    
    # 2. Structural Similarity
    ssim_val = metrics.structural_similarity(roi1, roi2, data_range=roi2.max()-roi2.min())
    
    # 3. Mean Squared Error
    mse = mean_squared_error(roi1, roi2)
    
    # 4. Mean Absolute Error
    mae = np.mean(np.abs(roi1 - roi2))
    
    return {
        'correlation': correlation,
        'ssim': ssim_val,
        'mse': mse,
        'mae': mae
    }


def compute_voi_bmode_similarity_over_time(
    bmode_volumes,
    base_mask,
    motion_compensation_result,
    reference_frame=0,
    use_mc=True
):
    """
    MEMORY EFFICIENT VERSION: Compute B-mode similarity WITHIN VOI over time.
    Generates MC masks on-the-fly instead of storing 4D array.
    
    Args:
        bmode_volumes: B-mode data (z, y, x, t)
        base_mask: Base 3D mask (z, y, x)
        motion_compensation_result: MotionCompensationResult object
        reference_frame: Reference frame index
        use_mc: Whether to apply motion compensation
        
    Returns:
        dict: Results with similarity metrics over time
    """
    n_frames = bmode_volumes.shape[-1]
    ref_volume = bmode_volumes[..., reference_frame]
    
    # Get reference mask (with or without MC)
    if use_mc:
        ref_mask = motion_compensation_result.apply_to_mask(base_mask, reference_frame, order=0)
    else:
        ref_mask = base_mask.copy()
    
    # Initialize results
    results = {
        'correlation': [],
        'ssim': [],
        'mse': [],
        'mae': []
    }
    
    print(f"Computing B-mode similarity {'WITH' if use_mc else 'WITHOUT'} motion compensation...")
    
    for frame_idx in tqdm(range(n_frames), desc="Processing frames"):
        current_volume = bmode_volumes[..., frame_idx]
        
        # Generate mask for current frame on-the-fly
        if use_mc:
            current_mask = motion_compensation_result.apply_to_mask(base_mask, frame_idx, order=0)
        else:
            current_mask = base_mask.copy()
        
        # Compare B-mode within current mask to B-mode within reference mask
        metrics_dict = compute_roi_similarity_metrics(
            ref_volume, current_volume, ref_mask, current_mask
        )
        
        for key in metrics_dict:
            results[key].append(metrics_dict[key])
    
    print("  Done!")
    
    return results


def compute_voi_bmode_similarity_comparison(
    bmode_volumes,
    base_mask,
    motion_compensation_result,
    reference_frame=0
):
    """
    Compare MC vs non-MC similarity using on-the-fly mask generation.
    
    Args:
        bmode_volumes: B-mode data (z, y, x, t)
        base_mask: Base 3D mask (z, y, x)
        motion_compensation_result: MotionCompensationResult object
        reference_frame: Reference frame index
        
    Returns:
        dict: Results for both MC and non-MC cases
    """
    # Compute with MC
    results_mc = compute_voi_bmode_similarity_over_time(
        bmode_volumes,
        base_mask,
        motion_compensation_result,
        reference_frame,
        use_mc=True
    )
    
    # Compute without MC
    results_no_mc = compute_voi_bmode_similarity_over_time(
        bmode_volumes,
        base_mask,
        motion_compensation_result,
        reference_frame,
        use_mc=False
    )
    
    return {
        'with_mc': results_mc,
        'without_mc': results_no_mc
    }


def plot_voi_bmode_similarity_comparison(results, output_path=None):
    """
    Create comprehensive plots comparing B-mode similarity within VOI.
    
    Args:
        results: Dictionary from compute_voi_bmode_similarity_comparison
        output_path: Optional path to save figure
    """
    metrics = ['correlation', 'ssim']
    metric_names = ['Pearson Correlation', 'SSIM']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    n_frames = len(results['with_mc']['correlation'])
    frames = np.arange(n_frames)
    
    # Plot similarity metrics
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        with_mc = results['with_mc'][metric]
        without_mc = results['without_mc'][metric]
        
        # Plot curves
        ax.plot(frames, with_mc, 'g-', linewidth=2, alpha=0.7, 
               label='With Motion Compensation')
        ax.plot(frames, without_mc, 'r-', linewidth=2, alpha=0.7,
               label='Without Motion Compensation')
        
        # Calculate mean values
        mean_mc = np.mean(with_mc)
        mean_no_mc = np.mean(without_mc)
        
        # Add horizontal lines for means
        ax.axhline(mean_mc, color='g', linestyle='--', alpha=0.5,
                  label=f'Mean MC: {mean_mc:.3f}')
        ax.axhline(mean_no_mc, color='r', linestyle='--', alpha=0.5,
                  label=f'Mean No-MC: {mean_no_mc:.3f}')
        
        ax.set_xlabel('Frame Number', fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f'{name} Within VOI Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set appropriate y-limits
        if metric in ['correlation', 'ssim']:
            ax.set_ylim([-0.1, 1.05])
    
    # Summary statistics in last subplot
    ax_summary = axes[-1]
    ax_summary.axis('off')
    
    summary_text = "B-mode Similarity Summary:\n\n"
    summary_text += "Measuring tissue appearance within VOI\n"
    summary_text += "(Higher = VOI tracking same tissue)\n\n"
    
    for metric, name in zip(metrics, metric_names):
        mean_mc = np.mean(results['with_mc'][metric])
        std_mc = np.std(results['with_mc'][metric])
        mean_no_mc = np.mean(results['without_mc'][metric])
        std_no_mc = np.std(results['without_mc'][metric])
        improvement = ((mean_mc - mean_no_mc) / (mean_no_mc + 1e-10)) * 100
        
        summary_text += f"{name}:\n"
        summary_text += f"  MC:     {mean_mc:.3f} ± {std_mc:.3f}\n"
        summary_text += f"  No-MC:  {mean_no_mc:.3f} ± {std_no_mc:.3f}\n"
        summary_text += f"  Improv: {improvement:+.1f}%\n\n"
    
    # MSE and MAE (lower is better)
    summary_text += "Error Metrics (lower is better):\n"
    mse_mc = np.mean(results['with_mc']['mse'])
    mse_no_mc = np.mean(results['without_mc']['mse'])
    mse_reduction = ((mse_no_mc - mse_mc) / (mse_no_mc + 1e-10)) * 100
    
    summary_text += f"MSE:\n"
    summary_text += f"  MC:     {mse_mc:.1f}\n"
    summary_text += f"  No-MC:  {mse_no_mc:.1f}\n"
    summary_text += f"  Reduct: {mse_reduction:.1f}%\n"
    
    ax_summary.text(0.1, 0.5, summary_text, fontsize=9, 
                   verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved VOI B-mode similarity plot to {output_path}")
    
    plt.show()
    
    return fig


# ============================================================================
# TIC Analysis with Updated Lognormal Fitting
# ============================================================================

def bolus_lognormal(t, auc, mu, sigma, t0):
    """Log-normal bolus function for contrast enhancement curve fitting."""
    t_shifted = t - t0
    t_shifted = np.maximum(t_shifted, 1e-10)
    exponent = -((np.log(t_shifted) - mu) ** 2) / (2 * sigma ** 2)
    return (auc / (t_shifted * sigma * np.sqrt(2 * np.pi))) * np.exp(exponent)


def fit_lognormal_curve(time, curve):
    """
    Fit a log-normal distribution to the given curve.
    
    Args:
        time (np.ndarray): The time array corresponding to the curve.
        curve (np.ndarray): The curve data to fit.
        
    Returns:
        tuple: Fitted parameters (auc, pe, tp, mtt, t0, mu, sigma, pe_loc) and fitted curve
    """
    curve = curve.copy()  # Don't modify original
    curve -= np.amin(curve)  # Shift to start at zero
    
    if np.amax(curve) == 0:
        print("Curve is constant, cannot normalize.")
        return tuple(np.nan for _ in range(8)), None
    
    curve = curve / np.amax(curve)  # Normalize
    
    # Initial guesses
    auc_guess = np.sum(curve) * (time[1] - time[0])
    peak_idx = np.argmax(curve)
    mu_guess = np.log(time[peak_idx] + 1e-10)  # Use time value, not index
    sigma_guess = 0.5
    t0_guess = time[peak_idx] * 0.15
    
    try:
        params, _ = curve_fit(
            bolus_lognormal,
            time,
            curve,
            p0=(auc_guess, mu_guess, sigma_guess, t0_guess),
            bounds=([0., -10., 0.01, 0.], [np.inf, 10., 5.0, time[-1]]),
            method='trf',
            maxfev=10000  # Increase evaluations
        )
    except Exception as e:
        print(f"Error fitting curve: {e}")
        return tuple(np.nan for _ in range(8)), None
    
    auc, mu, sigma, t0 = params
    
    # Calculate derived parameters
    mtt = np.exp(mu + sigma**2 / 2)
    tp = np.exp(mu - sigma**2)
    
    # Get fitted curve
    fitted_curve = bolus_lognormal(time, *params)
    pe = np.max(fitted_curve)
    pe_loc = np.argmax(fitted_curve)
    
    return (auc, pe, tp, mtt, t0, mu, sigma, pe_loc), fitted_curve


def compute_tic_from_volumes(
    ceus_volumes,
    base_mask,
    motion_compensation_result,
    use_mc=True
):
    """
    Compute TIC by generating masks on-the-fly.
    
    Args:
        ceus_volumes: CEUS data (z, y, x, t)
        base_mask: Base 3D mask (z, y, x)
        motion_compensation_result: MotionCompensationResult object
        use_mc: Whether to apply motion compensation
        
    Returns:
        np.ndarray: TIC curve (mean intensity per frame)
    """
    n_frames = ceus_volumes.shape[-1]
    tic = np.zeros(n_frames)
    
    print(f"Computing TIC {'WITH' if use_mc else 'WITHOUT'} motion compensation...")
    
    for frame_idx in tqdm(range(n_frames), desc="Computing TIC"):
        # Generate mask for this frame on-the-fly
        if use_mc:
            mask = motion_compensation_result.apply_to_mask(base_mask, frame_idx, order=0)
        else:
            mask = base_mask.copy()
        
        # Extract CEUS intensities within mask
        frame_volume = ceus_volumes[..., frame_idx]
        roi_intensities = frame_volume[mask > 0]
        
        # Compute mean intensity
        if len(roi_intensities) > 0:
            tic[frame_idx] = np.mean(roi_intensities)
        else:
            tic[frame_idx] = 0.0
    
    print("  Done!")
    return tic


def evaluate_tic_fitting(time_arr, tic_mc, tic_no_mc):
    """
    Evaluate and compare TIC fitting quality for MC vs non-MC.
    
    Args:
        time_arr: Time array
        tic_mc: TIC with motion compensation
        tic_no_mc: TIC without motion compensation
        
    Returns:
        dict: Comprehensive fitting metrics
    """
    results = {}
    
    print("\nFitting TIC curves with lognormal model...")
    
    # Fit with MC
    print("  Fitting TIC with motion compensation...")
    params_mc, fitted_mc = fit_lognormal_curve(time_arr, tic_mc)
    
    # Fit without MC
    print("  Fitting TIC without motion compensation...")
    params_no_mc, fitted_no_mc = fit_lognormal_curve(time_arr, tic_no_mc)
    
    # Store parameters
    param_names = ['AUC', 'PE', 'TP', 'MTT', 'T0', 'mu', 'sigma', 'PE_loc']
    
    results['with_mc'] = {
        'params': dict(zip(param_names, params_mc)),
        'fitted_curve': fitted_mc
    }
    
    results['without_mc'] = {
        'params': dict(zip(param_names, params_no_mc)),
        'fitted_curve': fitted_no_mc
    }
    
    # Calculate fitting quality metrics
    if fitted_mc is not None:
        tic_mc_norm = (tic_mc - np.min(tic_mc)) / (np.max(tic_mc) - np.min(tic_mc) + 1e-10)
        
        ss_res_mc = np.sum((tic_mc_norm - fitted_mc) ** 2)
        ss_tot_mc = np.sum((tic_mc_norm - np.mean(tic_mc_norm)) ** 2)
        r2_mc = 1 - (ss_res_mc / (ss_tot_mc + 1e-10))
        
        rmse_mc = np.sqrt(mean_squared_error(tic_mc_norm, fitted_mc))
        corr_mc, _ = pearsonr(tic_mc_norm, fitted_mc)
        
        results['with_mc']['r2'] = r2_mc
        results['with_mc']['rmse'] = rmse_mc
        results['with_mc']['correlation'] = corr_mc
        results['with_mc']['residual_sum_squares'] = ss_res_mc
    else:
        results['with_mc']['r2'] = 0.0
        results['with_mc']['rmse'] = np.inf
        results['with_mc']['correlation'] = 0.0
        results['with_mc']['residual_sum_squares'] = np.inf
    
    if fitted_no_mc is not None:
        tic_no_mc_norm = (tic_no_mc - np.min(tic_no_mc)) / (np.max(tic_no_mc) - np.min(tic_no_mc) + 1e-10)
        
        ss_res_no_mc = np.sum((tic_no_mc_norm - fitted_no_mc) ** 2)
        ss_tot_no_mc = np.sum((tic_no_mc_norm - np.mean(tic_no_mc_norm)) ** 2)
        r2_no_mc = 1 - (ss_res_no_mc / (ss_tot_no_mc + 1e-10))
        
        rmse_no_mc = np.sqrt(mean_squared_error(tic_no_mc_norm, fitted_no_mc))
        corr_no_mc, _ = pearsonr(tic_no_mc_norm, fitted_no_mc)
        
        results['without_mc']['r2'] = r2_no_mc
        results['without_mc']['rmse'] = rmse_no_mc
        results['without_mc']['correlation'] = corr_no_mc
        results['without_mc']['residual_sum_squares'] = ss_res_no_mc
    else:
        results['without_mc']['r2'] = 0.0
        results['without_mc']['rmse'] = np.inf
        results['without_mc']['correlation'] = 0.0
        results['without_mc']['residual_sum_squares'] = np.inf
    
    # Calculate TIC variability
    results['with_mc']['cv'] = np.std(tic_mc) / (np.mean(tic_mc) + 1e-10)
    results['without_mc']['cv'] = np.std(tic_no_mc) / (np.mean(tic_no_mc) + 1e-10)
    
    print("  Done!")
    
    return results


def plot_tic_fitting_comparison(time_arr, tic_mc, tic_no_mc, fitting_results, 
                                output_path=None):
    """Create comprehensive TIC fitting comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: TIC Curves with Fits
    ax1 = axes[0, 0]
    
    ax1.plot(time_arr, tic_mc, 'g-', linewidth=2, alpha=0.7, label='MC (Raw)')
    ax1.plot(time_arr, tic_no_mc, 'r-', linewidth=2, alpha=0.7, label='No-MC (Raw)')
    
    if fitting_results['with_mc']['fitted_curve'] is not None:
        fitted_mc_scaled = (fitting_results['with_mc']['fitted_curve'] * 
                           (np.max(tic_mc) - np.min(tic_mc)) + np.min(tic_mc))
        ax1.plot(time_arr, fitted_mc_scaled, 'g--', linewidth=2,
                label=f"MC (Fit, R²={fitting_results['with_mc']['r2']:.3f})")
    
    if fitting_results['without_mc']['fitted_curve'] is not None:
        fitted_no_mc_scaled = (fitting_results['without_mc']['fitted_curve'] * 
                              (np.max(tic_no_mc) - np.min(tic_no_mc)) + np.min(tic_no_mc))
        ax1.plot(time_arr, fitted_no_mc_scaled, 'r--', linewidth=2,
                label=f"No-MC (Fit, R²={fitting_results['without_mc']['r2']:.3f})")
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title('TIC Curves with Lognormal Fits', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax2 = axes[0, 1]
    
    if fitting_results['with_mc']['fitted_curve'] is not None:
        tic_mc_norm = (tic_mc - np.min(tic_mc)) / (np.max(tic_mc) - np.min(tic_mc) + 1e-10)
        residuals_mc = tic_mc_norm - fitting_results['with_mc']['fitted_curve']
        ax2.plot(time_arr, residuals_mc, 'g-', linewidth=2, alpha=0.7, label='MC Residuals')
    
    if fitting_results['without_mc']['fitted_curve'] is not None:
        tic_no_mc_norm = (tic_no_mc - np.min(tic_no_mc)) / (np.max(tic_no_mc) - np.min(tic_no_mc) + 1e-10)
        residuals_no_mc = tic_no_mc_norm - fitting_results['without_mc']['fitted_curve']
        ax2.plot(time_arr, residuals_no_mc, 'r-', linewidth=2, alpha=0.7, label='No-MC Residuals')
    
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Residual', fontsize=12)
    ax2.set_title('Fitting Residuals', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Quality Metrics
    ax3 = axes[1, 0]
    
    metrics = ['r2', 'correlation']
    metric_names = ['R²', 'Correlation']
    
    mc_values = [fitting_results['with_mc'].get(m, 0) for m in metrics]
    no_mc_values = [fitting_results['without_mc'].get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, mc_values, width, label='With MC', color='green', alpha=0.7)
    ax3.bar(x + width/2, no_mc_values, width, label='Without MC', color='red', alpha=0.7)
    
    ax3.set_xlabel('Metric', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Fitting Quality Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_names)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.05])
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "TIC Fitting Summary\n" + "="*40 + "\n\n"
    
    summary_text += "Fitting Quality:\n"
    summary_text += f"  R² (MC):      {fitting_results['with_mc'].get('r2', 0):.4f}\n"
    summary_text += f"  R² (No-MC):   {fitting_results['without_mc'].get('r2', 0):.4f}\n"
    r2_imp = ((fitting_results['with_mc'].get('r2', 0) - fitting_results['without_mc'].get('r2', 0)) / 
             (fitting_results['without_mc'].get('r2', 1) + 1e-10)) * 100
    summary_text += f"  Improvement:  {r2_imp:+.1f}%\n\n"
    
    summary_text += f"  RMSE (MC):    {fitting_results['with_mc'].get('rmse', 0):.4f}\n"
    summary_text += f"  RMSE (No-MC): {fitting_results['without_mc'].get('rmse', 0):.4f}\n"
    rmse_imp = ((fitting_results['without_mc'].get('rmse', 0) - fitting_results['with_mc'].get('rmse', 0)) / 
               (fitting_results['without_mc'].get('rmse', 1) + 1e-10)) * 100
    summary_text += f"  Improvement:  {rmse_imp:+.1f}%\n\n"
    
    summary_text += "TIC Variability:\n"
    summary_text += f"  CV (MC):      {fitting_results['with_mc']['cv']:.4f}\n"
    summary_text += f"  CV (No-MC):   {fitting_results['without_mc']['cv']:.4f}\n"
    cv_imp = ((fitting_results['without_mc']['cv'] - fitting_results['with_mc']['cv']) / 
             (fitting_results['without_mc']['cv'] + 1e-10)) * 100
    summary_text += f"  Improvement:  {cv_imp:+.1f}%\n"
    
    ax4.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved TIC fitting comparison to {output_path}")
    
    plt.show()
    
    return fig


def generate_comprehensive_report(voi_results, tic_results, output_path='mc_evaluation_report.txt'):
    """Generate comprehensive text report."""
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("3D MOTION COMPENSATION EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        # VOI B-mode Similarity Section
        f.write("1. VOI B-MODE SIMILARITY METRICS\n")
        f.write("-"*70 + "\n")
        f.write("Measuring tissue appearance within VOI\n")
        f.write("(Higher values = VOI tracking same tissue)\n\n")
        
        metrics = ['correlation', 'ssim']
        metric_names = ['Pearson Correlation', 'SSIM']
        
        for metric, name in zip(metrics, metric_names):
            mc_values = voi_results['with_mc'][metric]
            no_mc_values = voi_results['without_mc'][metric]
            
            mean_mc = np.mean(mc_values)
            std_mc = np.std(mc_values)
            mean_no_mc = np.mean(no_mc_values)
            std_no_mc = np.std(no_mc_values)
            
            improvement = ((mean_mc - mean_no_mc) / (mean_no_mc + 1e-10)) * 100
            
            f.write(f"{name}:\n")
            f.write(f"  With MC:    {mean_mc:.4f} ± {std_mc:.4f}\n")
            f.write(f"  Without MC: {mean_no_mc:.4f} ± {std_no_mc:.4f}\n")
            f.write(f"  Improvement: {improvement:+.2f}%\n\n")
        
        # TIC Analysis
        f.write("\n2. TIME-INTENSITY CURVE ANALYSIS\n")
        f.write("-"*70 + "\n\n")
        
        f.write("Lognormal Fit Quality:\n")
        f.write(f"  R² (MC):      {tic_results['with_mc'].get('r2', 0):.4f}\n")
        f.write(f"  R² (No-MC):   {tic_results['without_mc'].get('r2', 0):.4f}\n")
        r2_imp = ((tic_results['with_mc'].get('r2', 0) - tic_results['without_mc'].get('r2', 0)) / 
                 (tic_results['without_mc'].get('r2', 1) + 1e-10)) * 100
        f.write(f"  Improvement:  {r2_imp:+.2f}%\n\n")
        
        f.write("TIC Variability:\n")
        f.write(f"  CV (MC):      {tic_results['with_mc']['cv']:.4f}\n")
        f.write(f"  CV (No-MC):   {tic_results['without_mc']['cv']:.4f}\n")
        cv_imp = ((tic_results['without_mc']['cv'] - tic_results['with_mc']['cv']) / 
                 (tic_results['without_mc']['cv'] + 1e-10)) * 100
        f.write(f"  Improvement:  {cv_imp:+.2f}%\n\n")
    
    print(f"\nComprehensive report saved to: {output_path}")


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """
    Example of how to use the modified evaluation functions.
    """
    # Assuming you have:
    # - bmode_volumes: (z, y, x, t) B-mode data
    # - ceus_volumes: (z, y, x, t) CEUS data
    # - base_mask: (z, y, x) segmentation mask
    # - motion_compensation_result: MotionCompensationResult object
    
    print("Example usage:")
    print("""
    # 1. Compute B-mode similarity comparison
    voi_results = compute_voi_bmode_similarity_comparison(
        bmode_volumes,
        base_mask,
        motion_compensation_result,
        reference_frame=0
    )
    
    # 2. Plot results
    plot_voi_bmode_similarity_comparison(
        voi_results,
        output_path='voi_similarity_comparison.png'
    )
    
    # 3. Compute TICs
    tic_mc = compute_tic_from_volumes(
        ceus_volumes,
        base_mask,
        motion_compensation_result,
        use_mc=True
    )
    
    tic_no_mc = compute_tic_from_volumes(
        ceus_volumes,
        base_mask,
        motion_compensation_result,
        use_mc=False
    )
    
    # 4. Evaluate TIC fitting
    time_arr = np.arange(len(tic_mc)) * frame_rate  # e.g., frame_rate = 1/15
    tic_results = evaluate_tic_fitting(time_arr, tic_mc, tic_no_mc)
    
    # 5. Plot TIC comparison
    plot_tic_fitting_comparison(
        time_arr, tic_mc, tic_no_mc, tic_results,
        output_path='tic_fitting_comparison.png'
    )
    
    # 6. Generate report
    generate_comprehensive_report(
        voi_results, tic_results,
        output_path='mc_evaluation_report.txt'
    )
    """)

if __name__ == "__main__":
    example_usage()