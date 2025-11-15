import numpy as np
from scipy.optimize import curve_fit

def bolus_lognormal(x, auc, mu, sigma, t0):
    with np.errstate(divide='ignore', invalid='ignore'):
        shifted = x - t0
        result = (auc / (shifted * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(shifted) - mu) ** 2) / (2 * sigma ** 2))
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def fit_lognormal_curve(time, curve):
    """
    Fit a log-normal distribution to the given curve.
    
    Args:
        time (np.ndarray): The time array corresponding to the curve.
        curve (np.ndarray): The curve data to fit.
    
    Returns:
        tuple: Fitted parameters (auc, pe, tp, mtt, t0, mu, sigma, pe_loc).
    """
    curve -= np.amin(curve)  # Shift to start at zero
    if np.amax(curve) == 0:
        print("Curve is constant, cannot normalize.")
        return tuple(np.nan for _ in range(8))
    curve = curve / np.amax(curve)  # Normalize

    auc_guess = np.sum(curve) * (time[1] - time[0])
    mu_guess = np.log(np.argmax(curve))
    sigma_guess = 0.5
    t0_guess = time[np.argmax(curve)] * 0.15

    try:
        params, _ = curve_fit(
            bolus_lognormal,
            time,
            curve,
            p0=(auc_guess, mu_guess, sigma_guess, t0_guess),
            bounds=([0., 0., 0.01, 0.], [np.inf, np.inf, 5.0, time[-1]]),
            method='trf',
            maxfev=10000  # Increase evaluations
        )
    except Exception as e:
        print(f"Error fitting curve: {e}")
        return tuple(np.nan for _ in range(8))

    auc, mu, sigma, t0 = params
    mtt = np.exp(mu + sigma**2 / 2)
    tp = np.exp(mu - sigma**2)
    fitted_curve = bolus_lognormal(time, *params)
    pe = np.max(fitted_curve)
    pe_loc = np.argmax(fitted_curve)

    return auc, pe, tp, mtt, t0, mu, sigma, pe_loc