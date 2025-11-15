import numpy as np
from typing import Tuple
from numpy.matlib import repmat

NUM_FOURIER_POINTS = 8192

def compute_hanning_power_spec(rf_data: np.ndarray, start_frequency: int, end_frequency: int, 
                            sampling_frequency: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the power spectrum of 3D spatial RF data using a Hanning window.
    Args:
        rfData (np.ndarray): 3D RF data from the ultrasound volume (n lateral lines x m axial samples x l elevational lines).
        startFrequency (int): lower bound of the frequency range (Hz).
        endFrequency (int): upper bound of the frequency range (Hz).
        samplingFrequency (int): sampling frequency of the RF data (Hz).
    Returns:
        Tuple: frequency range and power spectrum.
    """
    # Create Hanning Window Function for the axial dimension
    unrm_wind = np.hanning(rf_data.shape[0])
    wind_func_computations = unrm_wind * np.sqrt(len(unrm_wind) / sum(np.square(unrm_wind)))
    wind_func = repmat(
        wind_func_computations.reshape((rf_data.shape[0], 1)), 1, rf_data.shape[1]
    )

    # Frequency Range
    frequency = np.linspace(0, sampling_frequency, NUM_FOURIER_POINTS)
    f_low = round(start_frequency * (NUM_FOURIER_POINTS / sampling_frequency))
    f_high = round(end_frequency * (NUM_FOURIER_POINTS / sampling_frequency))
    freq_chop = frequency[f_low:f_high]

    # Get PS
    if rf_data.ndim == 3:
        power_spectra = []
        for i in range(rf_data.shape[2]):
            fft = np.square(
                abs(np.fft.fft(np.transpose(np.multiply(rf_data[:,:,i], wind_func)), NUM_FOURIER_POINTS) * rf_data[:,:,i].size)
            )
            full_ps = np.mean(fft, axis=0)
            power_spectra.append(full_ps[f_low:f_high])
        ps = np.mean(power_spectra, axis=0)
    elif rf_data.ndim == 2:
        fft = np.square(
            abs(np.fft.fft(np.transpose(np.multiply(rf_data, wind_func)), NUM_FOURIER_POINTS) * rf_data.size)
        )
        full_ps = np.mean(fft, axis=0)
        ps = full_ps[f_low:f_high]
    else:
        raise ValueError("Invalid RF data dimensions. Expected 2D or 3D data.")

    return freq_chop, ps
