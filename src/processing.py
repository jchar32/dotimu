from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import rfft, rfftfreq


def _get_filter_coefs(type: str, cutoff: float | List[float], fs: int, order: int = 2):
    """
    Get the coefficients for a Butterworth filter.

    Args:
        type (str): The type of the filter. Can be "low", "high", "band", or "stop".
        cutoff (float | List[float]): The cutoff frequency or frequencies.
        fs (int): The sampling rate.
        order (int, optional): The order of the filter. Defaults to 2.

    Returns:
        tuple: The numerator (b) and denominator (a) polynomials of the IIR filter.
    """

    b, a = signal.butter(
        order,
        cutoff,
        type,
        fs=fs,
    )
    return b, a


def filter_signal(
    data: pd.Series | pd.DataFrame | np.ndarray,
    cutoff: float | List[float],
    fs: int = 100,
    type: str = "low",
    order: int = 2,
    return_as: str = "same",
    new_col_names: List[str] | None = None,
):
    """
    Apply a digital filter to the data.

    Args:
        data (pd.Series | pd.DataFrame | np.ndarray): The data to filter.
        cutoff (int, optional): The cutoff frequency. Defaults to 100.
        fs (int, optional): The sampling rate. Defaults to 100.
        type (str, optional): The type of the filter. Can be "low" or "high". Defaults to "low".
        order (int, optional): The order of the filter. Defaults to 2.
        return_as (str): [NotImplemented] The type of object to return. Can be "same", "ndarray", "pd.dataframe", or "pd.series". Defaults to "same".
    Returns:
        np.ndarray: The filtered data in the same form as the input.
    """
    b, a = _get_filter_coefs(type=type, cutoff=cutoff, fs=fs, order=order)

    # assumes time axis is 0
    filtered_signal = signal.filtfilt(
        b, a, data, axis=0
    )  # returns as np.ndarray by default

    if return_as == "same":
        if isinstance(data, pd.DataFrame):
            filtered_signal = pd.DataFrame(
                filtered_signal, index=data.index, columns=data.columns
            )
        elif isinstance(data, pd.Series):
            filtered_signal = pd.Series(filtered_signal, index=data.index)
    elif return_as == "pd.dataframe":
        non_time_axis = np.argmin(filtered_signal.shape)
        col_name_list = [
            f"s{i}"
            for i in (
                range(filtered_signal.shape[non_time_axis])
                if filtered_signal.ndim > 1
                else range(1)
            )
        ]
        index = np.arange(0, filtered_signal.shape[0], 1)
        filtered_signal = pd.DataFrame(
            filtered_signal,
            index=index,
            columns=col_name_list if new_col_names is None else new_col_names,
        )

    return filtered_signal


def power_freq_spectrum(
    data: pd.Series | pd.DataFrame | np.ndarray, plotme: bool = False, fs: int = 120
):
    """
    Compute the power frequency spectrum of the data.

    Args:
        data (pd.Series | pd.DataFrame | np.ndarray): The data to compute the spectrum for.
        plotme (bool, optional): Whether to plot the spectrum. Defaults to False.

    Returns:
        A tuple (xf, yf) where xf is an array of the frequencies and yf is an array of the amplitudes (in real number format).
    """
    N = data.shape[0]
    yf = rfft(data, axis=0)
    xf = rfftfreq(N, 1 / fs)

    if plotme:
        plt.figure()
        plt.plot(xf, np.abs(yf))  # type: ignore
        plt.show()
    return xf, yf


def time_normalize_samples(array: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Normalize an  array of m samples by n signals to a preset number of samples (num_samples).

    Parameters
    ----------
    array : np.ndarray
        A 2D numpy array of shape (m, n), where m is the number of samples and n is the number of signals.
    num_samples : int
        The number of samples to which each signal should be normalized.

    Returns
    -------
    np.ndarray
        A 2D numpy array of shape (num_samples, n), where each signal has been normalized to have num_samples samples.

    Raises
    ------
    ValueError
        If m < 3 or n < 1.
    """
    m, n = array.shape
    if m < 3:
        raise ValueError("Must have more than 3 samples")
    if n < 1:
        raise ValueError("Must be aleast 1 signal to normalize")

    normalized_array = np.zeros((num_samples, n), dtype=np.float64)
    for i in range(n):
        normalized_array[:, i] = np.interp(
            np.linspace(0, m, num_samples), np.arange(m), array[:, i]
        )
    return normalized_array


def v_residual(nd_signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """V_residual caluclated from 1D signals normalized over some time period.
    Implementation of eq 1-4 in MacNeilage & Glasauer 2017 Front Comp Neursci doi: 10.3389/fncom.2017.00047

    Parameters
    ----------
    avg_signal : np.ndarray
        ensemble mean signal over all instances of the normalized signal. shape (n,) where n= number of normalized samples
    nd_signals : np.ndarray
        matrix of shape (n,m) where n=normalized sample number and m=number of instances of the normalized signal

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        v_res : np.ndarray and v_exp : np.ndarray where v_res is the residual variance and v_exp is the explained variance (R2)

    Raises
    ------
    ValueError
        avg_signal must have 1 dimension
        nd_signals must have 2 dimensions
    """

    avg_signal = np.nanmean(nd_signals, axis=-1)
    vres_out = np.full(avg_signal.shape, np.nan, dtype=np.float64)

    for i in range(avg_signal.shape[-1]):
        # Use variable naming as in MacNeilage & Glasauer 2017
        m = nd_signals[:, i, :]  # normalized waveforms in matrix
        md = np.mean(avg_signal[:, i])  # overall mean of the signals
        f = avg_signal[:, i]  # mean ensemble waveform

        ss_tot = (1 / (m.shape[-1])) * np.sum(np.square(m - md), axis=1)
        ss_res = (1 / (m.shape[-1])) * np.sum(np.square(m.T - f).T, axis=1)

        v_res = np.divide(ss_res, ss_tot)
        # v_exp = 1 - v_res
        vres_out[:, i] = v_res
    return v_res


def waveform_summary_metrics_over_trials(
    nd_waveforms: dict, start_trial_idx: int = 0, end_trial_idx: int | None = None
):
    (
        mean_,
        std_,
        median_,
        cov_,
        vres_,
        vres_mean_,
        phase_sd_,
        phase_sd_idx_,
        odd_even_noise_,
    ) = [[] for _ in range(9)]

    for t, trial in enumerate(list(nd_waveforms.keys())):
        data = nd_waveforms[trial][:, :, start_trial_idx:end_trial_idx]

        mean_.append(np.nanmean(data, axis=-1))
        std_.append(np.nanstd(data, axis=-1))
        median_.append(np.nanmedian(data, axis=-1))
        cov_.append(np.nanstd(data, axis=-1) / np.nanmean(data, axis=-1))
        vres_.append(v_residual(data))
        vres_mean_.append(np.nanmean(v_residual(data), axis=-1))

        phi, phase_sd_waveform_, phase_sd_index_ = lewek_phase_variability(data)
        phase_sd_.append(phase_sd_waveform_)
        phase_sd_idx_.append(phase_sd_index_)
        odd_even_noise_.append(
            np.nanmean(data[:, :, 1::2], axis=-1) - np.nanmean(data[:, :, ::2], axis=-1)
        )
    return (
        mean_,
        std_,
        median_,
        cov_,
        vres_,
        vres_mean_,
        phase_sd_,
        phase_sd_idx_,
        odd_even_noise_,
    )


def lewek_phase_variability(nd_waveforms: np.ndarray):
    """
    Calculate phase variability based on Lewek et al. 2006.

    This function calculates the phase variability of a given set of waveforms
    using the method described in the paper by Lewek et al. (2006). The phase
    variability is a measure of the variability in the phase angle across the
    gait cycle.

    Parameters:
    - nd_waveforms (np.ndarray): A 2D array of shape (n, m) representing the
        waveforms. Each row corresponds to a different time point, and each column
        corresponds to a different waveform.

    Returns:
    - phi (np.ndarray): A 2D array of shape (n-1, m) representing the phase angle
        for each waveform at each time point, excluding the first time point.
    - phi_sd (np.ndarray): A 1D array of shape (n-1,) representing the standard
        deviation of the phase angle across the gait cycle for each waveform.
    - phi_mean_sd (float): The mean of the phase standard deviations across all
        waveforms, representing the overall phase variability.

    References:
    - Lewek MD, et al. (2006). Gait & Posture, 24(4), 482-492.
        doi:10.1016/j.gaitpost.2005.06.003
    """

    theta_norm = (
        (2 * (nd_waveforms - np.min(nd_waveforms, axis=0)))
        / (np.max(nd_waveforms, axis=0) - np.min(nd_waveforms, axis=0))
    ) - 1

    theta_dot = np.diff(nd_waveforms, axis=0, prepend=0)
    theta_dot_norm = theta_dot / np.max(np.abs(theta_dot), axis=0)
    phi = np.unwrap(np.arctan2(theta_norm, theta_dot_norm), axis=0)
    phi = phi[1:, :]
    phi_sd = np.std(phi, axis=-1)
    phi_mean_sd = np.mean(phi_sd)
    return phi, phi_sd, phi_mean_sd


def remove_discontinuity(
    sig,
    idx,
    original_data,
    trailing_window=1000,
    percentile=[2.5, 97.5],
    threshold_multiplier=0.1,  # 1.5,
    in_degrees=True,
    angle_discont_limit=400,
):
    """
    Remove discontinuities in a signal.

    Parameters
    ----------
    sig : array_like
        The signal in which to remove discontinuities.
    idx : array_like
        Indices where discontinuities may occur.
    original_data : DataFrame
        The original data of the signal.

    Returns
    -------
    DataFrame
        The signal with discontinuities removed.

    """
    idx = np.append(idx, original_data.shape[0])

    # Copy the original data
    x_cont = original_data.copy(deep=True)

    # Calculate the thresholds for the test interval
    def calculate_threshold(percentile, threshold_multiplier):
        if percentile < 0:
            return percentile + (percentile * threshold_multiplier)
        else:
            return percentile - (percentile * threshold_multiplier)

    # Iterate over pairs of indices
    for d1, d2 in zip(idx[::1], idx[1::1]):
        # d1=idx[2]
        # d2=idx[3]
        # reupdate the derivative to more accurately reflect the corrections needed at each discontinuity
        sig = np.diff(x_cont)

        # Calculate the percentile of the trailing interval
        start = np.max([0, d1 - trailing_window])
        end = d1
        trailing_interval_percentiles = np.nanpercentile(
            x_cont.loc[start:end],
            percentile,
        )

        # Calculate the mean of the test interval
        test_interval_mean = np.nanmean(x_cont.loc[d1 + 1 : d2])
        # test_interval_mean = np.nanmean(original_data.loc[d1 + 1 : d2])

        # Check to get id of higher and lower percentile as they switch based on sign
        lower_perc_id = np.argmin(trailing_interval_percentiles)
        upper_perc_id = np.argmax(trailing_interval_percentiles)
        # Calculate the thresholds for the test interval
        lower_threshold = calculate_threshold(
            trailing_interval_percentiles[lower_perc_id], threshold_multiplier
        )
        upper_threshold = calculate_threshold(
            trailing_interval_percentiles[upper_perc_id], -threshold_multiplier
        )

        # If the test interval mean is within the thresholds, update d0 and continue
        if lower_threshold < test_interval_mean < upper_threshold:
            continue

        correction_dir = np.sign(sig[d1])
        correction_factor = (
            np.round(sig[d1] / 360) * 360 if np.abs(sig[d1] / 360) > 1.2 else 360
        )

        # Apply the correction to the test interval
        x_cont.loc[d1 + 1 : d2] -= correction_dir * correction_factor

    # Return the corrected signal
    return x_cont


if __name__ == "__main__":
    # test filter_signal
    fs = 120
    cutoff = 2
    order = 2
    type = "lowpass"
    duration = 4  # in seconds

    # Generate the time values
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Generate 5 sine waves of different frequencies
    frequencies = [1, 2, 4, 6]
    y = np.array(sum(np.sin(2 * np.pi * freq * t) for freq in frequencies))

    yfilt = filter_signal(y, cutoff, fs, type, order)

    plt.figure()
    plt.plot(t, y)
    plt.plot(t, yfilt)
    plt.show()

    # test power_freq_spectrum
    xf, yf = power_freq_spectrum(np.stack([y, yfilt], axis=1), plotme=True)
