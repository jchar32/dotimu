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

    normalized_array = np.zeros((num_samples, n))
    for i in range(n):
        normalized_array[:, i] = np.interp(
            np.linspace(0, m, num_samples), np.arange(m), array[:, i]
        )
    return normalized_array


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
