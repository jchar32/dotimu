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

    # TODO: consider implementing a return_as option to return the data in a different format (pd.dataframe, pd.series, or np.ndarray)
    # if return_as == "same":
    #     if isinstance(data, pd.DataFrame):
    #         filtered_signal = pd.DataFrame(filtered_signal, index=data.index, columns=data.columns)
    #     elif isinstance(data, pd.Series):
    #         filtered_signal = pd.Series(filtered_signal, index=data.index)
    # elif return_as == "pd.dataframe":
    #     filtered_signal = pd.DataFrame(filtered_signal, index=data.index, columns=data.columns)

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
