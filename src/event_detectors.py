from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal

from processing import filter_signal


@dataclass
class GaitEvents:
    hs: np.ndarray
    midstance: np.ndarray
    to: np.ndarray
    midswing: np.ndarray


def mariani(data: pd.DataFrame, midswing: np.ndarray, negpeak_idx: np.ndarray):
    """calculates gait events based on the Mariani et al 2021 paper.

    # Mariani B, Rouhani H, Crevoisier X, Aminian K.
    # Quantitative estimation of foot-flat and stance phase of gait using foot-worn inertial sensors. Gait Posture. 2013 Feb;
    # 37(2):229-34. doi: 10.1016/j.gaitpost.2012.07.012.

    Args:
        data (pd.DataFrame): acceleration and angular rate data collected from a foot-mounted imu.
        events (GaitEvents): events class to deposit gait events
        negpeak_idx (np.ndarray): indices of the negative peaks in the mediolateral gyroscope data that approximate heel strike and toe-off.

    Returns:
        GaitEvents: events is returned populated with indices of gait events.
    """
    # TODO: bring in only the accel and gyro data already filtered, instead of sending full dataframe in.
    # generate the signals needed for this algo
    filtered_accel = filter_signal(
        data.loc[:, "Acc_X":"Acc_Z"], cutoff=17, fs=120, type="lowpass", order=2
    )

    filtered_gyro = filter_signal(
        data.loc[:, "Gyr_X":"Gyr_Z"], cutoff=17, fs=120, type="lowpass", order=2
    )

    accel_norm = np.linalg.norm(filtered_accel, axis=1)
    gyro_norm = np.linalg.norm(filtered_gyro, axis=1)

    # loop from the 2nd midswing event to the end (eliminates odd first strides that are common after a restart of walking)
    hs = np.array([], dtype=int)
    midstance = np.array([], dtype=int)
    to = np.array([], dtype=int)
    for stride in range(1, midswing.shape[0] - 1):
        startsample = midswing[stride]
        endsample = midswing[stride + 1]
        current_negpeak_hs = negpeak_idx[np.argmax(negpeak_idx > startsample)]
        current_negpeak_to = negpeak_idx[int(np.argmin(negpeak_idx < endsample)) - 1]

        # 1. HS detection
        try:
            k3_tmp = signal.find_peaks(
                -accel_norm[current_negpeak_hs - 15 : current_negpeak_hs + 5],
                prominence=1,
            )
            k3 = int(
                (
                    k3_tmp[0][np.argmax(k3_tmp[1]["prominences"])]
                    + current_negpeak_hs
                    - 15
                )
            )
            hs = np.append(hs, k3)

        except ValueError:
            k6 = int(
                np.argmax(np.diff(gyro_norm[startsample : current_negpeak_hs + 10]))
            )
            hs = np.append(hs, k6 + startsample)

        # 2. TO detection
        try:
            k22_tmp = signal.find_peaks(
                accel_norm[
                    current_negpeak_to - 15 : np.min(
                        [endsample, current_negpeak_to + 15]
                    )
                ],
                prominence=1,
            )
            k22 = int(
                (
                    k22_tmp[0][np.argmax(k22_tmp[1]["prominences"])]
                    + current_negpeak_to
                    - 15
                )
            )
            to = np.append(to, k22)

        except ValueError:
            to = np.append(to, current_negpeak_to)

        # 3. Midstance detection
        # this is not calculated in the paper. I have implemented a ZUPT approach loosely based on [Rebula et al 2013 Gait & Posture].
        gyro_thresh = 1.7  # rad/s
        accel_thresh = 1.5  # g
        if gyro_norm[midswing].mean() > 20:
            gyro_thresh *= 57.29578
        if accel_norm[hs].mean() > 5:
            accel_thresh *= 9.81

        try:
            zupt_tmp = (gyro_norm[hs[-1] : to[-1]] < gyro_thresh) & (
                accel_norm[hs[-1] : to[-1]] < accel_thresh
            )
            mid_zupt = int(
                (np.where(zupt_tmp)[0][-1] - np.where(zupt_tmp)[0][0]) / 2
                + (np.where(zupt_tmp)[0][0] + hs[-1])
            )
            midstance = np.append(midstance, mid_zupt)
        except (ValueError, IndexError):
            mid_zupt = int((to[-1] - hs[-1]) / 2 + hs[-1])
            midstance = np.append(midstance, mid_zupt)

    events = GaitEvents(hs=hs, to=to, midstance=midstance, midswing=midswing)

    return events


def roth():
    pass


def zeni():
    pass


def midswing_peak(gyroml: np.ndarray, negpeak_idx: np.ndarray, min_peak_dist: int = 50):
    """Detects midswing peaks in the gyro mediolateral axis signal.

    Args:
        gyroml (np.ndarray): The gyro mediolateral axis signal.
        negpeak_idx (np.ndarray): The indices of negative peaks (approximations of hs and to) in the gyro mediolateral axis signal signal.
        min_peak_dist (int, optional): The minimum distance between negative peaks in the gyro mediolateral signal. Defaults to 50.

    Returns:
        Tuple[np.ndarray, Dict[str, np.ndarray]]: A tuple containing the indices of midswing peaks and the properties of the peaks.
    """
    idx_tmp, props_tmp = signal.find_peaks(
        gyroml,
        prominence=1,
        distance=min_peak_dist,
        height=-0.5 * gyroml[negpeak_idx].mean(),
    )
    return idx_tmp, props_tmp


def index_gyro_negpeaks(data: pd.DataFrame | pd.Series | np.ndarray, mlaxis: int = 1):
    """Detects the indices of negative peaks in the mediolateral gyroscope axis signal.

    Args:
        data (pd.DataFrame | pd.Series | np.ndarray): The input data containing gyroscope readings.
            If `data` is a DataFrame, it should contain gyroscope data with column names "Gyr_X", "Gyr_Y", and "Gyr_Z".
            If `data` is a Series or ndarray, it should contain gyroscope data.
        mlaxis (int, optional): The axis along which to detect negative peaks. Defaults to 1.

    Returns:
        np.ndarray: An array of indices corresponding to the detected negative peaks.
    """
    if isinstance(data, pd.DataFrame):
        gyro = data.loc[:, "Gyr_X":"Gyr_Z"].to_numpy()
        gyroml = gyro[:, mlaxis]
    elif isinstance(data, pd.Series):  # pd.series or ndarray
        gyro = data
        gyroml = data.to_numpy()
    elif isinstance(data, np.ndarray):
        if data.ndim > 1:
            gyro = data
            gyroml = data[:, mlaxis]
        else:
            gyroml = data

    # set initial dynamic threshold for peaks
    negpeak_thresh_tmp = (gyroml[gyroml < 0]).mean() * 1.75

    # check peaks using initial threshold
    idx_tmp, props_tmp = signal.find_peaks(
        -gyroml, prominence=2, height=negpeak_thresh_tmp
    )

    # use mean of peak heights as new threshold
    negpeak_thresh = props_tmp["peak_heights"].mean()

    # use mean of peak distances as new distance between peaks
    frames_between_peaks = np.diff(idx_tmp).mean().round().astype(int)

    # find peaks using new threshold and distance between peaks
    idx, _ = signal.find_peaks(
        -gyroml,
        prominence=2,
        height=-negpeak_thresh * 0.75,
        distance=int(frames_between_peaks * 0.75),
    )

    return idx, negpeak_thresh, frames_between_peaks


def gait_events(
    data: pd.DataFrame, method: str = "mariani", fs: int = 120, gyro_ml_axis=1
):
    """main function for calculating gait events using several common algorithms.

    Args:
        data (pd.DataFrame): The input data containing sensor measurements. Must at least contain columns "Gyr_X", "Gyr_Y", "Gyr_Z", "Acc_X", "Acc_Y", and "Acc_Z".
        method (str, optional): The method to use for calculating gait events. Defaults to "mariani".
        fs (int, optional): The sampling frequency of the sensor data. Defaults to 120.
        gyro_ml_axis (int, optional): The index of the gyro mediolateral axis. Defaults to index 1.

    Returns:
        events (GaitEvents): A class containing the hs, to, midswing, and midstance indices in np.ndarrays.
    """
    # prefilter signals for efficieny
    filtered_gyro = filter_signal(
        data.loc[:, "Gyr_X":"Gyr_Z"], cutoff=10, fs=120, type="lowpass", order=2
    )

    # filtered_accel = filter_signal(
    #     data.loc[:, "Acc_X":"Acc_Z"], cutoff=10, fs=120, type="lowpass", order=2
    # )

    # 1. get a rough estimate of swing events for segmenting the data and looking closer at individual stride events
    # 1a.detect negative peaks in gyro ml axis
    negpeak_idx, negpks_thresh, frames_between_pks = index_gyro_negpeaks(
        filtered_gyro.iloc[:, gyro_ml_axis].to_numpy(), mlaxis=gyro_ml_axis
    )

    # 1b. detect midswing peaks
    midswing, midswing_pk_props = midswing_peak(
        gyroml=filtered_gyro.iloc[:, gyro_ml_axis].to_numpy(),
        negpeak_idx=negpeak_idx,
        min_peak_dist=frames_between_pks,
    )

    events = mariani(data, midswing, negpeak_idx)
    return events
