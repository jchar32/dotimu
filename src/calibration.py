import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

@dataclass
class Calibrations:
    matrix: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray


def mean_dot_signals(dotdata: dict) -> dict:
    """_summary_

    Args:
        dotdata (dict | pd.DataFrame): dict of dot objects or datarfame of dot data of samples x signals

    Returns:
        dict | pd.DataFrame: dict of dot objects or datarfame of dot data with the mean of each signal.
    """

    temp = {}
    meandotdata = {}
    for d in dotdata.keys():
        for filenum in dotdata[d].data.keys():
            temp[filenum] = (
                dotdata[d].data[filenum].mean(axis=0, numeric_only=True)
            )
        meandotdata[d] = temp
    return meandotdata


def ideal_ori() -> np.ndarray:
    """the orientation matrix of a perfectly calibrated accelerometer.

    Returns:
        np.ndarray: 3x3 matrix of the ideal orientation of the accelerometer.
    """
    return np.array(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    )


def gather_calibration_data(data: dict,
                            collected_order: dict = {"bias": 0,
                                                     "xup": 1,
                                                     "xdown": 2,
                                                     "yup": 3,
                                                     "ydown": 4,
                                                     "zup": 5,
                                                     "zdown": 6}) -> pd.DataFrame:
    """compile individual dataframes that were collected for each calibration step.

    ASSUMPTION - Data were collected in the following order:
        bias, xup, xdown, yup, ydown, zup, zdown

        Suggest approximately 3 sections in each position.
        Procedure should be done with sensors in a calibration box with right angles.


    Args:
        data (dict): dict containing a single dot sensor's data for each calibration step.
        collected_order (dict): key value pairs of sensor position (e.g., xup) and the file number (e.g., 1) that it was collected in.
        "bias" can be "None" if no bias was collected as this can be taken from any of the other positions. Retaining this option  allows for a long bias collection time if desired.

    Returns:
        pd.DataFrame: dataframe with each row correspoding to a collected calibration orientation.
    """
    if collected_order["bias"] is None:
        bias = pd.DataFrame(data[collected_order["xup"]]).T

    bias = pd.DataFrame(data[collected_order["bias"]]).T
    xup = pd.DataFrame(data[collected_order["xup"]]).T
    xdown = pd.DataFrame(data[collected_order["xdown"]]).T
    yup = pd.DataFrame(data[collected_order["yup"]]).T
    ydown = pd.DataFrame(data[collected_order["ydown"]]).T
    zup = pd.DataFrame(data[collected_order["zup"]]).T
    zdown = pd.DataFrame(data[collected_order["zdown"]]).T

    return pd.concat([bias, xup, xdown, yup, ydown, zup, zdown], axis=0)


def leastsquare_calibration(measured: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    """calculate the optimal correction matrix in a least square sense.
    Following  [w * wt]^-1 * w * x = y

    Args:
        measured (np.array): 6x4 matrix of accelerometer data (xup, xdown, yup, ydown, zup, zdown) by (x ,y z,ones)
        ideal (_type_): 3x4 matrix of the ideal orientation of the accelerometer.

    Returns:
        np.ndarray: correction matrix [:3,:] and bias [3,:]
    """
    return np.linalg.inv(measured.T @ measured) @ measured.T @ ideal


def ori_and_bias(data: dict) -> dict:
    """Gather the specific files for each calibration orientation (xup, yup ect) then calculate the correction matrix and bias for each sensor.

    Args:
        data (dict): dict containing a single dot sensor's data for each calibration step.

    Returns:
        dict: dict for each sensor with a Calibration object containing a matrix, accel_bias, and gyro_bias.
    """
    calib_data = {}
    for id in data.keys():
        print(f"Calculating calibration for sensor {id}")

        cali_data = gather_calibration_data(data[id])

        # gather accelerometer data
        accel_data = cali_data.loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy()
        accel_data = np.append(accel_data, np.ones((accel_data.shape[0], 1)), axis=1)

        # calculate correction and bias in a least squares sense
        X = leastsquare_calibration(accel_data[1:, :], ideal_ori())

        correction_matrix = X[:3, :]
        accel_bias = X[3, :]

        gyro_bias = cali_data.loc[:, ["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy()

        calib_data[id] = Calibrations(correction_matrix, accel_bias, gyro_bias)

    return calib_data
