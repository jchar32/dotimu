import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Calibrations:
    matrix: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray


def mean_data(dotdata):
    temp = {}
    meandotdata = {}
    for d in dotdata.keys():
        for filenum in dotdata[d].data.keys():
            temp[str(filenum)] = (
                dotdata[str(d)].data[str(filenum)].mean(axis=0, numeric_only=True)
            )
        meandotdata[str(d)] = temp
    return meandotdata


def ideal_ori():
    return np.array(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    )


def gather_calibration_data(data):
    bias = pd.DataFrame(data["0"]).T
    xup = pd.DataFrame(data["1"]).T
    xdown = pd.DataFrame(data["2"]).T
    yup = pd.DataFrame(data["3"]).T
    ydown = pd.DataFrame(data["4"]).T
    zup = pd.DataFrame(data["5"]).T
    zdown = pd.DataFrame(data["6"]).T
    return pd.concat([bias, xup, xdown, yup, ydown, zup, zdown], axis=0)


def leastsquare_calibration(measured, ideal):
    return np.linalg.inv(measured.T @ measured) @ measured.T @ ideal


def get_calibration(data):
    calib_data = {}
    for d in data.keys():
        print(f"Calculating calibration for sensor {d}")
        id = str(d)
        cali_data = gather_calibration_data(data[d])

        # gather accelerometer data
        accel_data = cali_data.loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy()
        accel_data = np.append(accel_data, np.ones((accel_data.shape[0], 1)), axis=1)

        # calculate correction and bias in a least squares sense
        X = leastsquare_calibration(accel_data[1:, :], ideal_ori())
        # X = np.linalg.inv(a.T @ a) @ a.T @ ideal_ori()

        correction_matrix = X[:3, :]
        accel_bias = X[3, :]

        gyro_bias = cali_data.loc[:, ["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy()

        calib_data[id] = Calibrations(correction_matrix, accel_bias, gyro_bias[0, :])
        # calibrations[id] = (correction_matrix, accel_bias, gyro_bias[0, :])
    return calib_data
