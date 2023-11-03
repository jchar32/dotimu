import numpy as np
import pandas as pd
from dataclasses import dataclass
import copy

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
    """the orientation matrix of a perfectly calibrated accelerometer. This expects a sensor reading of + when pointing in the upward direction away from earths surface.

    Returns:
        np.ndarray: 3x3 matrix of the ideal orientation of the accelerometer.
    """
    return np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
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

        unit_correction = 1
        if np.linalg.norm(accel_data[:, :-1], axis=1).mean() > 1.5:
            unit_correction = 9.80994  # local gravity

        # calculate correction and bias in a least squares sense
        X = leastsquare_calibration(accel_data[1:, :], ideal_ori() * unit_correction)

        correction_matrix = X[:3, :]
        accel_bias = X[3, :]

        gyro_bias = cali_data[["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy()

        calib_data[id] = Calibrations(correction_matrix, accel_bias, gyro_bias[0, :])

    return calib_data

def apply_sensor_correction(dotdata: dict, cal: dict) -> dict:
    calibrated_data = copy.deepcopy(dotdata)

    for d in dotdata.keys():
        for filenum in dotdata[d].data.keys():
            # apply calibration
            accel2cal = dotdata[d].data[filenum].loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy()
            corrected_accel = (
                (cal[d].matrix @ accel2cal.T).T + cal[d].accel_bias
            )
            calibrated_data[d].data[filenum].loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]] = corrected_accel

            gyro2cal = dotdata[d].data[filenum].loc[:, ["Gyr_X", "Gyr_Y", "Gyr_Z"]]
            corrected_gyro = (gyro2cal + cal[d].gyro_bias)
            calibrated_data[d].data[filenum].loc[:, ["Gyr_X", "Gyr_Y", "Gyr_Z"]] = corrected_gyro.values

    return calibrated_data


def sensor2body(trialsmap: dict, data: dict) -> dict:
    """_summary_

    Args:
        trialsmap (dict): maping each calibration trial to a data file number. Mandatory movements for a 7 sensor (pelvis and bilateral foot, shank, thigh) are:
            Npose
            Lean forward at the hips
            right leg air cycle (cycle leg as though on a bike)
            left leg air cycle (cycle leg as though on a bike)

        data (dict): dict of dot sensor data with keys being the sensor locations and values being the dot objects containing the sensor dataframes.
        valid sensor location names:
            lfoot, rfoot, lshank, rshank, lthigh, rthigh, pelvis

    Returns:
        dict: sensor to body calibration matrices for each sensor in same structure as data dict.
    """

    sensors = list(data.keys())

    s2b = {}

    # npose -> find gravity vector
    for s in sensors:
        temp = data[s].data[trialsmap["npose"]]
        gvec = temp[["Acc_X", "Acc_Y", "Acc_Z"]].mean().to_numpy()
        s2b[s] = np.full((3, 3), dtype=float, fill_value=np.nan)
        s2b[s][-1, :] = -1 * (gvec / np.linalg.norm(gvec))

    # pelvis
    temp = data["pelvis"].data[trialsmap["lean"]]
    gvec_ap = temp[["Acc_X", "Acc_Y", "Acc_Z"]].mean().to_numpy()
    gvec_ap /= np.linalg.norm(gvec_ap)

    s2b["pelvis"][1, :] = np.cross(gvec_ap, s2b["pelvis"][-1, :])

    s2b["pelvis"][0, :] = np.cross(s2b["pelvis"][1, :], s2b["pelvis"][-1, :])

    # Functional Calibration for lower limbs
    for s in sensors:
        if s == "pelvis":
            continue
        if 'r' in s:
            temp = data[s].data[trialsmap["rcycle"]]
        else:
            temp = data[s].data[trialsmap["lcycle"]]

        gyr = temp[["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy()
        _, evecs = np.linalg.eig(np.cov((gyr - np.mean(gyr)).T))

        # flip ML axis for left side so all axes match
        if 'l' in s and (evecs[-1, 0] > 0):
            s2b[s][1, :] = -1 * evecs[:, 0].T
        s2b[s][1, :] = evecs[:, 0].T

        # AP axis
        s2b[s][0, :] = np.cross(s2b[s][1, :], s2b[s][-1, :])

    return s2b
