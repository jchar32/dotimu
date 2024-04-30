import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

import quaternions as quat


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
        dict | pd.DataFrame: dict of dot objects or dataframe of dot data with the mean of each signal.
    """

    temp = {}
    meandotdata = {}
    for d in dotdata.keys():
        try:
            dotdata[d].keys()
        except AttributeError:
            for d in dotdata.keys():
                for filenum in range(len(dotdata[d])):
                    temp[filenum] = dotdata[d][filenum].mean(axis=0, numeric_only=True)
                meandotdata[d] = temp
                temp = {}
        else:
            for filenum in dotdata[d].keys():
                temp[filenum] = dotdata[d][filenum].mean(axis=0, numeric_only=True)
            meandotdata[d] = temp
            temp = {}

    return meandotdata


def ideal_ori() -> np.ndarray:
    """the orientation matrix of a perfectly calibrated accelerometer. This expects a sensor reading of + when pointing in the upward direction away from earths surface.

    Returns:
        np.ndarray: 3x3 matrix of the ideal orientation of the accelerometer.
    """
    return np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    )


def gather_calibration_data(
    data: dict,
    collected_order: dict = {
        "bias": 0,
        "xup": 1,
        "xdown": 2,
        "yup": 3,
        "ydown": 4,
        "zup": 5,
        "zdown": 6,
    },
) -> pd.DataFrame:
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
        cal_results = Calibrations(correction_matrix, accel_bias, gyro_bias[0, :])
        calib_data[id] = cal_results

    return calib_data


def apply_sensor_correction(dotdata: dict, cal: dict) -> dict:
    calibrated_data = copy.deepcopy(dotdata)
    for d in dotdata.keys():
        for i, data in enumerate(dotdata[d]):
            # apply calibration
            accel2cal = data.loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy()
            corrected_accel = (cal[d].matrix @ accel2cal.T).T + cal[d].accel_bias
            calibrated_data[d][i].loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]] = corrected_accel

            gyro2cal = data.loc[:, ["Gyr_X", "Gyr_Y", "Gyr_Z"]]
            corrected_gyro = gyro2cal + cal[d].gyro_bias
            calibrated_data[d][i].loc[:, ["Gyr_X", "Gyr_Y", "Gyr_Z"]] = (
                corrected_gyro.values
            )

    return calibrated_data


def apply_sensor2body(
    model_data, s2b, accel=True, gyro=True, mag=True, quaternion=False
):
    """applies a sensor to body calibration matrix to the dot sensor data. Assumes the sensor data is in the sensor frame and the calibration matrix is from the sensor frame to the body frame. The result is the sensor data in the body frame.

    Args:
        model_data (dict): keys are sensor locations and values are dot objects containing the sensor dataframes.
        s2b (dict): keys are sensor locations, values are 3x3 rotation matrices from sensor to body.
        accel (bool, optional): calibrate accelerometer data. Defaults to True.
        gyro (bool, optional): calibrate gyro data. Defaults to True.
        mag (bool, optional): calibrate mag data. Defaults to True.
        quaternion (bool, optional): calibrate quaternion data. Defaults to False.

    Raises:
        ValueError: must have equal number of sensor to body calibration matrices and sensor data arrays

    Returns:
        dict: calibrated data in same structure as input (model_data)
    """

    calibrated_data = copy.deepcopy(model_data)

    num_sensors = len(model_data.keys())
    if num_sensors != len(s2b.keys()):
        raise ValueError(
            f"Mismatch between number of sensors: {num_sensors} and number of sensor to body calibration matrices: {len(s2b.keys())}."
        )

    for s in calibrated_data.keys():
        print(f"Applying calibration to {s}")
        # check if signals are present
        accel = True and "Acc_X" in calibrated_data[s][0].columns
        gyro = True and "Gyr_X" in calibrated_data[s][0].columns
        mag = True and "Mag_X" in calibrated_data[s][0].columns
        quatern = True and "Quat_X" in calibrated_data[s][0].columns
        eul = True and "Eul_X" in calibrated_data[s][0].columns

        for data in calibrated_data[s]:
            if accel:
                acc_data = (
                    data.loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy() @ s2b[s].T
                )
                data.loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]] = acc_data
            if gyro:
                gyro_data = (
                    data.loc[:, ["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy() @ s2b[s].T
                )
                data.loc[:, ["Gyr_X", "Gyr_Y", "Gyr_Z"]] = gyro_data
            if mag:
                mag_data = (
                    data.loc[:, ["Mag_X", "Mag_Y", "Mag_Z"]].to_numpy() @ s2b[s].T
                )
                data.loc[:, ["Mag_X", "Mag_Y", "Mag_Z"]] = mag_data

            if quatern:
                Rq = quat.from_rotmat(s2b[s])
                ori = data.loc[:, ["Quat_W", "Quat_X", "Quat_Y", "Quat_Z"]].to_numpy()

                # looped since product function is not vectorized
                for i in range(ori.shape[0]):
                    ori[i, :] = quat.product(Rq, ori[i, :])

                data.loc[:, ["Quat_W", "Quat_X", "Quat_Y", "Quat_Z"]] = ori
    return calibrated_data


def get_sensor2body(trialsmap: dict, data: dict) -> dict:
    """Obtain functional calibration matrix for each sensor. The result is a rotation matrix from the sensor frame to the body frame (the segment the dot sensor is affixed to). Assumes 'data' is a dict of 7 dot sensors (pelvis and bilateral foot, shank, thigh).

    Args:
        trialsmap (dict): maping each calibration trial to a data file number. Mandatory movements for a 7 sensor (pelvis and bilateral foot, shank, thigh) are:
            1. Npose
            2. Lean forward at the hips
            3. Right leg air cycle (cycle leg as though on a bike)
            4. Left leg air cycle (cycle leg as though on a bike)

        data (dict): dict of dot sensor data with keys being the sensor locations and values being the dot objects containing the sensor dataframes.
        valid sensor location names:
            lfoot, rfoot, lshank, rshank, lthigh, rthigh, pelvis

    Returns:
        dict: sensor to body calibration matrices for each sensor in same structure as data dict.
    """

    sensors = list(data.keys())

    s2b = {}

    # npose -> find gravity vector
    nan_mat = np.full((3, 3), dtype=float, fill_value=np.nan)
    for s in sensors:
        s2b[s] = nan_mat.copy()
        s2b[s][-1, :] = __set_vertical_axis(data[s][trialsmap["npose"]])

    # pelvis temp ap axis. This is used to find the ML axis
    s2b["pelvis"][:, :] = __set_pelvis_axes(
        data["pelvis"][trialsmap["lean"]], s2b["pelvis"][-1, :]
    )

    # Functional Calibration for lower limbs
    for s in sensors:
        if s == "pelvis":
            continue
        if "r" in s:
            # temp = data[s].data[trialsmap["rcycle"]]
            s2b[s][:, :] = __set_func_ml_axis(
                data[s][trialsmap["rcycle"]], s2b[s][-1, :], "r"
            )
        else:
            # temp = data[s].data[trialsmap["lcycle"]]
            s2b[s][:, :] = __set_func_ml_axis(
                data[s][trialsmap["lcycle"]], s2b[s][-1, :], "l"
            )

    return s2b


def __set_vertical_axis(data):
    gvec = data.loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]].mean().to_numpy()
    gvec /= np.linalg.norm(gvec)

    return -1 * gvec


def __set_pelvis_axes(forward_lean_data, pelvis_vert_axis):
    # temp = data["pelvis"].data[trialsmap["lean"]]
    ap_temp = forward_lean_data.loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]].mean().to_numpy()
    ap_temp /= np.linalg.norm(ap_temp)
    ml = np.cross(ap_temp, pelvis_vert_axis)
    ml /= np.linalg.norm(ml)
    ap = np.cross(ml, pelvis_vert_axis)
    ap /= np.linalg.norm(ap)
    infsup = np.cross(ap, ml)
    infsup /= np.linalg.norm(infsup)
    return np.vstack([ap, ml, infsup])


def __find_temp_vec_4_ml_axis_dir_check(data, vertical_axis):
    # The estimated ml axis may not be pointing in the correct ml direction. To fix the axis direction so it points to the right of the body, we need to make a temporary vector from which we can use the gravity vector and the right hand rule to determine the correct direction.
    # During the cycling motion, most of the acceleration will be vertical with some in the ap and ml. We will use this as a quick way to get an extra temporary vector.
    temp_cycle_accel = data.loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]].mean().to_numpy()
    temp_cycle_accel /= np.linalg.norm(temp_cycle_accel)
    ml_direc_vec = np.cross(
        vertical_axis,
        temp_cycle_accel,
    )

    return ml_direc_vec


def __set_func_ml_axis(data, vertical_axis, side):
    # eigvector decomp to get gyr signal variance. Largest variance is ML axis
    gyr = data.loc[:, ["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy()
    _, evecs = np.linalg.eig(np.cov((gyr - np.mean(gyr, axis=0)).T))

    ml_direc_vec = __find_temp_vec_4_ml_axis_dir_check(data, vertical_axis)

    # flip the eigenvector if it is pointing in the wrong direction
    if not np.array_equal(np.sign(evecs[:, 0]), np.sign(ml_direc_vec)):
        evecs *= -1

    # so primary axis is always in positive wrt. to the sensor frame
    # if evecs[np.argmax(np.abs(evecs[:, 0])), 0] <= 0:
    #     evecs *= -1

    ml_axis = -1 * evecs[:, 0]

    ap_axis = np.cross(
        ml_axis,
        vertical_axis,
    )
    ap_axis /= np.linalg.norm(ap_axis)
    infsup = np.cross(ap_axis, ml_axis)
    infsup /= np.linalg.norm(infsup)
    return np.vstack([ap_axis, ml_axis, infsup])


def __flip_left_ml_axis(evecs, side):
    # TODO: remove later as this is unused.
    # flip ML axis for left side so all axis conventions match Right to Left
    if side == "l" and (evecs[-1] > 0):
        evecs *= -1
    return evecs


def find_common_start_time(data: dict) -> dict:
    num_files = len(data[list(data.keys())[0]])

    # collect the first time stamp for each trial on each sensor
    test = np.full((len(data.keys()), num_files), np.nan)
    for j, s in enumerate(data.keys()):
        for i, trial in enumerate(data[s]):
            test[j, i] = trial.loc[0, "SampleTimeFine"]

    latest_starttime = np.max(test, axis=0)  # the highest common timestamp

    # get the row index for the common time stamps for every sensor across each trial.
    index_to_trim_start = {s: [] for s in data.keys()}
    for s in data.keys():
        for i, trial in enumerate(data[s]):  # range(num_files):
            index_to_trim_start[s].append(
                np.where(trial.loc[:, "SampleTimeFine"] == latest_starttime[i])[0][0]
            )

    return index_to_trim_start


def sync_multi_dot(data, syncidx):
    syncd_data = copy.deepcopy(data)

    for s in syncd_data:
        for i, trial in enumerate(syncd_data[s]):
            trimmed_data = trial.iloc[syncidx[s][i] :, :]
            trimmed_data.reset_index(inplace=True)
            syncd_data[s][i] = trimmed_data
    return syncd_data
