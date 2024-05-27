import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import warnings

import quaternions as quat
import orientation as ori


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
            if data is None:
                continue
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


def set_frame_to_horizontal(data: dict, static_trial_num: int):
    reset_data = copy.deepcopy(data)
    rotation_reset = {}
    # get rotation from sensor-on-body (in segment frame) to horizontal plane
    for s in reset_data.keys():
        if reset_data[s] is None:
            continue
        if reset_data[s][static_trial_num] is None:
            continue

        # get the mean of the accelerometer data for the static trial
        static_trial = reset_data[s][static_trial_num]
        static_accel = (
            static_trial.loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]].mean().to_numpy()
        )
        static_accel /= np.linalg.norm(static_accel)

        # get the rotation matrix from the sensor frame to the horizontal plane

        R = quat.to_rotmat(quat.from_rpy(ori.static_tilt(static_accel))).T

        # apply the rotation to all the trials
        for i, trial in enumerate(reset_data[s]):
            accel = trial.loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy()
            accel = (R @ accel.T).T
            reset_data[s][i].loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]] = accel

            gyro = trial.loc[:, ["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy()
            gyro = (R @ gyro.T).T
            reset_data[s][i].loc[:, ["Gyr_X", "Gyr_Y", "Gyr_Z"]] = gyro

            if "Mag_X" in trial.columns:
                mag = trial.loc[:, ["Mag_X", "Mag_Y", "Mag_Z"]].to_numpy()
                mag = (R @ mag.T).T
                reset_data[s][i].loc[:, ["Mag_X", "Mag_Y", "Mag_Z"]] = mag
        rotation_reset[s] = R
    return reset_data, rotation_reset


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
    print("Applying sensor to body calibration")

    for s in calibrated_data.keys():
        if calibrated_data[s] is None:
            continue
        for data in calibrated_data[s]:
            if data is None:
                continue
            # check if signals are present
            accel = True and "Acc_X" in data.columns
            gyro = True and "Gyr_X" in data.columns
            mag = True and "Mag_X" in data.columns
            quatern = True and "Quat_X" in data.columns
            eul = True and "Eul_X" in data.columns

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
            if eul:
                eul_data = (
                    data.loc[:, ["Eul_X", "Eul_Y", "Eul_Z"]].to_numpy() @ s2b[s].T
                )
                data.loc[:, ["Eul_X", "Eul_Y", "Eul_Z"]] = eul_data
            if quatern:
                Rq = quat.from_rotmat(s2b[s])
                ori = data.loc[:, ["Quat_W", "Quat_X", "Quat_Y", "Quat_Z"]].to_numpy()

                # looped since product function is not vectorized
                for i in range(ori.shape[0]):
                    ori[i, :] = quat.product(Rq, ori[i, :])

                data.loc[:, ["Quat_W", "Quat_X", "Quat_Y", "Quat_Z"]] = ori
    return calibrated_data


def get_sensor2body(
    trialsmap: dict, data: dict, shank_imu_placement="anterior"
) -> dict:
    """
    Obtain functional calibration matrix for each sensor.

    The result is a rotation matrix from the sensor frame to the body frame (the segment the dot sensor is affixed to).
    Assumes 'data' is a dict of 7 dot sensors (pelvis and bilateral foot, shank, thigh).

    Parameters:
    -----------
        trialsmap (dict): Mapping each calibration trial to a data file number. Mandatory movements for a 7 sensor (pelvis and bilateral foot, shank, thigh) are:
            1. Npose
            2. Lean forward at the hips
            3. Right leg air cycle (cycle leg as though on a bike)
            4. Left leg air cycle (cycle leg as though on a bike)

        data (dict): Dict of dot sensor data with keys being the sensor locations and values being the dot objects containing the sensor data frames.
            Valid sensor location names:
                - lfoot
                - rfoot
                - lshank
                - rshank
                - lthigh
                - rthigh
                - pelvis
        shank_placement (str): Placement of the shank sensor. Options are "anterior" or "lateral". Defaults to "anterior". If "anterior", the shank sensor is placed on the anterior side of the shank. If "lateral", the shank sensor is placed on the lateral side of the shank. This is used to determine if the calibrated axes need to be flipped to maintain consistency of the reference frames between limbs and across sensors.
    Returns:
    --------
        dict: Sensor to body calibration matrices for each sensor in the same structure as the data dict.
    """

    sensors = list(data.keys())

    s2b = {}

    for s in sensors:
        if data[s] is None:
            s2b[s] = np.full((3, 3), dtype=float, fill_value=np.nan)
            continue
        s2b[s] = np.full((3, 3), dtype=float, fill_value=np.nan)
        s2b[s][-1, :] = __set_vertical_axis(data[s][trialsmap["npose"]])

    # pelvis temp ap axis. This is used to find the ML axis
    if data["pelvis"] is not None:
        if data["pelvis"][trialsmap["lean"]] is not None:
            s2b["pelvis"][:, :] = __set_pelvis_axes(
                data["pelvis"][trialsmap["lean"]], s2b["pelvis"][-1, :]
            )

    # Functional Calibration for lower limbs
    for s in sensors:
        if (
            data[s] is None
            or data[s][trialsmap["rcycle"]] is None
            or data[s][trialsmap["lcycle"]] is None
            or s == "pelvis"
        ):
            continue

        if "r" in s:
            gyr = (
                data[s][trialsmap["rcycle"]]
                .loc[:, ["Gyr_X", "Gyr_Y", "Gyr_Z"]]
                .to_numpy()
            )
            s2b[s][:, :] = __set_func_ml_axis(gyr, s2b[s][-1, :], negate_axis=False)
            # Anteroposterior axis
            s2b[s][0, :] = __axis_cross_product(s2b[s][1, :], s2b[s][-1, :])
            # orthog inf_sup axis
            s2b[s][-1, :] = __axis_cross_product(s2b[s][0, :], s2b[s][1, :])

        elif "l" in s:
            gyr = (
                data[s][trialsmap["lcycle"]]
                .loc[:, ["Gyr_X", "Gyr_Y", "Gyr_Z"]]
                .to_numpy()
            )
            if "shank" in s and shank_imu_placement == "anterior":
                # anterior shank imu placements do not need to their ml axis flipped.
                s2b[s][:, :] = __set_func_ml_axis(gyr, s2b[s][-1, :], negate_axis=False)
                # Anteroposterior axis
                s2b[s][0, :] = __axis_cross_product(s2b[s][1, :], s2b[s][-1, :])
                # orthog inf_sup axis
                s2b[s][-1, :] = __axis_cross_product(s2b[s][0, :], s2b[s][1, :])
                continue
            s2b[s][:, :] = __set_func_ml_axis(gyr, s2b[s][-1, :], negate_axis=True)
            # Anteroposterior axis
            s2b[s][0, :] = __axis_cross_product(s2b[s][1, :], s2b[s][-1, :])
            # orthog inf_sup axis
            s2b[s][-1, :] = __axis_cross_product(s2b[s][0, :], s2b[s][1, :])
        else:
            warnings.warn(
                f"Sensor{s} not attributed to side - no functional ML axis generated.",
                UserWarning,
            )

        # if shank_placement == "anterior":
        #     if "r" in s:
        #         if "shank" in s:
        #             s2b[s][:, :] = __set_func_ml_axis(
        #                 data[s][trialsmap["rcycle"]],
        #                 s2b[s][-1, :],
        #                 "r",
        #                 s,
        #                 isshank=True,
        #             )

        #         else:
        #             s2b[s][:, :] = __set_func_ml_axis(
        #                 data[s][trialsmap["rcycle"]], s2b[s][-1, :], "r", s
        #             )
        #     elif "l" in s:
        #         if "shank" in s:
        #             s2b[s][:, :] = __set_func_ml_axis(
        #                 data[s][trialsmap["lcycle"]],
        #                 s2b[s][-1, :],
        #                 "r",
        #                 s,
        #                 isshank=True,
        #             )
        #         else:
        #             s2b[s][:, :] = __set_func_ml_axis(
        #                 data[s][trialsmap["lcycle"]], s2b[s][-1, :], "l", s
        #             )
        # elif shank_placement == "lateral":
        #     if "r" in s:
        #         s2b[s][:, :] = __set_func_ml_axis(
        #             data[s][trialsmap["rcycle"]], s2b[s][-1, :], "r", s
        #         )
        #     elif "l" in s:
        #         s2b[s][:, :] = __set_func_ml_axis(
        #             data[s][trialsmap["lcycle"]], s2b[s][-1, :], "l", s
        #         )

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
    # temp_cycle_accel = data.loc[:, ["Acc_X", "Acc_Y", "Acc_Z"]].mean().to_numpy()
    # temp_cycle_accel /= np.linalg.norm(temp_cycle_accel)
    # ml_direc_vec = np.cross(
    #     vertical_axis,
    #     temp_cycle_accel,
    # )
    raise NotImplementedError
    return  # ml_direc_vec


def __set_func_ml_axis(gyr, vertical_axis, negate_axis=False):
    # eigvector decomp to get gyr signal variance. Largest variance is ML axis

    evals, evecs = np.linalg.eig(np.cov((gyr - np.mean(gyr, axis=0)).T))
    evec_col_idx = np.argsort(evals)[-1]
    # ml_direc_vec = __find_temp_vec_4_ml_axis_dir_check(data, vertical_axis)

    # flip the eigenvector if it is pointing in the wrong direction
    # if not np.array_equal(np.sign(evecs[:, 0]), np.sign(ml_direc_vec)):
    #     evecs *= -1

    # // commented out but works
    # if (isshank and side == "r") and evecs[1, 0] < 0:
    #     evecs[:, 0] *= -1
    # elif (isshank and side == "l") and evecs[1, 0] > 0:
    #     # leave as is
    #     evecs[:, 0] *= 1
    # elif not isshank:
    #     if "foot" in sensorid:
    #         if (np.argmax(np.abs(evecs[:, 0])) == 0) and (
    #             np.abs(evecs[np.argmax(np.abs(evecs[:, 0])), 0]) > 0.9
    #         ):
    #             evecs[:, 0] = np.flip(evecs[:, 0])
    #     if side == "r" and np.sign(evecs[np.argmax(np.abs(evecs[:, 0])), 0]) == -1:
    #         evecs *= -1
    #     if side == "l" and np.sign(evecs[np.argmax(np.abs(evecs[:, 0])), 0]) == 1:
    #         # print(f"Left {-evecs[:,0]}")
    #         evecs *= -1
    # # so primary axis is always in positive wrt. to the sensor frame
    # # if evecs[np.argmax(np.abs(evecs[:, 0])), 0] <= 0:
    # #     evecs *= -1
    # //
    if negate_axis:
        evecs *= -1
    ml_axis = evecs[
        :, evec_col_idx
    ]  # had this as -1*evecs[:,0] before...i think it doesnt make sense

    ap_axis = np.cross(
        ml_axis,
        vertical_axis,
    )
    ap_axis /= np.linalg.norm(ap_axis)
    infsup = np.cross(ap_axis, ml_axis)
    infsup /= np.linalg.norm(infsup)
    return np.vstack([ap_axis, ml_axis, infsup])


def __axis_cross_product(axis1, axis2):
    return np.cross(axis1, axis2) / np.linalg.norm(np.cross(axis1, axis2))


# def __set_ap_axis(ml_axis, vertical_axis):
#     ap_axis = np.cross(
#         ml_axis,
#         vertical_axis,
#     )
#     ap_axis /= np.linalg.norm(ap_axis)
#     return ap_axis


# def __set_infsup_axis(ap_axis, ml_axis):
#     infsup = np.cross(ap_axis, ml_axis)
#     infsup /= np.linalg.norm(infsup)
#     return infsup


def __flip_left_ml_axis(evecs, side):
    # TODO: remove later as this is unused.
    # flip ML axis for left side so all axis conventions match Right to Left
    # if side == "l" and (evecs[-1] > 0):
    #     evecs *= -1
    raise NotImplementedError
    return evecs


def find_common_start_time(data: dict) -> dict:
    """
    Find the highest common start time across sensors for each trial.

    Parameters
    ----------
    data : dict
        A dictionary containing sensor data for each trial. data[sensor][trial] contains a pandas DataFrame.

    Returns
    -------
    dict
        A dictionary where the keys are trial numbers and the values are the highest common start time
        across sensors for each trial.
    """
    num_files = len(data[list(data.keys())[0]])

    # collect the first time stamp for each trial on each sensor
    test = np.full((len(data.keys()), num_files), np.nan)
    for j, s in enumerate(data.keys()):
        if data[s] is None:
            test[j, :] = 0
            continue

        for i, trial in enumerate(data[s]):
            if trial is None:
                test[j, i] = 0
                continue
            test[j, i] = trial.loc[0, "SampleTimeFine"]  # test is sensor by trial

    # the highest common timestamp across sensors for each trial
    latest_starttime_idx = np.sort(test, axis=0)
    stamp_exists = []
    num_sensors, num_trials = test.shape
    common_stamp = {t: [] for t in range(num_trials)}
    for i in range(num_trials):
        stamp_exists = np.full(
            (latest_starttime_idx.shape[0], len(data.keys())),
            dtype=bool,
            fill_value=False,
        )
        stamp_exists = []
        start_indices = []
        stamp_exists_all = []
        for m in range(1, latest_starttime_idx.shape[0]):
            stamp2test = int(latest_starttime_idx[-m][i])
            for j, s in enumerate(data.keys()):
                if data[s] is None:
                    continue
                if data[s][i] is None:
                    continue
                else:  # skip if no data
                    # print(data[s][i].loc[:, "SampleTimeFine"].isin([stamp2test]).any())
                    # stamp_exists[m, j] = data[s][i].loc[:, "SampleTimeFine"].isin([stamp2test]).any()

                    stamp_exists.append(
                        data[s][i].loc[:, "SampleTimeFine"].isin([stamp2test]).any()
                    )

                    # stamp_exists[m,j]= data[s][i].loc[:, "SampleTimeFine"].isin([latest_starttime_idx[-m][i]]).any()

                    # stamp_exists.append(
                    #     data[s][i]
                    #     .loc[:, "SampleTimeFine"]
                    #     .isin([latest_starttime_idx[-m][i]])
                    #     .any()
                    # )
                    start_indices.append(data[s][i].loc[0, "SampleTimeFine"])
            stamp_exists_all.append(stamp_exists)
            # // On a few occasions the sensors "syncd" in a way that looks right but the timestamp values are off. In this case, the timestamps values are much different and a syncing on these values isnt possible so we default to the index of 0. Here the trial is left as a blank list "[]" and when syncing in next step, if there is no value in the common_stamp, the timestamp used will just be the 0 index.

            if not all(stamp_exists):
                stamp_exists = []
                continue
            if all(stamp_exists):
                common_stamp[i] = latest_starttime_idx[-m][i]

            break
            # else:
            #     stamp_exists = []
            #     continue

    return common_stamp


def sync_multi_dot(data: dict, syncidx: dict) -> dict:
    """
    Synchronize multiple dot sensor data based on sync indices.

    Parameters
    ----------
    data : dict
        A dictionary containing dot sensor data for different trials.
        data[sensor][trial] contains a pandas DataFrame.
    syncidx : list
        A list of sync indices corresponding to each trial. Values are expecte to be within the SampleTimeFine column of the dot sensor data that corresponds to the specific timestamp for syncronizing across sensors for a given trial.

    Returns
    -------
    dict
        A synchronized version of the dot sensor data.

    """
    syncd_data = copy.deepcopy(data)

    for s in syncd_data:
        if syncd_data[s] is None:
            continue
        for i, trial in enumerate(syncd_data[s]):
            if trial is None:
                continue
            if not syncidx[
                i
            ]:  # List is left empty if syncing not possible - default to idx=0
                first_row = 0
            else:
                first_row = trial.loc[:, "SampleTimeFine"].eq(syncidx[i]).idxmax()
            trimmed_data = trial.iloc[first_row:, :].reset_index(drop=True)
            syncd_data[s][i] = trimmed_data
    return syncd_data
