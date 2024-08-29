import numpy as np
import quaternions as quat
from vqf import VQF


def static_tilt(a):
    """Estimate the pitch and roll of an imu when it is assumed stationary

    Parameters
    ----------
    acc : ndarray
        3x1 (channel x frame) accelerometer data
    Returns
    -------
    tuple of roll and pitch angles in radians
    """
    acc = a / np.linalg.norm(a)
    # pitch roll estimation
    pitch_am = np.arctan2(acc[0], np.sqrt(acc[1] * acc[1] + acc[2] * acc[2]))
    roll_am = np.arctan2(acc[1], -acc[2])
    rpy = np.array([roll_am, pitch_am, 0])
    return rpy


def yaw_from_mag(ori, m):
    """
    Calculate the yaw angle from magnetometer readings.

    Parameters
    ----------
    ori : array-like
        Orientation angles [roll, pitch, yaw] in radians.
    m : array-like
        Magnetometer readings [mx, my, mz].

    Returns
    -------
    float
        Yaw angle in radians.
    """

    mag = m / np.linalg.norm(m)
    # yaw estimation
    by = mag[2] * np.sin(ori[1]) - mag[1] * np.cos(ori[1])
    bx = mag[0] * np.cos(ori[0]) + np.sin(ori[0]) * (
        mag[1] * np.sin(ori[1]) + mag[2] * np.cos(ori[1])
    )
    yaw_am = np.arctan2(by, bx)
    return yaw_am


def gyro_euler_conversion(rpy):
    """Angular velocity data from a gyroscope are measured about the orthogonal body fixed axes simultaneously. However, when integrating the angular velocity data at time t with the orientation estimate at time t-1, we run into a problem. The orientation estimate is (usually) represented by Euler angles which are about intermediate axes generated during the rotation sequence. So we are adding two different types of data. This applies a rotation matrix to the angular velocity to place each angular velocity in the same frame as the previous orientation estimations.

    For a detailed derivation and explanation: https://youtu.be/9GZjtfYOXao?si=lAYdKQmngrx49qSe

    Parameters
    ----------
    rpy : ndarray
        roll, pitch, yaw angles at one sample in time. If using this in the context of angular velocity integration, this should be time=t-1

    Returns
    -------
    ndarray
        Rotation matrix to be right multiplied with vector of angular velocity data.
    """
    R = np.array(
        [
            [1, np.sin(rpy[0]) * np.tan(rpy[1]), np.cos(rpy[0]) * np.tan(rpy[1])],
            [0, np.cos(rpy[0]), -np.sin(rpy[0])],
            [0, np.sin(rpy[0]) / np.cos(rpy[1]), np.cos(rpy[0]) / np.cos(rpy[1])],
        ]
    )

    return R


def complementary_filter_rpy(
    acc,
    gyr,
    mag,
    rpy=np.array([0, 0, 0]),
    weight=0.985,
    timestep=1 / 120,
    gyro_correction=True,
):
    """
    Perform orientation estimation (RPY) using a complementary filter approach and represent the orientation as a roll, pitch, yaw.
    Call this function for each time-step in your data and pass the output rpy back into the function for the next time-step.

    Parameters:
    ----------
        acc ndarray : 3x1 (channel x frame) accelerometer data
        gyr ndarray : 3x1 (channel x frame) gyroscope data in radians per second
        mag ndarray : 3x1 (channel x frame) magnetometer data
        rpy ndarray : roll, pitch, yaw angles in radians (default: [0, 0, 0])
        weight float : weight factor between 0 and 1. High weight biases the gyro data, low weight biases the accel/mag data (default: 0.985)
        timestep float : time in seconds between each sample (e.g., inverse of sample frequency) (default: 1 / 120)

    Returns:
    -------
        ndarray: newly estimated roll, pitch, yaw angles in radians
    """

    # gyro integration
    if gyro_correction:
        rpy_gyro = rpy + (gyro_euler_conversion(rpy) @ gyr) * timestep
    else:
        rpy_gyro = rpy + gyr * timestep

    a = acc / np.linalg.norm(acc)
    m = mag / np.linalg.norm(mag)

    # pitch roll estimation
    rpy_tilt = static_tilt(a)
    rpy_am_temp = rpy_tilt * (1 - weight) + rpy_gyro * weight

    yaw_am = yaw_from_mag(rpy_am_temp, m)
    rpy_am = np.array([rpy_tilt[0], rpy_tilt[1], yaw_am])

    weight_adaptive = adaptive_gain(
        acc, g=9.8094, static_alpha=0.1, function_form="sigmoid"
    )
    # full complementary
    # static
    rpy_out = rpy_gyro * weight + rpy_am * (1 - weight)

    # adaptive
    # rpy_out = rpy_gyro * weight_adaptive + rpy_am * (1 - weight_adaptive)

    return rpy_out, rpy_gyro, rpy_am, weight_adaptive


def qmag_from_mag(magfield: np.ndarray):
    gamma = (
        magfield[0] ** 2 + magfield[1] ** 2
    )  # magnitude of magnetic field in in N and U plane

    # eq 35
    if magfield[0] >= 0:
        return np.array(
            [
                np.sqrt(gamma + magfield[0] * np.sqrt(gamma)) / np.sqrt(2 * gamma),
                0,
                0,
                magfield[1]
                / (np.sqrt(2) * np.sqrt((gamma + magfield[0] * np.sqrt(gamma)))),
            ]
        )
    elif magfield[0] < 0:
        return np.array(
            [
                magfield[1]
                / (np.sqrt(2) * np.sqrt((gamma - magfield[0] * np.sqrt(gamma)))),
                0,
                0,
                np.sqrt(gamma - magfield[0] * np.sqrt(gamma)) / np.sqrt(2 * gamma),
            ]
        )
    else:
        raise ValueError("magnetic field undefined")


def qcomp_init(acc, mag):
    # initialize quaternion using acc and mag. Section 4 of https://doi.org/10.3390/s150819302
    # normalize acc and mag
    acc = acc / np.linalg.norm(acc)
    # implement eq 25
    qa = to_quaternion_form(acc)
    if mag is not None:
        mag = mag / np.linalg.norm(mag)
        mag_field = quat.to_rotmat(qa) @ mag  # eq 26
        qmag = qmag_from_mag(mag_field)
        q_init = quat.product(qa, qmag)
    q_init = qa
    return q_init


def qcomp(
    acc,
    gyr,
    mag,
    q0,
    timestep=0.01,
    alpha=0.01,
    beta=0.01,
    acc_threshold=0.9,
    mag_threshold=0.9,
):
    """
    From Laidig 2023 (https://doi.org/10.1016/j.inffus.2022.10.014), they compared their VQF algorithm to this qcomp filter from Valenti et al (10.3390/s150819302) by doing a grid search on the tuning parameters. they found that the ideal parameters for Valenti were:
    alpha_acc = 0.00085
    beta_mag = 0.0005
    a_bias = 0.00055,
    a_adapt = False
    beta_est = True
    """

    # gyro must be in rad/s
    # acc_raw = acc.copy()  # retain unnormalized acc
    # get adaptive gain
    # alpha = adaptive_gain(acc)

    # integrate angular rate
    acc = acc / np.linalg.norm(acc)
    qdotg = -0.5 * quat.product(np.array([0, gyr[0], gyr[1], gyr[2]]), q0)  # eq 38

    qg = q0 + qdotg * timestep  # eq 42
    qg = qg / np.linalg.norm(qg)  # normalize - based on reccomendation from AHRS.com

    # roll pitch correction with qa
    pred_grav = (
        quat.to_rotmat(quat.inverse(qg)) @ acc
    )  # rotmat_from_q(qg).squeeze() @ acc  # eq 43

    # eq 47
    delta_qa = to_quaternion_form(pred_grav)

    # small correction to gravity prediction using either SIMPLE linear interpolation or SPHERICAL linear interpolation
    if delta_qa[0] > acc_threshold:  # simple
        delta_qa_filtered = LERP(alpha, delta_qa, omega=None, type="simple")
    elif delta_qa[0] <= acc_threshold:  # Spherical
        omega_acc = delta_qa[0]  # np.arccos(delta_qa[0])
        delta_qa_filtered = LERP(beta, delta_qa, omega=omega_acc, type="spherical")

    # Corrected q based on accelerometer
    # implement eq 53
    qprime = quat.product(qg, delta_qa_filtered)  # quatmult(qg, delta_qa).squeeze()

    if mag is not None:
        mag = mag / np.linalg.norm(mag)

        # Mag correction for yaw
        mag_world_frame = quat.to_rotmat(quat.inverse(qprime)) @ mag
        # rotmat_from_q(qinv(qprime)).squeeze() @ mag
        delta_qmag = qmag_from_mag(mag_world_frame)

        if delta_qmag[0] > mag_threshold:  # simple
            delta_qmag_filtered = LERP(alpha, delta_qmag, omega=None, type="simple")
        elif delta_qmag[0] <= mag_threshold:  # Spherical
            omega_mag = delta_qmag[0]  # np.arccos(delta_qmag[0])
            delta_qmag_filtered = LERP(
                alpha, delta_qmag, omega=omega_mag, type="spherical"
            )
        q_global2local = quat.normalize(quat.product(qprime, delta_qmag_filtered))

    else:
        q_global2local = quat.normalize(qprime)
    return q_global2local


def adaptive_gain(acc_raw, g=9.81, static_alpha=0.1, function_form="piecewise"):
    magnitude_error = np.abs((np.linalg.norm(acc_raw) - g)) / g

    if function_form == "piecewise":
        # gain factor linear function based on Fig 5 using thresholds of 0.1-->0.2
        def piecewise_gain(x):
            conditions = [(x < 0.1), (x >= 0.1) & (x <= 0.2), (x > 0.2)]
            # functions = [lambda x: 1, lambda x: 10 - 10 * x, lambda x: 0]
            functions = [lambda x: 0.99, lambda x: 1 - 9 * (x - 0.09), lambda x: 0.01]

            gain = np.piecewise(x, conditions, functions)
            return gain

        return 1 - piecewise_gain(magnitude_error)
    elif function_form == "sigmoid":

        def sigmoid_gain(x):
            """y= a *(1 / (1 + e^(-bx - c))+f) + d"""
            a = 0.75
            b = 23
            c = -5
            d = 0.05
            f = -0.2

            return a * (1 / ((1 + np.exp(b * -x - c)) + f)) + d

        return sigmoid_gain(magnitude_error)
    else:
        return 0.985


def LERP(alpha, deltqa, omega, type="simple"):
    qI = np.array([1, 0, 0, 0])
    if type == "simple":
        q_out = (1 - alpha) * qI + alpha * deltqa
        return quat.normalize(q_out)  # / np.linalg.norm(q_out)
    elif type == "spherical":
        q_out = (np.sin((1 - alpha) * omega) / np.sin(omega)) * qI + (
            np.sin(alpha * omega) / np.sin(omega)
        ) * deltqa
        return quat.normalize(q_out)  # / np.linalg.norm(q_out)
    else:
        raise ValueError("type must be 'simple' or 'spherical'")


def to_quaternion_form(w):
    # based on eq 25 and can be used for accel adn delta_qa formlations
    if w[-1] >= 0:
        return np.array(
            [
                np.sqrt((w[2] + 1) / 2),
                -w[1] / np.sqrt(2 * (w[2] + 1)),
                w[0] / np.sqrt(2 * (w[2] + 1)),
                0,
            ]
        )
    elif w[-1] < 0:
        return np.array(
            [
                -w[1] / np.sqrt(2 * (1 - w[2])),
                np.sqrt((1 - w[2]) / 2),
                0,
                w[0] / np.sqrt(2 * (1 - w[2])),
            ]
        )
    else:
        raise ValueError(f"singularity or undefined found for w = {w}")


# MAHONEY FILTER -----------------------------------------
def mahoney(acc, gyr, mag, freq, k_P, k_I, q0, b0, frame="NED"):
    """
    Minor adaptation from github.com/Mayitzin/ahrs. Please go look at the excellent documentation and original code by Mario Garcia.

    Original algorithm derivation:
    Robert Mahony, Tarek Hamel, and Jean-Michel Pflimlin. Nonlinear Complementary Filters on the Special Orthogonal Group. IEEE Transactions on Automatic Control, Institute of Electrical and Electronics Engineers, 2008, 53 (5), pp.1203-1217
    """

    # initialize quaternion
    if q0 is None:
        Q = am2q(acc, mag, frame=frame)
    else:
        Q = q0 / np.linalg.norm(q0)

    q_out, b = updateMARG(Q, gyr, acc, mag, dt=1 / freq, k_I=k_I, k_P=k_P, b0=b0)
    return q_out, b


def am2DCM(a, m, frame="NED"):
    H = np.cross(m, a)
    H = H / np.linalg.norm(H)
    a = a / np.linalg.norm(a)
    M = np.cross(a, H)
    if frame.upper() == "ENU":
        return np.array([[H[0], M[0], a[0]], [H[1], M[1], a[1]], [H[2], M[2], a[2]]])
    else:
        return np.array([[M[0], H[0], -a[0]], [M[1], H[1], -a[1]], [M[2], H[2], -a[2]]])


def updateMARG(q, gyr, acc, mag, dt, k_I=0.3, k_P=1, b0=np.zeros(3)):
    # if gyr is None or not np.linalg.norm(gyr) > 0:
    #     return q
    b = b0

    omega = np.copy(gyr)
    a_norm = np.linalg.norm(acc)

    if a_norm > 0:
        m_norm = np.linalg.norm(mag)
        if not m_norm > 0:
            return updateIMU(q, gyr, acc, dt, k_P, k_I, b)

        a = np.copy(acc) / a_norm
        m = np.copy(mag) / m_norm
        R = quat.to_rotmat(q, homogenous=False)  # q2R(q)
        v_a = R.T @ np.array([0, 0, 1])  # expected grav

        # mag field to inertial frame
        h = R @ m
        v_m = R.T @ np.array([0.0, np.linalg.norm([h[0], h[1]]), h[2]])
        vm = v_m / np.linalg.norm(v_m)

        # ECF
        omega_mes = np.cross(a, v_a) + np.cross(m, vm)
        bDot = -k_I * omega_mes
        b += bDot * dt
        omega_corrected = omega - b + k_P * omega_mes
    else:
        return updateIMU(q, gyr, acc, dt, k_P, k_I, b)

    p = np.array([0.0, *omega_corrected])
    qDot = 0.5 * quat.product(q, p)  # quatmult(q, p).squeeze()
    q += qDot * dt
    q = q / np.linalg.norm(q)
    return q, b


def updateIMU(q, gyr, acc, dt, k_P, k_I, b):
    omega = np.copy(gyr)
    a_norm = np.linalg.norm(acc)
    if a_norm > 0:
        R = quat.to_rotmat(q)  # q2R(q)
        v_a = R.T @ np.array([0.0, 0.0, 1])  # expected grav
        # ECF
        omega_mes = np.cross(acc / a_norm, v_a)  # Cost function (eqs. 32c and 48a)
        bDot = -k_I * omega_mes  # Estimated change in Gyro bias
        b += bDot * dt  # Estimated Gyro bias (eq. 48c)
        omega = omega - b + k_P * omega_mes
    p = np.array([0.0, *omega])
    qDot = 0.5 * quat.product(
        q, p
    )  # quatmult(q, p)  # Rate of change of quaternion (eqs. 45 and 48b)
    q += qDot * dt  # Update orientation
    q /= np.linalg.norm(q)
    return q, b


def am2q(a, m, frame="NED"):
    R = am2DCM(a, m, frame)
    return quat.from_rotmat(R)  # dcm2q(R)


def laidig_vqf(acc, gyr, mag=None, freq=None, dt=None, filter_form="quat9D"):
    # VQF (https://vqf.readthedocs.io/en/latest/index.html)
    # Settings:
    # magDistRejectionEnabled set to False as it seems to be detecting disturbances at all frames

    if freq is None and dt is None:
        raise ValueError("Must provide either frequency (freq) or time step (dt)")
    if freq is not None:
        dt = 1 / freq

    if mag is None:
        mag = np.zeros_like(acc)
    vqf_filter = VQF(
        gyrTs=dt,
        accTs=-1,  # "-1" == same as gyr
        magTs=-1,  # "-1" == same as gyr
        magDistRejectionEnabled=False,
    )

    # // must be c-contiguous
    vqf_out = vqf_filter.updateBatchFullState(
        acc=acc.copy(order="C"),
        gyr=gyr.copy(order="C"),
        mag=mag.copy(order="C"),
    )
    vqf_quaternion = vqf_out[filter_form]
    return vqf_quaternion


def extract_yaw_from_quaternion(q):
    if q.ndim > 1 and q.shape[0] > 1:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        w, x, y, z = q[0], q[1], q[2], q[3]
    yaw_component = np.arctan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
    return yaw_component


def _remove_yaw_from_quaternion(q):
    q_yaw_euler = np.full((q.shape[0], 3), np.array([0.0, 0.0, 1.0]))
    q_yaw_euler[:, -1] *= extract_yaw_from_quaternion(q)
    q_yaw = quat.from_rpy(q_yaw_euler)
    yaw_removed_q = quat.product(quat.inverse(q_yaw), q)
    return yaw_removed_q


def relative_angle_from_quaternion(q1, q2, remove_yaw=True, return_euler=False):
    if remove_yaw:
        q1_ = _remove_yaw_from_quaternion(q1)
        q2_ = _remove_yaw_from_quaternion(q2)
    else:
        q1_ = q1
        q2_ = q2

    q_rel = quat.product(q1_, quat.inverse(q2_))

    if return_euler:
        return quat.to_angles(q_rel)
    else:
        return q_rel


def strapdown_imu_position(gait_events, data):
    # An python-based implementation of the strapdown inertial navigation algorithm from Gibson et al 2024 PLOS One.

    # 1. ori forwards midstance to midstance
    # 2. ori backwackwards midstance to midstance
    # 3. strapdown correction
    # 4. return corrected orientation.
    pass
