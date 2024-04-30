import numpy as np
import quaternions as quat


def static_tilt(acc):
    """Estimate the pitch and roll of an imu when it is assumed stationary

    Parameters
    ----------
    acc : ndarray
        _description_
    mag : ndarray
        _description_
    """
    acc = acc / np.linalg.norm(acc)
    # pitch roll estimation
    pitch_am = np.arctan2(acc[0], np.sqrt(acc[1] * acc[1] + acc[2] * acc[2]))
    roll_am = np.arctan2(acc[1], -acc[2])
    return roll_am, pitch_am


def yaw_from_mag(ori, m):
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
            [0, np.sin(rpy[0]) * np.cos(rpy[1]), np.cos(rpy[0]) / np.cos(rpy[1])],
        ]
    ).T

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

    Args:
        acc (ndarray): 3x1 (channel x frame) accelerometer data
        gyr (ndarray): 3x1 (channel x frame) gyroscope data in radians per second
        mag (ndarray): 3x1 (channel x frame) magnetometer data
        rpy (ndarray): roll, pitch, yaw angles in radians (default: [0, 0, 0])
        weight (float): weight factor between 0 and 1. High weight biases the gyro data, low weight biases the accel/mag data (default: 0.985)
        timestep (float): time in seconds between each sample (e.g., inverse of sample frequency) (default: 1 / 120)

    Returns:
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
    roll_am, pitch_am = static_tilt(a)
    # partial complement to be used for mag compensation
    rpy_am_temp = (
        np.array([roll_am, pitch_am, 0]) * (1 - weight)
        + np.array([rpy_gyro[0], rpy_gyro[1], rpy_gyro[2]]) * weight
    )
    yaw_am = yaw_from_mag(rpy_am_temp, m)

    # compile accel and mag estimate
    rpy_am = np.array([roll_am, pitch_am, yaw_am])

    # full complementary
    rpy_out = rpy_gyro * weight + rpy_am * (1 - weight)

    return rpy_out


def qcomp_init(acc, mag):
    # initialize quaternion using acc and mag. Section 4 of https://doi.org/10.3390/s150819302
    # normalize acc and mag
    acc = acc / np.linalg.norm(acc)
    mag = mag / np.linalg.norm(mag)

    # implement eq 25
    qa = to_quaternion_form(acc)

    mag_cor = rotmat_from_q(qa).T @ mag  # eq 26

    gamma = mag_cor[0] ** 2 + mag_cor[1] ** 2

    # implement eq 35
    if mag_cor[0] >= 0:
        qmag = np.array(
            [
                np.sqrt(gamma + mag_cor[0] * np.sqrt(gamma)) / np.sqrt(2 * gamma),
                0,
                0,
                mag_cor[1]
                / np.sqrt(2)
                * np.sqrt((gamma + mag_cor[0] * np.sqrt(gamma))),
            ]
        )
    elif mag_cor[0] < 0:
        qmag = np.array(
            [
                mag_cor[1]
                / np.sqrt(2)
                * np.sqrt((gamma - mag_cor[0] * np.sqrt(gamma))),
                0,
                0,
                np.sqrt(gamma - mag_cor[0] * np.sqrt(gamma)) / np.sqrt(2 * gamma),
            ]
        )
    else:
        raise ValueError("magnetic field undefined")

    return quatmult(qa, qmag)


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
    # gyro must be in rad/s
    acc_raw = acc.copy()  # retain unnormalized acc
    # get adaptive gain
    # alpha = adaptive_gain(acc_raw)

    # integrate angular rate
    # pure_gyro_q = np.array([0, gyr[0], gyr[1], gyr[2]])
    # qdotg = 0.5 * quatmult(pure_gyro_q, q0).squeeze(axis=-1)  # eq 38
    acc = acc / np.linalg.norm(acc)
    mag = mag / np.linalg.norm(mag)

    # eq 39 and 40
    angvel_sframe = np.array(
        [
            [
                [0, gyr[0], gyr[1], gyr[2]],
                [-gyr[0], 0, -gyr[2], gyr[1]],
                [-gyr[1], gyr[2], 0, -gyr[0]],
                [-gyr[2], -gyr[1], gyr[0], 0],
            ]
        ]
    ).squeeze()

    qdotg = -0.5 * angvel_sframe @ q0

    qg = q0 + qdotg * timestep  # eq 42
    qg = qg / np.linalg.norm(qg)  # normalize - based on reccomendation from AHRS.com

    # roll pitch correction with qa
    pred_grav = rotmat_from_q(qinv(qg)).squeeze() @ acc  # eq 44
    delta_qa = to_quaternion_form(pred_grav)  # eq 47

    # small correction to gravity prediction using either SIMPLE linear interpolation or SPHERICAL linear interpolation
    if delta_qa[0] > acc_threshold:  # simple
        delta_qa = LERP(alpha, delta_qa, omega=None, type="simple")
    elif delta_qa[0] <= acc_threshold:  # Spherical
        omega_acc = np.arccos(delta_qa[0])
        delta_qa = LERP(alpha, delta_qa, omega=omega_acc, type="spherical")

    # Corrected q based on accelerometer
    # implement eq 53
    qprime = quatmult(qg, delta_qa).squeeze()

    # Mag correction for yaw
    mag_world_frame = rotmat_from_q(qinv(qprime)).squeeze() @ mag

    gamma_wf = mag_world_frame[0] ** 2 + mag_world_frame[1] ** 2

    delta_qmag = np.array(
        [
            np.sqrt(gamma_wf + mag_world_frame[0] * np.sqrt(gamma_wf))
            / np.sqrt(2 * gamma_wf),
            0,
            0,
            mag_world_frame[1]
            / np.sqrt(2 * (gamma_wf + mag_world_frame[0] * np.sqrt(gamma_wf))),
        ]
    )

    if delta_qmag[0] >= mag_threshold:  # simple
        delta_qmag = LERP(beta, delta_qmag, omega=None, type="simple")
    elif delta_qmag[0] < mag_threshold:  # Spherical
        omega_mag = np.arccos(delta_qmag[0])
        delta_qmag = LERP(beta, delta_qmag, omega=omega_mag, type="spherical")

    # Corrected q based on magnetometer
    q_global2local = quatmult(qprime, delta_qmag)

    return q_global2local


def adaptive_gain(acc_raw, g=9.81):
    magnitude_error = np.abs((np.linalg.norm(acc_raw) - g)) / g

    # gain factor linear function based on Fig 5 using thresholds of 0.1-->0.2
    def gainfactor(x):
        # return np.piecewise(x, [x < 0.1, ((x >= 0.1) & (x <= 0.2)), x > 0.2], [0.99, lambda x: (-10) * x + 2, 0])

        # return np.piecewise(x, [x < 0.1, ((x >= 0.1) & (x <= 0.2)), x > 0.2], [0.5, lambda x: -(0.2 / 0.1) * x + 0.4, 0])
        x1 = 0.5
        x2 = 2
        y1 = 0.985
        y2 = 0
        return np.piecewise(
            x,
            [x < x1, ((x >= x1) & (x <= x2)), x > x2],
            [
                y1,
                lambda x: ((y2 - y1) / (x2 - x1)) * x + (y2 + (y2 + y1) + y2),
                y2,
            ],
            # [x < 0.1, ((x >= 0.1) & (x <= 0.2)), x > 0.2],
            # [0, lambda x: (10) * x - 1, 0.99],
            # [x < 0.1, ((x >= 0.1) & (x <= 0.2)), x > 0.2],
            # [
            #     0.995,
            #     lambda x: -((0.995 - 0.1) / 0.1) * x + (0.995 + (0.995 - 0.1)),
            #     0.1,
            # ],
            # [x < 0.1, ((x >= 0.1) & (x <= 0.2)), x > 0.2],
            # [
            #     1,
            #     lambda x: -((1) / 0.1) * x + (1 + (1)),
            #     0,
            # ],
        )

    return 1 - gainfactor(magnitude_error)


def LERP(alpha, deltaq, omega, type="simple"):
    qI = np.array([1, 0, 0, 0])
    if type == "simple":
        q_out = (1 - alpha) * qI + alpha * deltaq
        return q_out / np.linalg.norm(q_out)
    elif type == "spherical":
        q_out = (np.sin((1 - alpha) * omega) / np.sin(omega)) * qI + (
            np.sin(alpha * omega) / np.sin(omega)
        ) * deltaq
        return q_out / np.linalg.norm(q_out)
    else:
        raise ValueError("type must be 'simple' or 'spherical'")


def rotmat_from_q(q):
    return np.array(
        [
            [
                q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
                2 * (q[1] * q[2] - q[0] * q[3]),
                2 * (q[1] * q[3] + q[0] * q[2]),
            ],
            [
                2 * (q[1] * q[2] + q[0] * q[3]),
                q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2,
                2 * (q[2] * q[3] - q[0] * q[1]),
            ],
            [
                2 * (q[1] * q[3] - q[0] * q[2]),
                2 * (q[2] * q[3] + q[0] * q[1]),
                q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2,
            ],
        ]
    )


def quatmult(p, q):
    # Hamilton multiplication
    return np.array(
        [
            [p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]],
            [p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2]],
            [p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1]],
            [p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]],
        ]
    )


def qinv(q):
    # return np.array([[q[0]], [-q[1]], [-q[2]], [-q[3]]])
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q2eul(w, x, y, z):
    phi = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
    theta = np.arcsin(2.0 * (w * y - z * x))
    psi = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

    # WORKS (negative flexion)
    # phi = np.arctan2(-2 * (y * z) + 2 * (w * x), z * z - y * y - x * x + w * w)
    # theta = np.arcsin(2 * (x * z + w * y))
    # psi = np.arctan2(-2 * (x * y) + 2 * (w * z), x * x + w * w - z * z - y * y)

    # phi = np.arctan2(x - z, w + y) - np.arctan2(z - x, y + w)
    # theta = np.arccos((w - y) ** 2 + (x - z) ** 2 - 1) - np.pi / 2
    # psi = np.arctan2(x + z, w - y) + np.arctan2(z - x, y + w)
    return np.array([phi, theta, psi])


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


def rpy_to_quat(r, p, y):
    # convert roll, pitch, yaw to quaternion (w, x, y, z)
    return np.array(
        [
            [
                np.cos(p / 2) * np.cos(r / 2) * np.cos(y / 2)
                + np.sin(p / 2) * np.sin(r / 2) * np.sin(y / 2)
            ],
            [
                np.sin(p / 2) * np.cos(r / 2) * np.cos(y / 2)
                - np.cos(p / 2) * np.sin(r / 2) * np.sin(y / 2)
            ],
            [
                np.cos(p / 2) * np.sin(r / 2) * np.cos(y / 2)
                + np.sin(p / 2) * np.cos(r / 2) * np.sin(y / 2)
            ],
            [
                np.cos(p / 2) * np.cos(r / 2) * np.sin(y / 2)
                - np.sin(p / 2) * np.sin(r / 2) * np.cos(y / 2)
            ],
        ]
    )


# MAHONEY FILTER -----------------------------------------
def mahoney(acc, gyr, mag, freq, k_P, k_I, q0, b0):
    # initialize quaternion
    if q0 is None:
        Q = am2q(acc, mag)
    else:
        Q = q0 / np.linalg.norm(q0)

    q_out, b = updateMARG(Q, gyr, acc, mag, dt=1 / freq, k_I=k_I, k_P=k_P, b0=b0)
    return q_out, b


def am2DCM(a, m, frame="NED"):
    H = np.cross(m, a)
    H = H / np.linalg.norm(H)
    a = a / np.linalg.norm(a)
    M = np.cross(a, H)
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
        m = np.copy(mag) / a_norm
        R = q2R(q)
        v_a = R.T @ np.array([0, 0, 1])  # expected grav

        # mag field to inertial frame
        h = R @ m
        v_m = R.T @ np.array([0.0, np.linalg.norm([h[0], h[1]]), h[2]])
        vm = v_m / np.linalg.norm(v_m)

        # ECF
        omega_mes = np.cross(a, v_a) + np.cross(m, v_m)
        bDot = -k_I * omega_mes
        b += bDot * dt
        omega = omega - b + k_P * omega_mes
    else:
        return updateIMU(q, gyr, acc, dt, k_P, k_I, b)

    p = np.array([0.0, *omega])
    qDot = 0.5 * quatmult(q, p).squeeze()
    q += qDot * dt
    q = q / np.linalg.norm(q)
    return q, b


def updateIMU(q, gyr, acc, dt, k_P, k_I, b):
    omega = np.copy(gyr)
    a_norm = np.linalg.norm(acc)
    if a_norm > 0:
        R = q2R(q)
        v_a = R.T @ np.array([0.0, 0.0, 1])  # expected grav
        # ECF
        omega_mes = np.cross(acc / a_norm, v_a)  # Cost function (eqs. 32c and 48a)
        bDot = -k_I * omega_mes  # Estimated change in Gyro bias
        b += bDot * dt  # Estimated Gyro bias (eq. 48c)
        omega = omega - b + k_P * omega_mes
    p = np.array([0.0, *omega])
    qDot = 0.5 * quatmult(q, p)  # Rate of change of quaternion (eqs. 45 and 48b)
    q += qDot * dt  # Update orientation
    q /= np.linalg.norm(q)
    return q, b


def am2q(a, m, frame="NED"):
    R = am2DCM(a, m, frame)
    return dcm2q(R)


def dcm2q(R):
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q[0] = 0.5 * np.sqrt(1.0 + R.trace())
    q[1] = (R[1, 2] - R[2, 1]) / q[0]
    q[2] = (R[2, 0] - R[0, 2]) / q[0]
    q[3] = (R[0, 1] - R[1, 0]) / q[0]
    q[1:] /= 4.0
    return q / np.linalg.norm(q)


def q2R(q):
    q /= np.linalg.norm(q)
    return np.array(
        [
            [
                1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
                2.0 * (q[1] * q[2] - q[0] * q[3]),
                2.0 * (q[1] * q[3] + q[0] * q[2]),
            ],
            [
                2.0 * (q[1] * q[2] + q[0] * q[3]),
                1.0 - 2.0 * (q[1] ** 2 + q[3] ** 2),
                2.0 * (q[2] * q[3] - q[0] * q[1]),
            ],
            [
                2.0 * (q[1] * q[3] - q[0] * q[2]),
                2.0 * (q[0] * q[1] + q[2] * q[3]),
                1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2),
            ],
        ]
    )