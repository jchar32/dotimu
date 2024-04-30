import numpy as np
from warnings import warn
# TODO: This file could be combined with similar functions for rotations and potentially packaged separately. However, this has be done by people over and over again.


def to_scalar_first(q):
    """converts a quaternion from scalar last to scalar first

    Args:
        q (ndarray): quanternion with scalar last

    Returns:
        ndarray: quaternion with scalar first
    """
    return np.array([q[3], q[0], q[1], q[2]])


def conjugate(q, scalarLast=False):
    """returns the conjugate of a quaternion

    Args:
        q (ndarray): quanternion assuming scalar first

    Returns:
        ndarray: conjugate of q
    """
    return (
        np.array([q[0], -q[1], -q[2], -q[3]])
        if not scalarLast
        else np.array([-q[0], -q[1], -q[2], q[3]])
    )


def inverse(q, scalarLast=False):
    """returns the inverse of a quaternion

    Args:
        q (ndarray): quanternion assuming scalar first

    Returns:
        ndarray: inverse of q
    """
    return conjugate(q, scalarLast) / np.linalg.norm(q)


def exponential(q, scalarLast=False):
    """returns the exponential of a quaternion

    Args:
        q (ndarray): quanternion assuming scalar first

    Returns:
        ndarray: exponential of q
    """
    q0 = q[0] if not scalarLast else q[3]
    qv = q[1:4] if not scalarLast else q[0:3]
    qv_norm = np.linalg.norm(qv)
    return (
        np.array(
            [np.exp(q0) * np.cos(qv_norm), np.exp(q0) * qv * np.sin(qv_norm) / qv_norm]
        )
        if not scalarLast
        else np.array(
            [np.exp(q0) * qv * np.sin(qv_norm) / qv_norm, np.exp(q0) * np.cos(qv_norm)]
        )
    )


def logarithm(q, scalarLast=False):
    """returns the logarithm of a quaternion

    Args:
        q (ndarray): quanternion assuming scalar first

    Returns:
        ndarray: logarithm of q
    """
    q0 = q[0] if not scalarLast else q[3]
    qv = q[1:4] if not scalarLast else q[0:3]
    qv_norm = np.linalg.norm(qv)
    return (
        np.array(
            [
                np.log(np.linalg.norm(q)),
                qv * np.arccos(q0 / np.linalg.norm(q)) / qv_norm,
            ]
        )
        if not scalarLast
        else np.array(
            [
                qv * np.arccos(q0 / np.linalg.norm(q)) / qv_norm,
                np.log(np.linalg.norm(q)),
            ]
        )
    )


def normalize(q, scalarLast=False):
    """returns the normalized quaternion

    Args:
        q (ndarray): quanternion assuming scalar first

    Returns:
        ndarray: normalized q
    """
    return q / np.linalg.norm(q)


def product(q, p, scalarLast=False):
    """returns the product of two quaternions

    Args:
        q (ndarray): quanternion assuming scalar first
        p (ndarray): quanternion assuming scalar first

    Returns:
        ndarray: product of q and p
    """
    # TODO: should vectorize this.
    if scalarLast:
        q0, qx, qy, qz = to_scalar_first(q)
        p0, px, py, pz = to_scalar_first(p)
    else:
        q0, qx, qy, qz = q.squeeze().tolist()
        p0, px, py, pz = p.squeeze().tolist()

    prod = np.array(
        [
            q0 * p0 - qx * px - qy * py - qz * pz,
            q0 * px + qx * p0 + qy * pz - qz * py,
            q0 * py - qx * pz + qy * p0 + qz * px,
            q0 * pz + qx * py - qy * px + qz * p0,
        ]
    )

    return prod


def to_angles(q, scalarLast=False):
    """returns the angles of a quaternion

    Args:
        q (ndarray): quanternion assuming scalar first

    Returns:
        ndarray: angles of q in phi, theta, psi
    """
    if scalarLast:
        q0, qx, qy, qz = to_scalar_first(q)
    else:
        q0, qx, qy, qz = q.squeeze().tolist()

    phi = np.arctan2(2.0 * (q0 * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2))
    theta = np.arcsin(2.0 * (q0 * qy - qz * qx))
    psi = np.arctan2(2.0 * (q0 * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    return np.array([phi, theta, psi])


def to_DCM(q, scalarLast=False):
    """returns the direction cosine matrix of a quaternion

    Args:
        q (ndarray): quanternion assuming scalar first

    Returns:
        ndarray: direction cosine matrix of q
    """
    if scalarLast:
        q0, qx, qy, qz = to_scalar_first(q)
    else:
        q0, qx, qy, qz = q.squeeze().tolist()

    DCM = np.array(
        [
            [
                1.0 - 2.0 * (qy**2 + qz**2),
                2.0 * (qx * qy - q0 * qz),
                2.0 * (qx * qz + q0 * qy),
            ],
            [
                2.0 * (qx * qy + q0 * qz),
                1.0 - 2.0 * (qx**2 + qz**2),
                2.0 * (qy * qz - q0 * qx),
            ],
            [
                2.0 * (qx * qz - q0 * qy),
                2.0 * (qy * qz + q0 * qx),
                1.0 - 2.0 * (qx**2 + qy**2),
            ],
        ]
    )
    return DCM


def from_rpy(angles: np.ndarray):
    """returns the quaternion from roll, pitch, yaw angles

    Args:
        angles (ndarray): roll, pitch, yaw angles

    Returns:
        ndarray: quaternion in scalar first form
    """
    roll, pitch, yaw = angles

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = np.array(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
        ]
    )
    return q


def from_rotmat(R: np.ndarray):
    """converts a 3x3 orthonormal rotation matrix to a quaternion in scalar first form

    minor adaptions and combinations from ahrs and pytransform3d packages.
    https://dfki-ric.github.io/pytransform3d/index.html
    https://github.com/Mayitzin/ahrs/tree/master

    Args:
        R (np.ndarray): orthognal rotation matrix 3x3 or Nx3x3

    Returns:
        np.ndarray: quaternion in scalar first form
    """
    if R.ndim not in [2, 3]:
        raise ValueError("R must be a 2 or 3 dimensional matrix")
    if R.shape[-2:] != (3, 3):
        raise ValueError(
            f"Function expects a 3x3 or Nx3x3 matrix. You passed a {R.shape} matrix."
        )
    if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6):
        warn("R is not orthogonal")

    num_quats = R.shape[0] if R.ndim == 3 else 1

    q = np.empty((4))
    trace = np.trace(R)

    if trace > 0.0:
        sqrt_trace = np.sqrt(1.0 + trace)
        q[0] = 0.5 * sqrt_trace
        q[1] = 0.5 / sqrt_trace * (R[2, 1] - R[1, 2])
        q[2] = 0.5 / sqrt_trace * (R[0, 2] - R[2, 0])
        q[3] = 0.5 / sqrt_trace * (R[1, 0] - R[0, 1])
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            sqrt_trace = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[0] = 0.5 / sqrt_trace * (R[2, 1] - R[1, 2])
            q[1] = 0.5 * sqrt_trace
            q[2] = 0.5 / sqrt_trace * (R[1, 0] + R[0, 1])
            q[3] = 0.5 / sqrt_trace * (R[0, 2] + R[2, 0])
        elif R[1, 1] > R[2, 2]:
            sqrt_trace = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[0] = 0.5 / sqrt_trace * (R[0, 2] - R[2, 0])
            q[1] = 0.5 / sqrt_trace * (R[1, 0] + R[0, 1])
            q[2] = 0.5 * sqrt_trace
            q[3] = 0.5 / sqrt_trace * (R[2, 1] + R[1, 2])
        else:
            sqrt_trace = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[0] = 0.5 / sqrt_trace * (R[1, 0] - R[0, 1])
            q[1] = 0.5 / sqrt_trace * (R[0, 2] + R[2, 0])
            q[2] = 0.5 / sqrt_trace * (R[2, 1] + R[1, 2])
            q[3] = 0.5 * sqrt_trace

    q /= np.linalg.norm(q, axis=0, keepdims=True)
    return q


def from_axis_angle(ax: np.ndarray, angleFirst=False):
    """convert a rotation in axis angle form to a quaternion in scalar first form

    Args:
        ax (np.ndarray): array containing the unit vectors of the axis and the angle.
        angleFirst (bool, optional): order of elements in ax. Defaults to False.

    Returns:
        np.ndarray: quaternion in scalar first form
    """
    if angleFirst:
        angle = ax[:, 0]
        ax = ax[:, 1:]
    else:
        angle = ax[:, -1]
        ax = ax[:, :-1]

    q = np.empty((ax.shape[0], 4))
    q[:, 0] = np.cos(angle / 2)
    q[:, 1:] = np.sin(angle / 2) * ax / np.linalg.norm(ax, axis=1, keepdims=True)
    return q
