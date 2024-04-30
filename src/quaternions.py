import numpy as np
from warnings import warn
import numpy as np


def to_scalar_first(q):
    """
    Converts a quaternion from scalar last to scalar first.

    Parameters
    ----------
    q : ndarray
        Quaternion with scalar last.

    Returns
    -------
    ndarray
        Quaternion with scalar first.
    """
    return np.array([q[3], q[0], q[1], q[2]])


def conjugate(q, scalarLast=False):
    """
    Returns the conjugate of a quaternion.

    Parameters
    ----------
    q : ndarray
        Quaternion assuming scalar first.

    Returns
    -------
    ndarray
        Conjugate of q.
    """
    return (
        np.array([q[0], -q[1], -q[2], -q[3]])
        if not scalarLast
        else np.array([-q[0], -q[1], -q[2], q[3]])
    )


def inverse(q, scalarLast=False):
    """
    Returns the inverse of a quaternion.

    Parameters
    ----------
    q : ndarray
        Quaternion assuming scalar first.

    Returns
    -------
    ndarray
        Inverse of q.
    """
    return conjugate(q, scalarLast) / np.linalg.norm(q)


def exponential(q, scalarLast=False):
    """
    Returns the exponential of a quaternion.

    Parameters
    ----------
    q : ndarray
        Quaternion assuming scalar first.

    Returns
    -------
    ndarray
        Exponential of q.
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
    """
    Returns the logarithm of a quaternion.

    Parameters
    ----------
    q : ndarray
        Quaternion.

    Returns
    -------
    ndarray
        Logarithm of q.
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
    """
    Returns the normalized quaternion.

    Parameters
    ----------
    q : ndarray
        Quaternion assuming scalar first.

    Returns
    -------
    ndarray
        Normalized q.

    Notes
    -----
    This function calculates the normalized quaternion.

    """
    return q / np.linalg.norm(q)


def product(q, p, scalarLast=False):
    """
    Returns the Hamilton product of two quaternions.

    Parameters
    ----------
    q : ndarray
        Quaternion assuming scalar first.
    p : ndarray
        Quaternion assuming scalar first.

    Returns
    -------
    ndarray
        Product of q and p.

    Notes
    -----
    This function calculates the product of two quaternions.

    """
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
    """
    Returns the angles of a quaternion.

    Parameters
    ----------
    q : ndarray
        Quaternion assuming scalar first.
    scalarLast : bool, optional
        Flag indicating whether the scalar component is the last element of the quaternion.
        Default is False.

    Returns
    -------
    ndarray
        Angles of the quaternion in phi, theta, psi.

    Notes
    -----
    This function calculates the angles of a quaternion using the formula from the paper
    "Quaternion to Euler Angle Conversion for Arbitrary Rotation Sequence" by Shuster and Oh.
    The implementation is based on the pytransform3d and scipy libraries.

    References
    ----------
    - Shuster, M. D., & Oh, S. D. (1981). Quaternion to Euler angle conversion for arbitrary
      rotation sequence. AIAA Journal, 19(6), 838-839.
    - pytransform3d: https://github.com/rock-learning/pytransform3d
    - scipy: https://www.scipy.org/

    """
    if scalarLast:
        q0, qx, qy, qz = to_scalar_first(q)
    else:
        q0, qx, qy, qz = q.squeeze().tolist()

    phi = np.arctan2(qx + qz, q0 - qy) - np.arctan2(qz - qx, qy + q0)
    theta = np.arccos((q0 - qy) ** 2 + (qx + qz) ** 2 - 1) - np.pi / 2
    psi = np.arctan2(qx + qz, q0 - qy) + np.arctan2(qz - qx, qy + q0)

    return np.array([phi, theta, psi])


def to_rotmat(q, scalarLast=False, homogenous=True):
    """Converts a quaternion to a right hand rotation matrix. Choice of inhomogeneous or homogeneous representation (latter is preferred). If the quaternion is not a unit quaternion, the homogenous representation is still a scalar multiple of a rotation matrix while the inhomogeneous representation is not an orthogonal rotation matrix.

    Parameters
    ----------
    q : ndarray
        unit quaternion in 4x1
    scalarLast : bool, optional
        Flag indicating whether the quaternion is in scalar-last format. Default is False
    homogenous : bool, optional
        Flag indicating whether the rotation matrix should be in homogeneous form. Default is True

    Returns
    -------
    ndarray
        Rotation matrix representation of the original quaternion.
    """
    if scalarLast:
        q0, qx, qy, qz = to_scalar_first(q)
    else:
        q0, qx, qy, qz = q.squeeze().tolist()

    if homogenous:
        R = np.array(
            [
                [
                    q0**2 + qx**2 - qy**2 - qz**2,
                    2 * (qx * qy - q0 * qz),
                    2 * (q0 * qy + qx * qz),
                ],
                [
                    2 * (qx * qy + q0 * qz),
                    q0**2 - qx**2 + qy**2 - qz**2,
                    2 * (qy * qz - q0 * qx),
                ],
                [
                    2 * (qx * qz - q0 * qy),
                    2 * (q0 * qx + qy * qz),
                    q0**2 - qx**2 - qy**2 + qz**2,
                ],
            ]
        )
    else:
        R = np.array(
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
    return R


def from_rpy(angles: np.ndarray, order="rpy"):
    """
    Returns the quaternion from a series of roll (about x axis), pitch (about y axis), and yaw (about z axis) angles.
    From: NASA Mission Planning and Analysis Division (July 1977). "Euler Angles, Quaternions, and Transformation Matrices". NASA
    Parameters
    ----------
    angles : ndarray
        Roll, pitch, yaw angles.
    order : str, optional
        Order of the angles. Default is "rpy" or "XYZ".
    Returns
    -------
    ndarray
        Quaternion in scalar first form.
    """
    roll, pitch, yaw = angles

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    if order == "rpy":
        q = np.array(
            [
                -sr * sp * sy + cr * cp * cy,
                sr * cp * cy + sp * sy * cr,
                -sr * sy * cp + sp * cr * cy,
                sr * sp * cy + sy * cr * cr,
            ]
        )
    elif order == "ryp":
        q = np.array(
            [
                sr * sy * sp + cr * cy * cp,
                sr * cy * cp - sy * sp * cr,
                -sr * sy * cp + sp * cr * cy,
                sr * sp * cy + sy * cr * cr,
            ]
        )

    return q


def from_rotmat(R: np.ndarray):
    """
    Converts a 3x3 orthonormal rotation matrix to a quaternion in scalar first form.

    Parameters
    ----------
    R : np.ndarray
        Orthogonal rotation matrix 3x3 or Nx3x3.

    Returns
    -------
    np.ndarray
        Quaternion in scalar first form.

    Raises
    ------
    ValueError
        If R is not a 2 or 3 dimensional matrix or if R shape is not (3, 3).

    Warns
    -----
    UserWarning
        If R is not orthogonal.

    Notes
    -----
    This function expects a 3x3 or Nx3x3 matrix. If you pass a matrix with a different shape, a ValueError will be raised.
    If R is not orthogonal, a UserWarning will be issued.
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
    """
    Convert a rotation in axis angle form to a quaternion in scalar first form.

    Parameters:
        ax (np.ndarray): Array containing the unit vectors of the axis and the angle.
        angleFirst (bool, optional): Order of elements in ax. Defaults to False.

    Returns:
        np.ndarray: Quaternion in scalar first form.
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
