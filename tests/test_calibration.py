import pytest
import numpy as np
from src import calibration


@pytest.fixture
def make_calibration_data():
    data_nd = np.full((10, 3), 1.0)
    data_1d = np.full((1, 3), 1.0)
    return data_nd, data_1d


def test_set_vertical_axis(make_calibration_data):
    data_nd, data_1d = make_calibration_data
    data_nd *= np.array([[9.81, 0.01, 1.2]])
    axis_neg_grav = calibration._set_vertical_axis(data_nd, opposite_to_grav=True)
    assert axis_neg_grav[0] < 0, "Vertical axis is positive but should be negative"
    axis_pos_grav = calibration._set_vertical_axis(data_nd, opposite_to_grav=True)
    assert axis_pos_grav[0] < 0, "Vertical axis is negative but should be positive"
    assert np.allclose(
        np.linalg.norm(axis_neg_grav), 1.0
    ), f"Axis is not normalized, the norm is {np.linalg.norm(axis_neg_grav)}"
    assert np.allclose(
        np.linalg.norm(axis_pos_grav), 1.0
    ), f"Axis is not normalized, the norm is {np.linalg.norm(axis_pos_grav)}"


def test_set_pelvis_axes(make_calibration_data):
    data_nd, data_1d = make_calibration_data

    forward_lean_data = (data_nd * np.array([[0.5, 0.1, 0.4]])) / np.linalg.norm(
        np.array([[0.5, 0.1, 0.4]])
    )

    vert_axis = np.array([0.9, 0.1, 0.3]) / np.linalg.norm(np.array([0.9, 0.1, 0.3]))
    R = calibration._set_pelvis_axes(forward_lean_data, vert_axis)
    assert np.allclose(R @ R.T, np.eye(3)), "Rotation matrix is not orthogonal"
    assert np.allclose(
        np.linalg.det(R), 1.0
    ), f"Rotation matrix determinant is not 1, it is {np.linalg.det(R)}"


def test_axis_cross_product():
    axis1 = np.array([1, 0, 0])
    axis2 = np.array([0, 1, 0])
    cross_product = calibration._axis_cross_product(axis1, axis2)
    assert np.allclose(
        cross_product, np.array([0, 0, 1])
    ), "Cross product not orthogonal to axis 1 and 2"
