import pytest
import numpy as np
from src import quaternions as quat


# // Validate input-output features of functions
def test_incorrect_type():
    q = "not a quaternion"
    with pytest.raises(TypeError):
        quat.normalize(q)
    with pytest.raises(TypeError):
        quat.product(q, q)


# // Test product function
def test_product_identity():
    q = np.array([1, 2, 3, 4])
    q_inv = np.array([1, -2, -3, -4]) / 30  # inverse of q
    result = quat.product(q, q_inv)
    identity = np.array([1, 0, 0, 0])
    assert np.allclose(
        result, identity
    ), "Product of quaternion and its inverse is not identity"


def test_product_nDim():
    q_5 = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AttributeError):
        quat.product(q_5, q_5)
    q_nd = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    assert q_nd.shape == q_nd.shape


def test_product_zero():
    q = np.array([1, 2, 3, 4])
    zero = np.array([0, 0, 0, 0])
    result = quat.product(q, zero)
    assert np.allclose(result, zero), "Product with zero quaternion is not zero"


def test_product_not_commutative():
    q1 = np.array([1, 2, 3, 4])
    q2 = np.array([5, 6, 7, 8])
    assert not np.allclose(
        quat.product(q1, q2), quat.product(q2, q1)
    ), "Quaternion multiplication is commutative"


# // Test normalize function


def test_normalize():
    q = np.array([1, 2, 3, 4])
    q_normalized = quat.normalize(q)
    length = np.sqrt(np.sum(np.square(q_normalized)))
    assert np.isclose(length, 1), "Length of normalized quaternion is not 1"


def test_normalize_nDim():
    q_5 = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AttributeError):
        quat.normalize(q_5)
    q_nd = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    assert q_nd.shape == q_nd.shape


def test_normalize_zero():
    q = np.array([0, 0, 0, 0])
    with pytest.raises(ArithmeticError):
        quat.normalize(q)


def test_normalize_results():
    q = np.array([1, 2, 3, 4])
    q_normalized = quat.normalize(q)
    print(np.allclose(q_normalized, q / np.sqrt(np.sum(np.square(q)))))
    assert np.allclose(
        q_normalized, q / np.sqrt(np.sum(np.square(q)))
    ), "Normalization of quaternion is incorrect"
