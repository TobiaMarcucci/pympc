# external imports
import numpy as np
from scipy.linalg import expm

# internal imports
from pympc.dynamics.utils import check_affine_system

def explicit_euler(A, B, c, h):
    """
    Discretizes the continuous-time affine system dx/dt = A x + B u + c approximating x(t+1) with x(t) + h dx/dt(t).

    Arguments
    ----------
    A : numpy.ndarray
        State transition matrix.
    B : numpy.ndarray
        Input to state map.
    c : numpy.ndarray
        Offset term.
    h : float
        Discretization time step.

    Returns
    ----------
    A_d : numpy.ndarray
        Discrete-time state transition matrix.
    B_d : numpy.ndarray
        Discrete-time input to state map.
    c_d : numpy.ndarray
        Discrete-time offset term.
    """

    # check inputs
    check_affine_system(A, B, c, h)

    # discretize
    A_d = A*h + np.eye(A.shape[0])
    B_d = B*h
    c_d = c*h

    return A_d, B_d, c_d

def zero_order_hold(A, B, c, h):
    """
    Assuming piecewise constant inputs, it returns the exact discretization of the affine system dx/dt = A x + B u + c.

    Math
    ----------
    Solving the differential equation, we have
    x(h) = exp(A h) x(0) + int_0^h exp(A (h - t)) (B u(t) + c) dt.
    Being u(t) = u(0) constant between 0 and h we have
    x(h) = A_d x(0) + B_d u(0) + c_d,
    where
    A_d := exp(A h),
    B_d := int_0^h exp(A (h - t)) dt B,
    c_d = B_d := int_0^h exp(A (h - t)) dt c.
    I holds
         |A B c|      |A_d B_d c_d|
    exp (|0 0 0| h) = |0   I   0  |
         |0 0 0|      |0   0   1  |
    where both the matrices are square.
    Proof: apply the definition of exponential and note that int_0^h exp(A (h - t)) dt = sum_{k=1}^inf A^(k-1) h^k/k!.

    Arguments
    ----------
    A : numpy.ndarray
        State transition matrix.
    B : numpy.ndarray
        Input to state map.
    c : numpy.ndarray
        Offset term.
    h : float
        Discretization time step.

    Returns
    ----------
    A_d : numpy.ndarray
        Discrete-time state transition matrix.
    B_d : numpy.ndarray
        Discrete-time input to state map.
    c_d : numpy.ndarray
        Discrete-time offset term.
    """

    # check inputs
    check_affine_system(A, B, c, h)

    # system dimensions
    n_x = np.shape(A)[0]
    n_u = np.shape(B)[1]

    # zero order hold
    M_c = np.vstack((
        np.column_stack((A, B, c)),
        np.zeros((n_u+1, n_x+n_u+1))
        ))
    M_d = expm(M_c * h)

    # discrete time dynamics
    A_d = M_d[:n_x, :n_x]
    B_d = M_d[:n_x, n_x:n_x+n_u]
    c_d = M_d[:n_x, n_x+n_u]

    return A_d, B_d, c_d