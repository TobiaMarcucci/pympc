# external imports
import numpy as np

def check_affine_system(A, B, c=None, h=None):
    """
    Check that the matrices A, B, and c of an affine system have compatible sizes.

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
    """

    # A square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix.')

    # equal number of rows for A and B
    if A.shape[0] != B.shape[0]:
        raise ValueError('A and B must have the same number of rows.')

    # check c
    if c is not None:
        if c.ndim > 1:
            raise ValueError('c must be a 1-dimensional array.')
        if A.shape[0] != c.size:
            raise ValueError('A and c must have the same number of rows.')

    # check h
    if h is not None:
        if h < 0:
            raise ValueError('the time step h must be positive.')