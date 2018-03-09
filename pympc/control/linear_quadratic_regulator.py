# external imports
import numpy as np
from scipy.linalg import solve_discrete_are

def dare(A, B, Q, R):
	"""
	Returns the solution of the Discrete Algebraic Riccati Equation (DARE).
	Consider the linear quadratic control problem V*(x(0)) = min_{x(.), u(.)} 1/2 sum_{t=0}^inf x(t)' Q x(t) + u(t)' R u(t) subject to x(t+1) = A x(t) + B u(t).
	The optimal solution is u(0) = K x(0) which leads to V*(x(0)) = 1/2 x(0)' P x(0).
	The pair A, B is assumed to be invertible.

	Arguments
    ----------
    A : numpy.ndarray
        State transition matrix (assumed to be invertible).
    B : numpy.ndarray
        Input to state map.
    Q : numpy.ndarray
        Quadratic cost for the state (positive semidefinite).
    R : numpy.ndarray
        Quadratic cost for the input (positive definite).

    Returns
    ----------
    P : numpy.ndarray
        Hessian of the cost-to-go (positive definite).
    K : numpy.ndarray
        Optimal feedback gain matrix.
	"""

    # cost to go
    P = solve_discrete_are(A, B, Q, R)

    # feedback
    K = - np.linalg.inv(B.T.dot(P).dot(B)+R).dot(B.T).dot(P).dot(A)

    return P, K