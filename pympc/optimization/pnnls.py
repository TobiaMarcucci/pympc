# external imports
from scipy.optimize import nnls
import numpy as np

def pnnls(A, B, c):
    """
    Solves the Partial Non-Negative Least Squares problem min_{u, v} ||A v + B u - c||_2^2 s.t. v >= 0.
    (See "Bemporad - A Multiparametric Quadratic Programming Algorithm With Polyhedral Computations Based on Nonnegative Least Squares", Lemma 1.)

    Arguments
    ----------
    A : numpy.ndarray
        Coefficient matrix of nonnegative variables.
    B : numpy.ndarray
        Coefficient matrix of remaining variables.
    c : numpy.ndarray
        Offset term.

    Returns
    ----------
    v : numpy.ndarray
        Optimal value of v.
    u : numpy.ndarray
        Optimal value of u.
    r : numpy.ndarray
        Residuals of the least squares problem.
    """

    # matrices for nnls solver
    B_pinv = np.linalg.pinv(B)
    B_bar = np.eye(A.shape[0]) - B.dot(B_pinv)
    A_bar = B_bar.dot(A)
    b_bar = B_bar.dot(c)

    # solve nnls
    v, r = nnls(A_bar, b_bar.flatten())
    v = np.reshape(v, (A.shape[1],1))
    u = - B_pinv.dot(A.dot(v) - c)

    return v, u, r

def linear_program(f, A, b, C=None, d=None, tol=1.e-7):
    """
    Solves the linear program min_x f^T x s.t. A x <= b, C x = d.
    Finds a partially nonnegative least squares solution to the KKT conditions of the LP.

    Math
    ----------
    For the LP min_x f^T x s.t. A x <= b, we can substitute the complementarity condition with the condition of zero duality gap, to get the linear system:
    b' y + f' x = 0,         zero duality gap,
    A x + b = - s,   s >= 0, primal feasibility,
    A' y = f,        y >= 0, dual feasibility,
    where y are the Lagrange multipliers and s are slack variables for the residual of primal feasibility.
    (Each equality constraint is reformulated as two inequalities.)
    
    Arguments
    ----------
    f : numpy.ndarray
        Gradient of the cost function.
    A : numpy.ndarray
        Left-hand side of the inequality constraints.
    b : numpy.ndarray
        Right-hand side of the inequality constraints.
    C : numpy.ndarray
        Left-hand side of the equality constraints.
    d : numpy.ndarray
        Right-hand side of the equality constraints.
    tol : float
        Maximum value for: the residual of the pnnls to consider the problem feasible, for the residual of the inequalities to be considered active.

    Returns
    ----------
    sol : dict
        Dictionary with the solution of the LP.

        Fields
        ----------
        min : float
            Minimum of the LP (None if the problem is unfeasible or unbounded).
        argmin : numpy.ndarray
            Argument that minimized the LP (None if the problem is unfeasible or unbounded).
        active_set : list of int
            Indices of the active inequallities {i | A_i argmin = b} (None if the problem is unfeasible or unbounded).
        multiplier_inequality : numpy.ndarray
            Lagrange multipliers for the inequality constraints (None if the problem is unfeasible or unbounded).
        multiplier_equality : numpy.ndarray
            Lagrange multipliers for the equality constraints (None if the problem is unfeasible or unbounded or without equality constraints).
    """

    # check equalities
    if (C is None) != (d is None):
        raise ValueError("missing C or d.")

    # problem size
    n_ineq, n_x = A.shape
    if C is not None:
        n_eq = C.shape[0]
    else:
        n_eq = 0
    if len(f.shape) == 1:
        f = np.reshape(f, (f.shape[0], 1))
    if len(b.shape) == 1:
        b = np.reshape(b, (b.shape[0], 1))
    if n_eq > 0 and len(d.shape) == 1:
        d = np.reshape(d, (d.shape[0], 1))

    # state equalities as inequalities
    if n_eq > 0:
        AC = np.vstack((A, C, -C))
        bd = np.vstack((b, d, -d))
    else:
        AC = A
        bd = b

    # build and solve pnnls problem
    A_pnnls = np.vstack((
        np.hstack((
            bd.T,
            np.zeros((1, n_ineq + 2*n_eq))
            )),
        np.hstack((
            np.zeros((n_ineq + 2*n_eq, n_ineq + 2*n_eq)),
            np.eye(n_ineq + 2*n_eq)
            )),
        np.hstack((
            AC.T,
            np.zeros((n_x, n_ineq + 2*n_eq))
            ))
        ))
    B_pnnls = np.vstack((f.T, AC, np.zeros((n_x, n_x))))
    c_pnnls = np.vstack((0., bd, -f))
    ys, x, r = pnnls(A_pnnls, B_pnnls, c_pnnls)

    # initialize output
    sol = {
        'min': None,
        'argmin': None,
        'active_set': None,
        'multiplier_inequality': None,
        'multiplier_equality': None
    }

    # fill solution if residual is almost zero
    if r < tol:
        r_ineq = ys[n_ineq+2*n_eq:2*n_ineq+2*n_eq,:]
        sol['min'] = f.T.dot(x)[0,0]
        sol['argmin'] = x
        sol['active_set'] = sorted(list(np.where(r_ineq < tol)[0]))
        sol['multiplier_inequality'] = ys[:n_ineq,:]
        if n_eq > 0:
            mul_eq_pos = ys[n_ineq:n_ineq+n_eq, :]
            mul_eq_neg = - ys[n_ineq+n_eq:n_ineq+2*n_eq, :]
            sol['multiplier_equality'] = mul_eq_pos + mul_eq_neg

    return sol