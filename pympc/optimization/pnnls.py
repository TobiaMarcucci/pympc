from scipy.optimize import nnls
import numpy as np
from collections import namedtuple
import scipy.io
import time

LPSolution = namedtuple('LPSolution',  ['argmin', 'min', 'active_set', 'inequality_multipliers', 'equality_multipliers', 'primal_degenerate', 'dual_degenerate'])
QPSolution = namedtuple('QPSolution',  ['argmin', 'min'])


def pnnls(A, B, c):
    """
    Solves the Partial Non-Negative Least Squares problem
    min  ||A v + B u - c||_2^2
    s.t. v >= 0
    ("Bemporad - A Multiparametric Quadratic Programming Algorithm With Polyhedral Computations Based on Nonnegative Least Squares", Lemma 1.)

    OUTPUTS:
        v_star: optimal value of v
        u_star: optimal value of u
        r_star: residual of the least squares problem
    """

    # problem dimensions
    [n_ineq, n_v] = A.shape
    n_u = B.shape[1]

    # matrices for nnls solver
    B_pinv = np.linalg.pinv(B)
    B_bar = np.eye(n_ineq) - B.dot(B_pinv)
    A_bar = B_bar.dot(A)
    b_bar = B_bar.dot(c)

    # solve nnls
    [v_star, r_star] = nnls(A_bar, b_bar.flatten())
    v_star = np.reshape(v_star, (n_v,1))
    u_star = - B_pinv.dot(A.dot(v_star) - c)

    return v_star, u_star, r_star

def linear_program(f, A=None, b=None, C=None, d=None, tol=1.e-7):
    """
    Solves the linear program
    min  f^T x
    s.t. A x <= b
         C x  = d

    OUTPUTS:
        x_star: argument which minimizes the cost (=nan if the LP is unfeasible or unbounded)
        V_star: minimum of the cost function (=nan if the LP is unfeasible or unbounded)
    """

    # empty inequalities if not provided
    n_x = f.shape[0]
    if A is None or b is None:
        A = np.zeros((0, n_x))
        b = np.zeros((0, 1))
    n_ineq = A.shape[0]

    # state equalities as inequalities
    n_eq = 0
    if C is not None and d is not None:
        n_eq = C.shape[0]
        A = np.vstack((A, C, -C))
        b = np.vstack((b, d, -d))

    # matrices for the pnnls solver
    A_pnnls = np.vstack((
        np.hstack((                                     b.T,   np.zeros((1, n_ineq+2*n_eq)))),
        np.hstack((np.zeros((n_ineq+2*n_eq, n_ineq+2*n_eq)),          np.eye(n_ineq+2*n_eq))),
        np.hstack((                                     A.T, np.zeros((n_x, n_ineq+2*n_eq))))
        ))
    B_pnnls = np.vstack((f.T, A, np.zeros((n_x, n_x))))
    f = np.reshape(f, (n_x,1))
    c_pnnls = np.vstack((0., b, -f))

    # initialize output
    argmin = np.full((n_x,1), np.nan)
    V_star = np.nan
    active_set = None
    mult_ineq = np.full((n_ineq,1), np.nan)
    mult_eq = np.full((n_eq,1), np.nan)
    primal_degenerate = None
    dual_degenerate = None

    # solve pnnls
    try:
        ys_star, x_star, r_star = pnnls(A_pnnls, B_pnnls, c_pnnls)

        # populate output
        if r_star < tol:
            argmin = x_star
            V_star = (f.T.dot(x_star))[0,0]

            # inequality constraints
            if A is not None and b is not None:
                mult_ineq = ys_star[:n_ineq,:]
                residuals_ineq = ys_star[n_ineq+2*n_eq:2*n_ineq+2*n_eq,:]
                active_set = sorted(list(np.where(residuals_ineq < tol)[0]))
                primal_degenerate = len(active_set) > n_x - n_eq
                dual_degenerate = len(list(np.where(mult_ineq < tol)[0])) > n_ineq - n_x + n_eq

            # equality constraints
            if C is not None and d is not None:
                mult_eq = ys_star[n_ineq:n_ineq+n_eq,:] - ys_star[n_ineq+n_eq:n_ineq+2*n_eq,:]

        sol = LPSolution(
            argmin = argmin,
            min = V_star,
            active_set = active_set,
            inequality_multipliers = mult_ineq,
            equality_multipliers = mult_eq,
            primal_degenerate = primal_degenerate,
            dual_degenerate = dual_degenerate)

    # sometimes the nnls algorithms excedes the maximum number of iterations...
    except RuntimeError:
        # print('Too many iterations in PNNLS LP solver, switched to gurobi.')
        from gurobi import linear_program as lp_gurobi
        sol = lp_gurobi(f, A, b, C, d)

    return sol

def quadratic_program(H, f=None, A=None, b=None, C=None, d=None, tol=1.e-7):
    """
    Solves the strictly convex (H > 0) quadratic program
    min  .5 x^T H x + f^T x
    s.t. A x <= b
         C x  = d

    OUTPUTS:
        x_star: argument which minimizes the cost (=nan if the LP is unfeasible or unbounded)
        V_star: minimum of the cost function (=nan if the LP is unfeasible or unbounded)
    """

    try:
        if((C is not None) and (d is not None)):
            A = np.vstack((A,C,-C))
            b = np.vstack((b,d,-d))
        n_x = H.shape[1]
        L = np.linalg.cholesky(H)
        L = L.T
        M = A.dot(np.linalg.inv(L))
        f = np.reshape(f, (n_x,1))
        d = b + A.dot(np.linalg.inv(H)).dot(f)
        gamma = 1
        lhs_nnls = np.vstack((-M.T,-d.T))
        rhs_nnls = np.vstack((np.zeros((n_x,1)),gamma)).flatten()
        # scipy.io.savemat('nnls_matrices_' + str(time.time()), {'A': lhs_nnls, 'b': rhs_nnls})
        [y_star, rvalue_star] = nnls(lhs_nnls, rhs_nnls)
        y_star = np.reshape(y_star,(len(y_star),1))
        r_star = np.vstack((-M.T,-d.T)).dot(y_star)-np.vstack((np.zeros((n_x,1)),gamma))
        if (np.linalg.norm(r_star)<tol):
            x_star = np.full((n_x,1), np.nan)
            V_star = np.nan
        else:
            x_star = -np.linalg.inv(H).dot(f+A.T.dot(y_star)/(gamma+d.T.dot(y_star)))
            V_star = (0.5 * x_star.T.dot(H).dot(x_star) + f.T.dot(x_star))[0,0]
        sol = QPSolution(
            argmin = x_star,
            min = V_star)

    except RuntimeError:
        # print('Too many iterations in PNNLS LP solver, switched to gurobi.')
        from gurobi import quadratic_program as qp_gurobi
        sol = qp_gurobi(H, f, A, b, C, d)

    return sol
