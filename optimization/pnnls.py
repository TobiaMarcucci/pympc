from scipy.optimize import nnls
import numpy as np

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

def linear_program(f, A=None, b=None, C=None, d=None, tol=1.e-10):
    """
    Solves the linear program
    min  f^T x
    s.t. A x <= b
         C x  = d
         x_lb <= x <= x_ub

    OUTPUTS:
        x_star: argument which minimizes the cost (=nan if the LP is unfeasible or unbounded)
        V_star: minimum of the cost function (=nan if the LP is unfeasible or unbounded)
    """
    
    # empty inequalities if not provided
    n_x = f.shape[0]
    if A is None or b is None:
        A = np.array([[]]*n_x)
        b = np.array([[]])

    # state equalities as inequalities
    if C is not None and d is not None:
        A = np.vstack((A, C, -C))
        b = np.vstack((b, d, -d))

    # problem dimensions
    n_ineq = A.shape[0]

    # matrices for the pnnls solver
    A_pnnls = np.vstack((
        np.hstack((                       b.T,  np.zeros((1, n_ineq)))),
        np.hstack((np.zeros((n_ineq, n_ineq)),         np.eye(n_ineq))),
        np.hstack((                       A.T, np.zeros((n_x, n_ineq))))
        ))
    B_pnnls = np.vstack((f.T, A, np.zeros((n_x, n_x))))
    f = np.reshape(f, (n_x,1))
    c_pnnls = np.vstack((0., b, -f))

    # solve pnnls
    _, x_star, r_star = pnnls(A_pnnls, B_pnnls, c_pnnls)
    if r_star > tol:
        x_star[:] = np.nan
    V_star = (f.T.dot(x_star))[0,0]

    return x_star, V_star