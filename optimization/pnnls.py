from scipy.optimize import nnls
import numpy as np

def pnnls(A, B, c):
    """
    Solves the Partial Non-Negative Least Squares problem
    minimize ||A*v + B*u - c||_2^2
    s.t.     v >= 0
    through a NNLS solver.
    (From "Bemporad - A Multiparametric Quadratic Programming Algorithm With Polyhedral Computations Based on Nonnegative Least Squares", Lemma 1.)

    INPUTS:
        A: coefficient matrix for v in the PNNLS problem
        B: coefficient matrix for u in the PNNLS problem
        c: offset term in the PNNLS problem

    OUTPUTS:
        v_star: optimal values of v
        u_star: optimal values of u
        r_star: minimum of least squares
    """
    [n_ineq, n_v] = A.shape
    n_u = B.shape[1]
    B_pinv = np.linalg.pinv(B)
    B_bar = np.eye(n_ineq) - B.dot(B_pinv)
    A_bar = B_bar.dot(A)
    b_bar = B_bar.dot(c)
    [v_star, r_star] = nnls(A_bar, b_bar.flatten())
    v_star = np.reshape(v_star, (n_v,1))
    u_star = -B_pinv.dot(A.dot(v_star) - c)
    return [v_star, u_star, r_star]

def linear_program(f, A, b, C=None, d=None, toll=1.e-10):
    """
    Solves the linear program
    minimize f^T * x
    s. t.    A * x <= b
             C * x  = d

    OUTPUTS:
        x_min: argument which minimizes the cost (its elements are nan if unfeasible or unbounded)
        cost_min: minimum of the cost function (nan if unfeasible or unbounded)
    """

    A_augmented = A
    b_augmented = b
    if C is not None and d is not None:
        A_augmented = np.vstack((A, C, -C))
        b_augmented = np.vstack((b, d, -d))
    A_pnnls = np.vstack((
        np.hstack((b_augmented.T, np.zeros((1, b_augmented.shape[0])))),
        np.hstack((np.zeros((b_augmented.shape[0], b_augmented.shape[0])).T, np.eye(b_augmented.shape[0]))),
        np.hstack((A_augmented.T, np.zeros((A_augmented.shape[1], b_augmented.shape[0]))))
        ))
    B_pnnls = np.vstack((f.T, A_augmented, np.zeros((A_augmented.shape[1], A_augmented.shape[1]))))
    f = np.reshape(f, (f.shape[0],1))
    c_pnnls = np.vstack((np.zeros((1, 1)), b_augmented, -f))
    _, x_min, r_star = pnnls(A_pnnls, B_pnnls, c_pnnls)
    if r_star > toll:
        x_min[:] = np.nan
    cost_min = (f.T.dot(x_min))[0,0]
    return [x_min, cost_min]