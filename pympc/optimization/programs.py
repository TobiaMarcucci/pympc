# external imports
import numpy as np

# internal inputs
from pympc.optimization.solvers.pnnls import linear_program as lp_pnnls, quadratic_program as qp_pnnls
from pympc.optimization.solvers.gurobi import linear_program as lp_gurobi, quadratic_program as qp_gurobi, mixed_integer_quadratic_program as miqp_gurobi
# from pympc.optimization.solvers.drake import linear_program as lp_drake, quadratic_program as qp_drake, mixed_integer_quadratic_program as miqp_drake

def linear_program(f, A, b, C=None, d=None, solver='pnnls'):
    """
    Calls the desired solver to solve the linear program min_x f^T x s.t. A x <= b, C x = d.

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
    solver : str
        Name of the solver to be used, available solvers are: 'pnnls', 'gurobi', 'drake'.

    Returns
    ----------
    sol : dict
        Dictionary with the solution of the LP.

        Keys
        ----------
        min : float
            Minimum of the LP (None if the problem is unfeasible or unbounded).
        argmin : numpy.ndarray
            Argument that minimizes the LP (None if the problem is unfeasible or unbounded).
        active_set : list of int
            Indices of the active inequallities {i | A_i argmin = b} (None if the problem is unfeasible or unbounded).
        multiplier_inequality : numpy.ndarray
            Lagrange multipliers for the inequality constraints (None if the problem is unfeasible or unbounded).
        multiplier_equality : numpy.ndarray
            Lagrange multipliers for the equality constraints (None if the problem is unfeasible or unbounded or without equality constraints).
    """
    
    # select solver
    if solver == 'pnnls':
        return lp_pnnls(f, A, b, C, d)
    elif solver == 'gurobi':
        return lp_gurobi(f, A, b, C, d)
    elif solver == 'drake':
        return lp_drake(f, A, b, C, d)
    else:
        raise ValueError('unknown solver ' + str(solver) + '.')

def quadratic_program(H, f, A, b, C=None, d=None, solver='pnnls'):
    """
    Calls the desired solver to solve the strictly convex (H > 0) quadratic program min .5 x' H x + f' x s.t. A x <= b, C x  = d.

    Arguments
    ----------
    H : numpy.ndarray
        Positive definite Hessian of the cost function.
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
    solver : str
        Name of the solver to be used, available solvers are: 'pnnls', 'gurobi', 'drake'.

    Returns
    ----------
    sol : dict
        Dictionary with the solution of the QP.

        Fields
        ----------
        min : float
            Minimum of the QP (None if the problem is unfeasible).
        argmin : numpy.ndarray
            Argument that minimizes the QP (None if the problem is unfeasible).
        active_set : list of int
            Indices of the active inequallities {i | A_i argmin = b} (None if the problem is unfeasible).
        multiplier_inequality : numpy.ndarray
            Lagrange multipliers for the inequality constraints (None if the problem is unfeasible).
        multiplier_equality : numpy.ndarray
            Lagrange multipliers for the equality constraints (None if the problem is unfeasible or without equality constraints).
    """
    
    # select solver
    if solver == 'pnnls':
        return qp_pnnls(H, f, A, b, C, d)
    elif solver == 'gurobi':
        return qp_gurobi(H, f, A, b, C, d)
    elif solver == 'drake':
        return qp_drake(H, f, A, b, C, d)
    else:
        raise ValueError('unknown solver ' + str(solver) + '.')

def mixed_integer_quadratic_program(nc, H, f, A, b, C=None, d=None, solver='gurobi'):
    """
    Calls the desired solver to solve the strictly convex (H > 0) mixed-integer quadratic program min .5 x' H x + f' x s.t. A x <= b, C x  = d.
    The first nc variables in x are continuous, the remaining are binaries.

    Arguments
    ----------
    nc : int
        Number of continuous variables in the problem.
    H : numpy.ndarray
        Positive definite Hessian of the cost function.
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
    solver : str
        Name of the solver to be used, available solvers are: 'pnnls', 'gurobi', 'drake'.

    Returns
    ----------
    sol : dict
        Dictionary with the solution of the MIQP.

        Fields
        ----------
        min : float
            Minimum of the MIQP (None if the problem is unfeasible).
        argmin : numpy.ndarray
            Argument that minimizes the MIQP (None if the problem is unfeasible).
    """

    # select solver
    if solver == 'gurobi':
        return miqp_gurobi(nc, H, f, A, b, C, d)
    if solver == 'drake':
        return miqp_drake(nc, H, f, A, b, C, d)
    else:
        raise ValueError('unknown solver ' + str(solver) + '.')