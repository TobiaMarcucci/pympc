# external imports
import numpy as np
from pydrake.all import MathematicalProgram, SolutionResult
from pydrake.solvers.gurobi import GurobiSolver

def linear_program(f, A, b, C=None, d=None, tol=1.e-5, **kwargs):
    """
    Solves the linear program min_x f^T x s.t. A x <= b, C x = d.

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
        Maximum value of a residual of an inequality to consider the related constraint active.

    Returns
    ----------
    sol : dict
        Dictionary with the solution of the LP.

        Fields
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

    # check equalities
    if (C is None) != (d is None):
        raise ValueError('missing C or d.')

    # problem size
    n_ineq, n_x = A.shape
    if C is not None:
        n_eq = C.shape[0]
    else:
        n_eq = 0

    # build program
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(n_x)
    [prog.AddLinearConstraint(A[i].dot(x) <= b[i]) for i in range(n_ineq)]
    [prog.AddLinearConstraint(C[i].dot(x) == d[i]) for i in range(n_eq)]
    prog.AddLinearCost(f.dot(x))

    # solve
    solver = GurobiSolver()
    prog.SetSolverOption(solver.solver_type(), 'OutputFlag', 0)
    [prog.SetSolverOption(solver.solver_type(), parameter, value) for parameter, value in kwargs.items()]
    result = prog.Solve()

    # initialize output
    sol = {
        'min': None,
        'argmin': None,
        'active_set': None,
        'multiplier_inequality': None,
        'multiplier_equality': None
    }

    if result == SolutionResult.kSolutionFound:
        sol['argmin'] = prog.GetSolution(x)
        sol['min'] = f.dot(sol['argmin'])
        sol['active_set'] = np.where(A.dot(sol['argmin']) - b > -tol)[0].tolist()

        # retrieve multipliers through KKT conditions
        M = A[sol['active_set']].T
        if n_eq > 0:
            M = np.hstack((M, C.T))
        m = -np.linalg.pinv(M).dot(f)
        sol['multiplier_inequality'] = np.zeros(n_ineq)
        for i, j in enumerate(sol['active_set']):
            sol['multiplier_inequality'][j] = m[i]
        if n_eq > 0:
            sol['multiplier_equality'] = m[len(sol['active_set']):]

    return sol

def quadratic_program(H, f, A, b, C=None, d=None, tol=1.e-5, **kwargs):
    """
    Solves the strictly convex (H > 0) quadratic program min .5 x' H x + f' x s.t. A x <= b, C x  = d.

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
    tol : float
        Maximum value of a residual of an inequality to consider the related constraint active.

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

    # check equalities
    if (C is None) != (d is None):
        raise ValueError('missing C or d.')

    # problem size
    n_ineq, n_x = A.shape
    if C is not None:
        n_eq = C.shape[0]
    else:
        n_eq = 0

    # build program
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(n_x)
    [prog.AddLinearConstraint(A[i].dot(x) <= b[i]) for i in range(n_ineq)]
    [prog.AddLinearConstraint(C[i].dot(x) == d[i]) for i in range(n_eq)]
    inequalities = []
    prog.AddQuadraticCost(.5*x.dot(H).dot(x) + f.dot(x))

    # solve
    solver = GurobiSolver()
    prog.SetSolverOption(solver.solver_type(), 'OutputFlag', 0)
    [prog.SetSolverOption(solver.solver_type(), parameter, value) for parameter, value in kwargs.items()]
    result = prog.Solve()

    # initialize output
    sol = {
        'min': None,
        'argmin': None,
        'active_set': None,
        'multiplier_inequality': None,
        'multiplier_equality': None
    }

    if result == SolutionResult.kSolutionFound:
        sol['argmin'] = prog.GetSolution(x)
        sol['min'] = .5*sol['argmin'].dot(H).dot(sol['argmin']) + f.dot(sol['argmin'])
        sol['active_set'] = np.where(A.dot(sol['argmin']) - b > -tol)[0].tolist()

        # retrieve multipliers through KKT conditions
        lhs = A[sol['active_set']]
        rhs = b[sol['active_set']]
        if n_eq > 0:
            lhs = np.vstack((lhs, C))
            rhs = np.concatenate((rhs, d))
        H_inv = np.linalg.inv(H)
        M = lhs.dot(H_inv).dot(lhs.T)
        m = - np.linalg.inv(M).dot(lhs.dot(H_inv).dot(f) + rhs)
        sol['multiplier_inequality'] = np.zeros(n_ineq)
        for i, j in enumerate(sol['active_set']):
            sol['multiplier_inequality'][j] = m[i]
        if n_eq > 0:
            sol['multiplier_equality'] = m[len(sol['active_set']):]

    return sol

def mixed_integer_quadratic_program(nc, H, f, A, b, C=None, d=None, **kwargs):
    """
    Solves the strictly convex (H > 0) mixed-integer quadratic program min .5 x' H x + f' x s.t. A x <= b, C x  = d.
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

    # check equalities
    if (C is None) != (d is None):
        raise ValueError('missing C or d.')

    # problem size
    n_ineq, n_x = A.shape
    if C is not None:
        n_eq = C.shape[0]
    else:
        n_eq = 0

    # build program
    prog = MathematicalProgram()
    x = np.hstack((
        prog.NewContinuousVariables(nc),
        prog.NewBinaryVariables(n_x - nc)
        ))
    [prog.AddLinearConstraint(A[i].dot(x) <= b[i]) for i in range(n_ineq)]
    [prog.AddLinearConstraint(C[i].dot(x) == d[i]) for i in range(n_eq)]
    prog.AddQuadraticCost(.5*x.dot(H).dot(x) + f.dot(x))

    # solve
    solver = GurobiSolver()
    prog.SetSolverOption(solver.solver_type(), 'OutputFlag', 0)
    [prog.SetSolverOption(solver.solver_type(), parameter, value) for parameter, value in kwargs.items()]
    result = prog.Solve()

    # initialize output
    sol = {
        'min': None,
        'argmin': None
    }

    if result == SolutionResult.kSolutionFound:
        sol['argmin'] = prog.GetSolution(x)
        sol['min'] = .5*sol['argmin'].dot(H).dot(sol['argmin']) + f.dot(sol['argmin'])

    return sol