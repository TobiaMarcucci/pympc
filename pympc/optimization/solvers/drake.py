# external imports
import numpy as np
from pydrake.all import MathematicalProgram, SolutionResult
from pydrake.solvers.gurobi import GurobiSolver

def linear_program(f, A, b, C=None, d=None, tol=1.e-5):
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

    # reshape inputs
    if len(f.shape) == 2:
        f = np.reshape(f, f.shape[0])

    # build program
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(n_x)
    inequalities = []
    for i in range(n_ineq):
        lhs = A[i,:] + 1.e-20*np.random.rand((n_x)) # drake raises a RuntimeError if the in the expression x does not appear (e.g.: 0 x <= 1)
        rhs = b[i] + 1.e-15*np.random.rand(1) # in case the constraint is 0 x <= 0 the previous trick ends up adding the constraint x <= 0 to the program...
        inequalities.append(prog.AddLinearConstraint(lhs.dot(x) <= rhs))
    for i in range(n_eq):
        prog.AddLinearConstraint(C[i,:].dot(x) == d[i])
    prog.AddLinearCost(f.dot(x))

    # solve
    solver = GurobiSolver()
    prog.SetSolverOption(solver.solver_type(), "OutputFlag", 0)
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
        sol['argmin'] = prog.GetSolution(x).reshape(n_x, 1)
        sol['min'] = f.dot(sol['argmin'])[0]
        sol['active_set'] = np.where(A.dot(sol['argmin']) - b > -tol)[0].tolist()

        # retrieve multipliers through KKT conditions
        M = A[sol['active_set'], :].T
        if n_eq > 0:
            M = np.hstack((M, C.T))
        m = np.linalg.pinv(M).dot(-f.reshape(n_x, 1))
        sol['multiplier_inequality'] = np.zeros((n_ineq, 1))
        for i, j in enumerate(sol['active_set']):
            sol['multiplier_inequality'][j,0] = m[i, :]
        if n_eq > 0:
            sol['multiplier_equality'] = m[len(sol['active_set']):, :]

    return sol

def quadratic_program(H, f, A, b, C=None, d=None, tol=1.e-5):
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

    # reshape inputs
    if len(f.shape) == 2:
        f = np.reshape(f, f.shape[0])

    # build program
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(n_x)
    inequalities = []
    for i in range(n_ineq):
        lhs = A[i,:] + 1.e-15*np.random.rand((n_x)) # drake raises a RuntimeError if the in the expression x does not appear (e.g.: 0 x <= 1)
        rhs = b[i] + 1.e-15*np.random.rand(1) # in case the constraint is 0 x <= 0 the previous trick ends up adding the constraint x <= 0 to the program...
        inequalities.append(prog.AddLinearConstraint(lhs.dot(x) <= rhs))
    for i in range(n_eq):
        prog.AddLinearConstraint(C[i,:].dot(x) == d[i])
    prog.AddQuadraticCost(.5*x.dot(H).dot(x) + f.dot(x))

    # solve
    solver = GurobiSolver()
    prog.SetSolverOption(solver.solver_type(), "OutputFlag", 0)
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
        sol['argmin'] = prog.GetSolution(x).reshape(n_x, 1)
        sol['min'] = .5*sol['argmin'].T.dot(H).dot(sol['argmin'])[0,0] + f.dot(sol['argmin'])[0]
        sol['active_set'] = np.where(A.dot(sol['argmin']) - b > -tol)[0].tolist()

        # retrieve multipliers through KKT conditions
        lhs = A[sol['active_set'], :]
        rhs = b[sol['active_set'], :]
        if n_eq > 0:
            lhs = np.vstack((lhs, C))
            rhs = np.vstack((rhs, d))
        H_inv = np.linalg.inv(H)
        M = lhs.dot(H_inv).dot(lhs.T)
        m = - np.linalg.inv(M).dot(lhs.dot(H_inv).dot(f.reshape(n_x, 1)) + rhs)
        sol['multiplier_inequality'] = np.zeros((n_ineq, 1))
        for i, j in enumerate(sol['active_set']):
            sol['multiplier_inequality'][j,0] = m[i]
        if n_eq > 0:
            sol['multiplier_equality'] = m[len(sol['active_set']):, :]

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

    # reshape inputs
    if len(f.shape) == 2:
        f = np.reshape(f, f.shape[0])

    # build program
    prog = MathematicalProgram()
    x = np.hstack((
        prog.NewContinuousVariables(nc),
        prog.NewBinaryVariables(n_x - nc)
        ))
    inequalities = []
    for i in range(n_ineq):
        inequalities.append(prog.AddLinearConstraint(A[i,:].dot(x) <= b[i]))
    for i in range(n_eq):
        prog.AddLinearConstraint(C[i,:].dot(x) == d[i])
    prog.AddQuadraticCost(.5*x.dot(H).dot(x) + f.dot(x))

    # solve
    solver = GurobiSolver()
    prog.SetSolverOption(solver.solver_type(), "OutputFlag", 0)
    result = prog.Solve()

    # initialize output
    sol = {
        'min': None,
        'argmin': None
    }

    if result == SolutionResult.kSolutionFound:
        sol['argmin'] = prog.GetSolution(x).reshape(n_x, 1)
        sol['min'] = .5*sol['argmin'].T.dot(H).dot(sol['argmin'])[0,0] + f.dot(sol['argmin'])[0]

    return sol