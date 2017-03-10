from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver
import numpy as np


def linear_program(f, A, b, x_bound=1.e9, toll=1.e-3):
    """
    Solves the linear program
    minimize f^T * x
    s. t.    A * x <= b
             ||x||_inf <= x_bound

    INPUTS:
        f: gradient of the cost function (2D numpy array)
        A: left hand side of the constraints (2D numpy array)
        b: right hand side of the constraints (2D numpy array)
        x_bound: bound on the infinity norm of the solution (used to detect unbounded solutions!)
        toll: tollerance in the detection of unbounded solutions

    OUTPUTS:
        x_min: argument which minimizes the cost (its elements are nan if unfeasible and inf if unbounded)
        cost_min: minimum of the cost function (nan if unfeasible and inf if unbounded)
        status: status of the solution (=0 if solved, =1 if unfeasible, =2 if unbounded)
    """

    # program dimensions
    n_variables = f.shape[0]
    n_constraints = A.shape[0]

    # build program
    prog = mp.MathematicalProgram()
    x = prog.NewContinuousVariables(n_variables, "x")
    for i in range(0, n_constraints):
        prog.AddLinearConstraint((A[i,:] + 1e-15).dot(x) <= b[i])
    prog.AddLinearCost((f.flatten() + 1e-15).dot(x))

    # set bounds to the solution
    if x_bound is not None:
        for i in range(0, n_variables):
                prog.AddLinearConstraint(x[i] <= x_bound)
                prog.AddLinearConstraint(x[i] >= -x_bound)

    # solve
    solver = GurobiSolver()
    result = solver.Solve(prog)
    x_min = np.reshape(prog.GetSolution(x), (n_variables,1))
    cost_min = f.T.dot(x_min)
    status = 0

    # unfeasible
    if any(np.isnan(x_min)) or np.isnan(cost_min):
        status = 1
        return [x_min, cost_min, status]

    # unbounded
    x_min[np.where(x_min > x_bound - toll)] = np.inf
    x_min[np.where(x_min < - x_bound + toll)] = -np.inf
    if any(f[np.where(np.isinf(x_min))] != 0.):
        cost_min = -np.inf
        status = 2

    return [x_min, cost_min, status]



def quadratic_program(H, f, A, b, C=None, d=None):
    """
    Solves the convex (i.e., H > 0) quadratic program
    minimize x^T * H * x + f^T * x
    s. t.    A * x <= b
             C * x = d

    INPUTS:
        H: Hessian of the cost function (bidimensional numpy array)
        f: linear term of the cost function (monodimensional numpy array)
        A: left hand side of the inequalities (bidimensional numpy array)
        b: right hand side of the inequalities (monodimensional numpy array)
        C: left hand side of the equalities (bidimensional numpy array)
        d: right hand side of the equalities (monodimensional numpy array)

    OUTPUTS:
        x_min: argument which minimizes the cost (its elements are nan if unfeasible and inf if unbounded)
        cost_min: minimum of the cost function (nan if unfeasible and inf if unbounded)
        status: status of the solution (=0 if solved, =1 if unfeasible)
    """

    # program dimensions
    n_variables = f.shape[0]
    n_constraints = A.shape[0]

    # build program
    prog = mp.MathematicalProgram()
    x = prog.NewContinuousVariables(n_variables, "x")
    for i in range(0, n_constraints):
        prog.AddLinearConstraint((A[i,:] + 1e-15).dot(x) <= b[i])
    if C is not None:
        for i in range(C.shape[0]):
            prog.AddLinearConstraint(C[i, :].dot(x) == d[i])
    prog.AddQuadraticCost(H, f, x)

    # solve
    solver = GurobiSolver()
    result = solver.Solve(prog)
    x_min = np.reshape(prog.GetSolution(x), (n_variables,1))
    cost_min = .5*x_min.T.dot(H.dot(x_min)) + f.T.dot(x_min)
    status = 0

    # unfeasible
    if any(np.isnan(x_min)) or np.isnan(cost_min):
        status = 1

    return [x_min, cost_min, status]
