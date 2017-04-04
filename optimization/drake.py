from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver
from pydrake.solvers.mosek import MosekSolver
import numpy as np

def linear_program(f, A, b, C=None, d=None, x_bound=None, solver='gurobi', **kwargs):
    """
    Solves the linear program
    minimize f^T * x
    s. t.    A * x <= b
             C * x  = d

    INPUTS:
        f: gradient of the cost function (2D numpy array)
        A: left hand side of the constraints (2D numpy array)
        b: right hand side of the constraints (2D numpy array)
        C: left hand side of the equalities (2D numpy array)
        d: right hand side of the equalities (2D numpy array)

    OUTPUTS:
        x_min: argument which minimizes the cost (its elements are nan if unfeasible or unbounded)
        cost_min: minimum of the cost function (nan if unfeasible or unbounded)
    """

    # program dimensions
    n_variables = f.shape[0]
    n_constraints = A.shape[0]

    # build program
    prog = mp.MathematicalProgram()
    x = prog.NewContinuousVariables(n_variables, "x")
    for i in range(0, n_constraints):
        prog.AddLinearConstraint((A[i,:] + 1e-15).dot(x) <= b[i])
    if C is not None and d is not None:
        for i in range(C.shape[0]):
            prog.AddLinearConstraint(C[i, :].dot(x) == d[i])
    prog.AddLinearCost((f.flatten() + 1e-15).dot(x))

    # options
    if solver == 'gurobi':
        for (key, value) in kwargs.items():
            prog.SetSolverOption(mp.SolverType.kGurobi, key, value)

    # set bounds to the solution
    if x_bound is not None:
        for i in range(0, n_variables):
                prog.AddLinearConstraint(x[i] <= x_bound)
                prog.AddLinearConstraint(x[i] >= -x_bound)

    # solve
    if solver == 'gurobi':
        solver = GurobiSolver()
    elif solver == 'mosek':
        solver = MosekSolver()
    result = solver.Solve(prog)
    x_min = np.reshape(prog.GetSolution(x), (n_variables,1))
    cost_min = f.T.dot(x_min)

    return [x_min, cost_min]

def quadratic_program(H, f, A, b, C=None, d=None):
    """
    Solves the convex (i.e., H > 0) quadratic program
    minimize x^T * H * x + f^T * x
    s. t.    A * x <= b
             C * x  = d

    INPUTS:
        H: Hessian of the cost function (2D numpy array)
        f: linear term of the cost function (2D numpy array)
        A: left hand side of the inequalities (2D numpy array)
        b: right hand side of the inequalities (2D numpy array)
        C: left hand side of the equalities (2D numpy array)
        d: right hand side of the equalities (2D numpy array)

    OUTPUTS:
        x_min: argument which minimizes the cost (its elements are nan if unfeasible or unbounded)
        cost_min: minimum of the cost function (nan if unfeasible or unbounded)
    """

    # program dimensions
    n_variables = f.shape[0]
    n_constraints = A.shape[0]

    # build program
    prog = mp.MathematicalProgram()
    x = prog.NewContinuousVariables(n_variables, "x")
    for i in range(0, n_constraints):
        prog.AddLinearConstraint((A[i,:] + 1e-15).dot(x) <= b[i])
    if C is not None and d is not None:
        for i in range(C.shape[0]):
            prog.AddLinearConstraint(C[i, :].dot(x) == d[i])
    prog.AddQuadraticCost(H, f, x)

    # solve
    solver = GurobiSolver()
    result = solver.Solve(prog)
    x_min = np.reshape(prog.GetSolution(x), (n_variables,1))
    cost_min = .5*x_min.T.dot(H.dot(x_min)) + f.T.dot(x_min)

    return [x_min, cost_min]
