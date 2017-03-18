from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver
from pydrake.solvers.mosek import MosekSolver
from scipy.optimize import nnls


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
    # print B_bar
    A_bar = B_bar.dot(A)
    b_bar = B_bar.dot(c)
    [v_star, r_star] = nnls(A_bar, b_bar.flatten())
    v_star = np.reshape(v_star, (n_v,1))
    u_star = -B_pinv.dot(A.dot(v_star) - c)
    return [v_star, u_star, r_star]

def linear_program_pnnls(f, A, b, C=None, d=None, x_bound=None, toll=1.e-10):
    # remove equalities
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
    cost_min = f.T.dot(x_min)
    return [x_min, cost_min]


        








