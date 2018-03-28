import numpy as np
from pydrake.all import MathematicalProgram, SolutionResult
from pydrake.solvers.gurobi import GurobiSolver

def linear_program(f, A, b, C=None, d=None):

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
    x = prog.NewContinuousVariables(n_x, "x")
    inequlities = []
    for i in range(n_ineq):
        inequlities.append(prog.AddLinearConstraint(A[i,:].dot(x) <= b[i]))
    for i in range(n_eq):
        prog.AddLinearConstraint(C[i,:].dot(x) == d[i])
    prog.AddLinearCost(f.dot(x))
    """
    Would it be possible to just say prog.AddLinearConstraint(A.dot(x) <= b)?
    In c++ one can write prog.AddLinearConstraint((A * x).array() <= b.array()); with A Eigen::Matrix and b Eigen::Vector2d.
    """

    # solve
    solver = GurobiSolver()
    if not solver.available():
        print("Couldn't set up Gurobi :(")
    else:
        prog.SetSolverOption(solver.solver_type(), "OutputFlag", 1)
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
        #print prog.EvalBindingAtSolution(inequlities[0])
        #sol['min'] = prog.GetOptimalCost()
        sol['argmin'] = prog.GetSolution(x)

    return sol


f = np.ones((2,1))
A = - np.eye(2)
b = np.zeros((2,1))
sol = linear_program(f,A,b)
print sol['argmin']