import gurobipy as grb
import numpy as np

def linear_program(f, A=None, b=None, C=None, d=None, x_lb=None, x_ub=None):
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

    # initialize gurobi model
    model = grb.Model()

    # optimization variables
    n_x = f.shape[0]
    if x_lb is None:
        x_lb = [- grb.GRB.INFINITY]*n_x
    if x_ub is None:
        x_ub = [grb.GRB.INFINITY]*n_x
    x = model.addVars(n_x, lb=x_lb, ub=x_ub, name='x')
    x_np = np.array([[x[i]] for i in range(n_x)])

    # inequality constraints
    if A is not None and b is not None:
        expr = A.dot(x_np) - b
        model.addConstrs((expr[i,0] <= 0. for i in range(A.shape[0])))

    # equality constraints
    if C is not None and d is not None:
        expr = C.dot(x_np) - d
        model.addConstrs((expr[i,0] == 0. for i in range(C.shape[0])))

    # cost function
    V = f.T.dot(x_np)[0,0]
    model.setObjective(V)

    # run the optimization
    model.setParam('OutputFlag', False)
    model.optimize()

    # return the result
    if model.status == grb.GRB.Status.OPTIMAL:
        x_star = np.array([[model.getAttr('x', x)[i]] for i in range(n_x)])
        V_star = V.getValue()
    else:
        x_star = np.zeros((n_x,1))
        x_star[:] = np.nan
        V_star = np.nan
    return x_star, V_star

def quadratic_program(H, f=None, A=None, b=None, C=None, d=None, x_lb=None, x_ub=None):
    """
    Solves the convex (H > 0) quadratic program
    min  .5 x^T H x + f^T x
    s.t. A x <= b
         C x  = d
         x_lb <= x <= x_ub

    OUTPUTS:
        x_star: argument which minimizes the cost (=nan if the LP is unfeasible or unbounded)
        V_star: minimum of the cost function (=nan if the LP is unfeasible or unbounded)
    """

    # initialize gurobi model
    model = grb.Model()

    # optimization variables
    n_x = H.shape[0]
    if x_lb is None:
        x_lb = [- grb.GRB.INFINITY]*n_x
    if x_ub is None:
        x_ub = [grb.GRB.INFINITY]*n_x
    x = model.addVars(n_x, lb=x_lb, ub=x_ub, name='x')
    x_np = np.array([[x[i]] for i in range(n_x)])

    # inequality constraints
    if A is not None and b is not None:
        expr = A.dot(x_np) - b
        model.addConstrs((expr[i,0] <= 0. for i in range(A.shape[0])))

    # equality constraints
    if C is not None and d is not None:
        expr = C.dot(x_np) - d
        model.addConstrs((expr[i,0] == 0. for i in range(C.shape[0])))

    # cost function
    V = .5*x_np.T.dot(H).dot(x_np)[0,0] + f.T.dot(x_np)[0,0]
    model.setObjective(V)

    # run the optimization
    model.setParam('OutputFlag', False)
    model.optimize()

    # return the result
    if model.status == grb.GRB.Status.OPTIMAL:
        x_star = np.array([[model.getAttr('x', x)[i]] for i in range(n_x)])
        V_star = V.getValue()
    else:
        x_star = np.zeros((n_x,1))
        x_star[:] = np.nan
        V_star = np.nan
    return x_star, V_star