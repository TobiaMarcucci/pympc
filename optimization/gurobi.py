import gurobipy as grb
import numpy as np

def linear_program(f, A=None, b=None, C=None, d=None, lb=None, ub=None):
    """
    Solves the linear program
    minimize f^T * x
    s. t.    A * x <= b
             C * x  = d
             lb <= x <= ub

    OUTPUTS:
        x_min: argument which minimizes the cost (=nan if the LP is unfeasible or unbounded)
        cost_min: minimum of the cost function (=nan if the LP is unfeasible or unbounded)
    """

    # initialize gurobi model
    model = grb.Model()

    # optimization variables
    n_var = f.shape[0]
    vars = []
    if lb is None:
        lb = [- grb.GRB.INFINITY]*n_var
    if ub is None:
        ub = [grb.GRB.INFINITY]*n_var
    for i in range(n_var):
        vars.append(model.addVar(lb=lb[i], ub=ub[i], vtype=grb.GRB.CONTINUOUS))

    # inequality constraints
    if A is not None and b is not None:
        n_ineq = A.shape[0]
        for i in range(n_ineq):
            expr = grb.LinExpr()
            for j in range(n_var):
                if A[i,j] != 0:
                    expr += A[i,j]*vars[j]
            model.addConstr(expr, grb.GRB.LESS_EQUAL, b[i])

    # equality constraints
    if C is not None and d is not None:
        n_eq = C.shape[0]
        for i in range(n_eq):
            expr = grb.LinExpr()
            for j in range(n_var):
                if C[i,j] != 0:
                    expr += C[i,j]*vars[j]
            model.addConstr(expr, grb.GRB.EQUAL, d[i])

    # cost function
    obj = grb.LinExpr()
    for i in range(n_var):
        if f[i] != 0:
              obj += f[i,0]*vars[i]
    model.setObjective(obj)

    # run the optimization
    model.setParam('OutputFlag', False)
    model.optimize()

    # return the result
    if model.status == grb.GRB.Status.OPTIMAL:
        x_min = np.array(model.getAttr('x', vars)).reshape(n_var,1)
        cost_min = obj.getValue()
    else:
        x_min = np.zeros((n_var,1))
        x_min[:] = np.nan
        cost_min = np.nan
    return x_min, cost_min

def quadratic_program(H, f=None, A=None, b=None, C=None, d=None, lb=None, ub=None):
    """
    Solves the convex (H > 0) quadratic program
    minimize x^T * H * x + f^T * x
    s. t.    A * x <= b
             C * x  = d
             lb <= x <= ub

    OUTPUTS:
        x_min: argument which minimizes the cost (=nan if the LP is unfeasible or unbounded)
        cost_min: minimum of the cost function (=nan if the LP is unfeasible or unbounded)
    """

    # initialize gurobi model
    model = grb.Model()

    # optimization variables
    n_var = H.shape[0]
    vars = []
    if lb is None:
        lb = [- grb.GRB.INFINITY]*n_var
    if ub is None:
        ub = [grb.GRB.INFINITY]*n_var
    for i in range(n_var):
        vars.append(model.addVar(lb=lb[i], ub=ub[i], vtype=grb.GRB.CONTINUOUS))

    # inequality constraints
    if A is not None and b is not None:
        n_ineq = A.shape[0]
        for i in range(n_ineq):
            expr = grb.LinExpr()
            for j in range(n_var):
                if A[i,j] != 0:
                    expr += A[i,j]*vars[j]
            model.addConstr(expr, grb.GRB.LESS_EQUAL, b[i])

    # equality constraints
    if C is not None and d is not None:
        n_eq = C.shape[0]
        for i in range(n_eq):
            expr = grb.LinExpr()
            for j in range(n_var):
                if C[i,j] != 0:
                    expr += C[i,j]*vars[j]
            model.addConstr(expr, grb.GRB.EQUAL, d[i])

    # cost function
    obj = grb.QuadExpr()
    for i in range(n_var):
        for j in range(n_var):
            if H[i,j] != 0:
                obj += .5*H[i,j]*vars[i]*vars[j]
    if f is not None:
        for i in range(n_var):
            if f[i] != 0:
                  obj += f[i]*vars[i]
    model.setObjective(obj)

    # run the optimization
    model.setParam('OutputFlag', False)
    model.optimize()

    # return the result
    if model.status == grb.GRB.Status.OPTIMAL:
        x_min = np.array(model.getAttr('x', vars)).reshape(n_var,1)
        cost_min = obj.getValue()
    else:
        x_min = np.zeros((n_var,1))
        x_min[:] = np.nan
        cost_min = np.nan
    return x_min, cost_min