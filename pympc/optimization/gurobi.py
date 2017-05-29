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

    with suppress_stdout():

        # inequality constraints
        if A is not None and b is not None:
            expr = A.dot(x_np) - b
            model.addConstrs((expr[i,0] <= 0. for i in range(A.shape[0])))

        # equality constraints
        if C is not None and d is not None:
            expr = C.dot(x_np) - d
            model.addConstrs((expr[i,0] == 0. for i in range(C.shape[0])))

        # cost function
        f = np.reshape(f, (n_x, 1))
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
        x_star = np.full((n_x,1), np.nan)
        V_star = np.nan

    return x_star, V_star

def quadratic_program(H, f=None, A=None, b=None, C=None, d=None, x_lb=None, x_ub=None):
    """
    Solves the strictly convex (H > 0) quadratic program
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
    f = np.reshape(f, (n_x, 1))
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
        x_star = np.full((n_x,1), np.nan)
        V_star = np.nan
        
    return x_star, V_star

def real_variable(model, d_list):
    """
    Creates a Gurobi variable with dimension d_list (e.g., [3,4,5]) with minus infinity as lower bounds.
    """
    lb_x = [-grb.GRB.INFINITY]
    for d in d_list:
        lb_x = [lb_x * d]
    x = model.addVars(*d_list, lb=lb_x, name='x')
    return x, model

def point_inside_polyhedron(model, A, b, x):
    """
    Adds to the model the constraint
    x \in P
    where P := {x | A x <= b} is a polyhedron.
    """
    # number of facets
    n_f = A.shape[0]

    # constraint
    x_np = np.array([[x[k]] for k in range(len(x))])
    expr = A.dot(x_np) - b
    model.addConstrs((expr[k,0] <= 0. for k in range(n_f)))

    return model

# def iff_point_in_polyhedron(model, A, b, x, X, eps=0.):
#     """
#     Adds to the model the logical constraint
#     [d = 1] <-> [A x <= b]
#     where P := {x | A x <= b} is a polyhedron.
#     """

#     # inputs
#     n_f = A.shape[0]
#     x_np = np.array([[x[i]] for i in range(len(x))])

#     # bigMs
#     alpha = 0.
#     beta = 0.
#     for i in range(n_f):
#         alpha_i = linear_program(A[i,:], X.lhs_min, X.rhs_min)[1] - b[i,0]
#         beta_i = - linear_program(-A[i,:], X.lhs_min, X.rhs_min)[1] - b[i,0]
#         alpha = min(alpha, alpha_i)
#         beta = max(beta, beta_i)

#     # check bigM
#     if alpha == 0:
#         raise ValueError('Disjunct polyhedron and domain.')
#     if beta == 0:
#         raise ValueError('Domain included in the polyhedron.')

#     # binary variable
#     d = model.addVar(vtype=grb.GRB.BINARY)

#     # slack variable
#     s = model.addVar(lb=-grb.GRB.INFINITY)

#     # MI constraints
#     expr = A.dot(x_np) - b
#     model.addConstrs((s >= expr[k,0] for k in range(expr.shape[0])))
#     model.addConstr(d >= s/alpha + eps)
#     model.addConstr(d <= 1 - s/beta)
#     # model.addConstr(d >= (s-1.)/(alpha-1.))
#     # model.addConstr(d <= (s-beta)/(1.-beta))
    
#     return model, d, s

def iff_point_in_halfspace(model, A, b, x, X, eps=1.e-4):
    """
    Adds to the model the logical constraint
    [d = 1] <-> [A x <= b]
    where H := {x | A x <= b} is an halfspace.
    """
    x_np = np.array([[x[i]] for i in range(len(x))])
    d = model.addVar(vtype=grb.GRB.BINARY)
    model.update()
    m = linear_program(A.T, X.lhs_min, X.rhs_min)[1] - b[0,0]
    M = - linear_program(- A.T, X.lhs_min, X.rhs_min)[1] - b[0,0]
    expr = (A.dot(x_np) - b)[0,0]
    model.addConstr(expr >= m*d + eps)
    model.addConstr(expr <= M*(1.-d))
    return model, d


def iff_point_in_polyhedron(model, A, b, x, X):
    """
    Adds to the model the logical constraint
    [d = 1] <-> [A x <= b]
    where P := {x | A x <= b} is a polyhedron.
    """
    x_np = np.array([[x[i]] for i in range(len(x))])
    d = model.addVar(vtype=grb.GRB.BINARY)
    model.update()
    n_f = A.shape[0]
    d_list = []
    for i in range(n_f):
        model, d_i = iff_point_in_halfspace(model, A[i:i+1,:], b[i:i+1,:], x, X)
        d_list.append(d_i)
    model.addConstr(d >= sum(d_list) - n_f + .5)
    model.addConstr(d <= sum(d_list)/n_f)

    return model, d, d_list


import sys, os
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
