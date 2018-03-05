import gurobipy as grb
import numpy as np
from collections import namedtuple

Solution = namedtuple('Solution',  ['min', 'argmin','inequality_multipliers', 'equality_multipliers',  'active_set'])

def linear_program(f, A=None, b=None, C=None, d=None, active_set=None):
    """
    Solves the linear program
    min  f^T x
    s.t. A x <= b
         C x  = d
    warm_start is a list of indices of active inequalities.
    """

    # get model
    model = build_model(f=f, A=A, b=b, C=C, d=d)

    # warm start
    if active_set is not None:
        model = warm_start(model, active_set, A, C)

    # run the optimization
    model.setParam('OutputFlag', 0)
    model.optimize()
    #print model.Runtime

    # return result
    argmin, cost, ineq_mult, eq_mult = reorganize_solution(model, A, C)
    active_set = get_active_set_lp(model, A)

    return Solution(
        argmin = argmin,
        min = cost,
        inequality_multipliers = ineq_mult,
        equality_multipliers = eq_mult,
        active_set = active_set
        )

def quadratic_program(H, f=None, A=None, b=None, C=None, d=None):
    """
    Solves the strictly convex (H > 0) quadratic program
    min  .5 x^T H x + f^T x
    s.t. A x <= b
         C x  = d
    warm_start is a list of indices of active inequalities.
    """
    
    # get model
    model = build_model(H=H, f=f, A=A, b=b, C=C, d=d)

    # run the optimization
    model.setParam('OutputFlag', False)
    model.optimize()

    # return result
    argmin, cost, ineq_mult, eq_mult = reorganize_solution(model, A, C)
    active_set = get_active_set_qp(model, ineq_mult)

    return Solution(
        argmin = argmin,
        min = cost,
        inequality_multipliers = ineq_mult,
        equality_multipliers = eq_mult,
        active_set = active_set
        )

def build_model(H=None, f=None, A=None, b=None, C=None, d=None):

    # initialize model
    model = grb.Model()
    if H is not None:
        n_x = H.shape[0]
    elif f is not None:
        n_x = max(f.shape)
    x = model.addVars(n_x, lb=[- grb.GRB.INFINITY]*n_x)

    # linear inequalities
    if A is not None and b is not None:
        for i, expr in enumerate(linear_expression(A, b, x)):
            model.addConstr(expr <= 0., name='ineq_'+str(i))

    # linear equalities
    if C is not None and d is not None:
        for i, expr in enumerate(linear_expression(C, d, x)):
            model.addConstr(expr == 0., name='eq_'+str(i))

    # cost function
    if H is not None:
        cost = grb.QuadExpr()
        expr = quadratic_expression(H, x)
        cost.add(.5*expr)
    else:
        cost = grb.LinExpr()
    if f is not None:
        f = f.reshape(1, max(f.shape))
        expr = linear_expression(f, np.zeros((1,1)), x)
        cost.add(expr[0])
    model.setObjective(cost)

    return model

def warm_start(model, active_set, A, C):

    # retrieve variables
    model.update()
    x = model.getVars()

    # warm start variable bounds   
    for i in range(len(x)):
        x[i].setAttr('VBasis', 0)

    # warm start inequalities
    if A is not None:
        for i in range(A.shape[0]):
            constr = model.getConstrByName('ineq_'+str(i))
            if i in active_set:
                constr.setAttr('CBasis', -1)
            else:
                constr.setAttr('CBasis', 0)

    # warm start equalities
    if C is not None:
        for i in range(C.shape[0]):
            constr = model.getConstrByName('eq_'+str(i))
            constr.setAttr('CBasis', -1)

    return model

def reorganize_solution(model, A, C):

    # primal solution
    x = model.getVars()
    cost = np.nan
    argmin = np.full((len(x),1), np.nan)
    if model.status == grb.GRB.Status.OPTIMAL:
        cost = model.objVal
        argmin = np.array(model.getAttr('x')).reshape(len(x), 1)

    # dual inequalities
    ineq_mult = None
    if A is not None:
        if model.status == grb.GRB.Status.OPTIMAL:
            ineq_mult = []
            for i in range(A.shape[0]):
                constr = model.getConstrByName('ineq_'+str(i))
                ineq_mult.append(-constr.getAttr('Pi'))
                # if constr.getAttr('CBasis') == -1:
                #     active_set.append(i)
            ineq_mult = np.vstack(ineq_mult)
        else:
            ineq_mult = np.full((A.shape[0],1), np.nan)

    # dual equalities
    eq_mult = None
    if C is not None:
        if model.status == grb.GRB.Status.OPTIMAL:
            eq_mult = []
            for i in range(C.shape[0]):
                constr = model.getConstrByName('eq_'+str(i))
                eq_mult.append(-constr.getAttr('Pi'))
            eq_mult = np.vstack(eq_mult)
        else:
            eq_mult = np.full((C.shape[0],1), np.nan)

    return argmin, cost, ineq_mult, eq_mult

def get_active_set_lp(model, A):
    active_set = None
    if A is not None and model.status == grb.GRB.Status.OPTIMAL:
        active_set = []
        for i in range(A.shape[0]):
            constr = model.getConstrByName('ineq_'+str(i))
            if constr.getAttr('CBasis') == -1:
                active_set.append(i)
    return active_set

def get_active_set_qp(model, ineq_mult, tol=1.e-6):
    active_set = None
    if ineq_mult is not None:
        active_set = []
        for i, mult in enumerate(ineq_mult.flatten().tolist()):
            if mult > tol:
                active_set.append(i)
    return active_set
  
def linear_expression(A, b, x, tol=1.e-7):
    exprs = []
    for i in range(A.shape[0]):
        expr = grb.LinExpr()
        for j in range(A.shape[1]):
            if np.abs(A[i,j]) > tol:
                expr.add(A[i,j]*x[j])
        if np.abs(b[i]) > tol:
            expr.add(-b[i])
        exprs.append(expr)
    return exprs

def quadratic_expression(H, x, tol=1.e-7):
    expr = grb.QuadExpr()
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if np.abs(H[i,j]) > tol:
                expr.add(x[i]*H[i,j]*x[j])
    return expr

def quadratically_constrained_linear_program(f, A=None, b=None, C=None, d=None, P=None, q=None, r=None, tol=1.e-9):

    # initialize gurobi model
    model = grb.Model()

    # optimization variables
    n_x = f.shape[0]
    x = model.addVars(n_x, lb=[- grb.GRB.INFINITY]*n_x, name='x')

    # linear inequalities
    if A is not None and b is not None:
        for i in range(A.shape[0]):
            lhs = grb.LinExpr()
            for j in range(n_x):
                if np.abs(A[i,j]) > tol:
                    lhs.add(A[i,j]*x[j])
            model.addConstr(lhs <= b[i])

    # linear equalities
    if C is not None and d is not None:
        for i in range(C.shape[0]):
            lhs = grb.LinExpr()
            for j in range(n_x):
                if np.abs(C[i,j]) > tol:
                    lhs.add(C[i,j]*x[j])
            model.addConstr(lhs == d[i])

    # quadratic inequalities
    lhs = grb.QuadExpr()
    for i in range(n_x):
        for j in range(n_x):
            if np.abs(P[i,j]) > tol:
                lhs.add(x[i]*P[i,j]*x[j])
    if q is not None:
        for i in range(n_x):
            if np.abs(q[i,0]) > tol:
                lhs.add(q[i,0]*x[i])
    model.addConstr(lhs <= r)

    # cost function
    f = np.reshape(f, (n_x, 1))
    cost = grb.LinExpr()
    for i in range(n_x):
        if np.abs(f[i,0]) > tol:
            cost.add(f[i,0]*x[i])
    model.setObjective(cost)

    # run the optimization
    model.setParam('OutputFlag', False)
    model.setParam(grb.GRB.Param.OptimalityTol, 1.e-9)
    model.setParam(grb.GRB.Param.FeasibilityTol, 1.e-9)
    model.optimize()

    x_star = np.array([[model.getAttr('x', x)[i]] for i in range(n_x)])
    V_star = cost.getValue()

    return x_star, V_star

def read_status(status):
    status = str(status)
    table = {
        '2': 'OPTIMAL',
        '3': 'INFEASIBLE',
        '4': 'INFEASIBLE OR UNBOUNDED',
        '9': 'TIME LIMIT',
        '11': 'INTERRUPTED',
        '13': 'SUBOPTIMAL',
        }
    return table[status]

def real_variable(model, d_list, **kwargs):
    """
    Creates a Gurobi variable with dimension d_list (e.g., [3,4,5]) with minus infinity as lower bounds.
    """
    lb_x = [-grb.GRB.INFINITY]
    for d in d_list:
        lb_x = [lb_x * d]
    x = model.addVars(*d_list, lb=lb_x, **kwargs)
    return x, model

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