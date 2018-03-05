# external imports
import numpy as np
import gurobipy as grb

def linear_program(f, A, b, C=None, d=None):
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

    # get model
    model = _build_model(f=f, A=A, b=b, C=C, d=d)

    # run the optimization
    model.setParam('OutputFlag', 0)
    model.optimize()

    # return result
    sol = _reorganize_solution(model, A, C)
    sol['active_set'] = _get_active_set_lp(model, A)

    return sol

def quadratic_program(H, f, A, b, C=None, d=None):
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
    
    # get model
    model = _build_model(H=H, f=f, A=A, b=b, C=C, d=d)

    # run the optimization
    model.setParam('OutputFlag', False)
    model.optimize()

    # return result
    sol = _reorganize_solution(model, A, C)
    sol['active_set'] = _get_active_set_qp(model, sol['multiplier_inequality'])

    return sol

def _build_model(H=None, f=None, A=None, b=None, C=None, d=None):

    # initialize model
    model = grb.Model()
    n_x = max(f.shape)
    x = model.addVars(n_x, lb=[- grb.GRB.INFINITY]*n_x)

    # linear inequalities
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
    f = f.reshape(1, max(f.shape))
    expr = linear_expression(f, np.zeros((1,1)), x)
    cost.add(expr[0])
    model.setObjective(cost)

    return model

def _reorganize_solution(model, A, C):

    # intialize solution
    sol = {
        'min': None,
        'argmin': None,
        'active_set': None,
        'multiplier_inequality': None,
        'multiplier_equality': None
    }

    # primal solution
    if model.status == grb.GRB.Status.OPTIMAL:
        x = model.getVars()
        sol['min'] = model.objVal
        sol['argmin'] = np.array(model.getAttr('x')).reshape(len(x), 1)

        # dual inequalities
        ineq_mult = []
        for i in range(A.shape[0]):
            constr = model.getConstrByName('ineq_'+str(i))
            ineq_mult.append(-constr.getAttr('Pi'))
        sol['multiplier_inequality'] = np.vstack(ineq_mult)

        # dual equalities
        if C is not None and C.shape[0] > 0:
            eq_mult = []
            for i in range(C.shape[0]):
                constr = model.getConstrByName('eq_'+str(i))
                eq_mult.append(-constr.getAttr('Pi'))
            sol['multiplier_equality'] = np.vstack(eq_mult)

    return sol

def _get_active_set_lp(model, A):
    active_set = None
    if model.status == grb.GRB.Status.OPTIMAL:
        active_set = []
        for i in range(A.shape[0]):
            constr = model.getConstrByName('ineq_'+str(i))
            if constr.getAttr('CBasis') == -1:
                active_set.append(i)
    return active_set

def _get_active_set_qp(model, ineq_mult, tol=1.e-6):
    active_set = None
    if ineq_mult is not None:
        active_set = []
        for i, mult in enumerate(ineq_mult.flatten().tolist()):
            if mult > tol:
                active_set.append(i)
    return active_set

##########
  
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