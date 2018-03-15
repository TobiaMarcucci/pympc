# external imports
import numpy as np
import gurobipy as grb

def linear_program(f, A, b, C=None, d=None, **kwargs):
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
    for parameter, value in kwargs.items():
        model.setParam(parameter, value)
    model.optimize()

    # return result
    sol = _reorganize_solution(model, A, C)
    sol['active_set'] = _get_active_set_lp(model, A)

    return sol

def quadratic_program(H, f, A, b, C=None, d=None, **kwargs):
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
    model.setParam('OutputFlag', 0)
    for parameter, value in kwargs.items():
        model.setParam(parameter, value)
    model.optimize()

    # return result
    sol = _reorganize_solution(model, A, C)
    sol['active_set'] = _get_active_set_qp(model, sol['multiplier_inequality'])

    return sol

def _build_model(H=None, f=None, A=None, b=None, C=None, d=None):
    """
    Builds the Gurobi model the LP or the QP.

    Arguments
    ----------
    H, f, A, b, C, d : numpy.ndarray
        Matrices of the mathematical program.

    Returns
    ----------
    model : instance of gurobipy.Model
        Model of the mathematical program.
    """

    # initialize model
    model = grb.Model()
    n_x = max(f.shape)
    x = model.addVars(n_x, lb=[- grb.GRB.INFINITY]*n_x)

    # linear inequalities
    for i, expr in enumerate(linear_expression([A], b, [x])):
        model.addConstr(expr <= 0., name='ineq_'+str(i))

    # linear equalities
    if C is not None and d is not None:
        for i, expr in enumerate(linear_expression([C], d, [x])):
            model.addConstr(expr == 0., name='eq_'+str(i))

    # cost function
    if H is not None:
        cost = grb.QuadExpr()
        expr = quadratic_expression(H, x)
        cost.add(.5*expr)
    else:
        cost = grb.LinExpr()
    f = f.reshape(1, max(f.shape))
    expr = linear_expression([f], np.zeros((1,1)), [x])
    cost.add(expr[0])
    model.setObjective(cost)

    return model

def _reorganize_solution(model, A, C):
    """
    Organizes the solution in a dictionary.

    Arguments
    ----------
    model : instance of gurobipy.Model
        Model of the mathematical program.
    A : numpy.ndarray
        Left-hand side of the inequality constraints.
    C : numpy.ndarray
        Left-hand side of the equality constraints.

    Returns
    ----------
    sol : dict
        Dictionary with the solution of the mathematical program.
    """

    # intialize solution
    sol = {
        'min': None,
        'argmin': None,
        'active_set': None,
        'multiplier_inequality': None,
        'multiplier_equality': None
    }

    # if feasible
    if model.status == grb.GRB.Status.OPTIMAL:

        # primal solution
        x = model.getVars()
        sol['min'] = model.objVal
        sol['argmin'] = np.array(model.getAttr('x')).reshape(len(x), 1)

        # dual inequalities
        ineq_mult = []
        for i in range(A.shape[0]):
            constr = model.getConstrByName('ineq_' + str(i))
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
    """
    Retrieves the active set through gurobi (None if unfeasible).

    Arguments
    ----------
    model : instance of gurobipy.Model
        Model of the mathematical program.
    A : numpy.ndarray
        Left-hand side of the inequality constraints.

    Returns
    ----------
    active_set : list of int
        Indices of the active inequallities {i | A_i argmin = b} (None if the problem is unfeasible).
    """

    # if unfeasible return None
    if model.status != grb.GRB.Status.OPTIMAL:
        return None

    # if feasible check CBasis
    active_set = []
    for i in range(A.shape[0]):
        constr = model.getConstrByName('ineq_'+str(i))
        if constr.getAttr('CBasis') == -1:
            active_set.append(i)

    return active_set

def _get_active_set_qp(model, ineq_mult, tol=1.e-6):
    """
    Checks the multipliers t find active inequalities.

    Arguments
    ----------
    model : instance of gurobipy.Model
        Model of the mathematical program.
    ineq_mult : numpy.ndarray
        Lagrange multipliers for the inequality constraints.
    tol : float
        Maximum value of a multiplier to consider the related constraint inactive.

    Returns
    ----------
    active_set : list of int
        Indices of the active inequallities {i | A_i argmin = b} (None if the problem is unfeasible).
    """

    # if unfeasible return None
    if model.status != grb.GRB.Status.OPTIMAL:
        return None

    # otherwise check the magnitude of the multipliers
    if ineq_mult is not None:
        active_set = []
        for i, mult in enumerate(ineq_mult.flatten().tolist()):
            if mult > tol:
                active_set.append(i)

    return active_set

def linear_expression(A_list, b, x_list, tol=1.e-10):
    """
    Generates a list of Gurobi linear expressions
    A[0] x[0] + A[1] x[1] + ... - b
    (one element per row of A).

    Arguments
    ----------
    A_list : list of numpy.ndarray
        List of the Jacobians of the linear expression wrt the variables in x_list.
    b : numpy.ndarray
        Offest term of the linear expression.
    x_list : list of instances of gurobipy.Var
        Variables of the linear expression, one per matrix A[i].
    tol : float
        Maximum absolute value for the elements of A and b to be considered nonzero.

    Returns
    ----------
    exprs : list of gurobipy.LinExpr
        List of linear expressions.
    """

    # initialize expressions
    exprs = []
    for i in range(A_list[0].shape[0]):

        # initialize expression
        expr = grb.LinExpr()

        # loop over the variables
        for k, A in enumerate(A_list):
            for j in range(A.shape[1]):

                # add jacobian
                if np.abs(A[i,j]) > tol:
                    expr.add(A[i,j]*x_list[k][j])

        # add offset term
        if np.abs(b[i]) > tol:
            expr.add(-b[i])
        exprs.append(expr)

    return exprs

def quadratic_expression(H, x, tol=1.e-7):
    """
    Generates a Gurobi quadratic expressions x' H x.

    Arguments
    ----------
    H : numpy.ndarray
        Hessian of the quadratic expression.
    x : instance of gurobipy.Var
        Variable of the linear expression.
    tol : float
        Maximum absolute value for the elements of H to be considered nonzero.

    Returns
    ----------
    expr : gurobipy.LinExpr
        Quadratic expressions.
    """

    # initialize expression
    expr = grb.QuadExpr()
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):

            # add only sufficiently big numbers
            if np.abs(H[i,j]) > tol:
                expr.add(x[i]*H[i,j]*x[j])

    return expr

def mixed_integer_quadratic_program(Huu, Hzz, fz, Au, Az, Ad, b, **kwargs):
    """
    Solves the MIQP coming out the MPC problem for PWA systems.

    Arguments
    ----------
    Huu : numpy.ndarray
        Hessian of the quadratic expression wrt the control inputs.
    Hzz : numpy.ndarray
        Hessian of the quadratic expression wrt the auxiliary continuous variables.
    fz : numpy.ndarray
        Term in objective linear in the auxiliary continuous variables.
    Au : numpy.ndarray
        Left-hand side of the constraints, term in u.
    Az : numpy.ndarray
        Left-hand side of the constraints, term in z.
    Ad : numpy.ndarray
        Left-hand side of the constraints, term in d.
    b : numpy.ndarray
        Right-hand side of the constraints.

    Returns
    ----------
    sol : dict
        Dictionary with the solution of the MIQP.

        Keys
        ----------
        min : float
            Optimal value of the objective function.
        u : numpy.ndarray
            Optimal value for the inputs.
        z : numpy.ndarray
            Optimal value for the auxiliary continuous variables.
        d : numpy.ndarray
            Optimal value for the binary variables.
    """

    # initialize model
    model = grb.Model()
    nu = Au.shape[1]
    nz = Az.shape[1]
    nd = Ad.shape[1]
    u = model.addVars(nu, lb=[- grb.GRB.INFINITY]*nu, name='u')
    z = model.addVars(nz, lb=[- grb.GRB.INFINITY]*nz, name='v')
    d = model.addVars(nd, vtype=grb.GRB.BINARY, name='d')
    model.update()

    # constraints
    for expr in linear_expression([Au, Az, Ad], b, [u, z, d]):
        model.addConstr(expr <= 0.)

    # cost function
    cost = grb.QuadExpr()
    cost.add(.5*quadratic_expression(Huu, u))
    cost.add(.5*quadratic_expression(Hzz, z))
    cost.add(linear_expression([fz.T], np.zeros((1,1)), [z])[0])
    model.setObjective(cost)

    # run the optimization
    model.setParam('OutputFlag', 0)
    for parameter, value in kwargs.items():
        model.setParam(parameter, value)
    model.optimize()

    # intialize solution
    sol = {
        'min': None,
        'u': None,
        'z': None,
        'd': None
    }

    # if feasible
    if model.status == grb.GRB.Status.OPTIMAL:

        # primal solution
        sol['min'] = model.objVal
        sol['u'] = np.array([u[i].x for i in range(nu)]).reshape(nu, 1)
        sol['z'] = np.array([z[i].x for i in range(nz)]).reshape(nz, 1)
        sol['d'] = np.array([d[i].x for i in range(nd)]).reshape(nd, 1)

    return sol