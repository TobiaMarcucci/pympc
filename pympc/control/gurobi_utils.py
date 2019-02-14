# external imports
import numpy as np
import gurobipy as grb

def add_variables(model, n, lb=None, **kwargs):
    """
    Adds n variables to the Gurobi model.
    By default Gurobi sets the lower bouns to zero, here the default is -infinity.

    Arguments
    ----------
    model : instance of gurobi.Model
        Mathematical program.
    n : int
        Number of optimization variables to be added to the program.
    lb : list of floats
        Lower bound on the optimization variables.

    Returns
    ----------
    x_np : numpy.ndarray
        Optimization variables added to the program, organized in a numpy array.
    """

    # reset the lower bound from 0 to -infinity
    if lb is None:
        lb = [-grb.GRB.INFINITY] * n

    # create optimization variables and organize numpy array
    x = model.addVars(n, lb=lb, **kwargs)
    x_np = np.array([xi for xi in x.values()])

    # update model to be sure that new variables are immediately visible
    model.update()

    return x_np

def add_linear_inequality(model, x, y):
    """
    Adds the linear inequality x <= y to the Gurobi model.

    Arguments
    ----------
    x : numpy array of gurobi.LinExpr
        Left hand side of the inequality.
    y : numpy array of gurobi.LinExpr
        Right hand side of the inequality.

    Returns
    ----------
    c : list of gurobi.Constr
        New constraints added to the optimization problem.
    """

    # check input size
    if x.size != y.size:
        raise ValueError('right and left hand side of the inequality must have the same dimension.')
    
    # add constraints to the model
    # (sometimes x is a vector of floats: gurobi raises errors if variables are not in the lhs)
    c = [model.addConstr(x[k] - y[k] <= 0.) for k in range(x.size)] 

    # update model to be sure that new variables are immediately visible
    model.update()

    return c

def add_linear_equality(model, x, y):
    """
    Adds the linear equality x = y to the Gurobi model.

    Arguments
    ----------
    x : numpy array of gurobi.LinExpr
        Left hand side of the equality.
    y : numpy array of gurobi.LinExpr
        Right hand side of the equality.

    Returns
    ----------
    c : list of gurobi.Constr
        New constraints added to the optimization problem.
    """

    # check input size
    if x.size != y.size:
        raise ValueError('right and left hand side of the equality must have the same dimension.')

    # add constraints to the model
    # (sometimes x is a vector of floats: gurobi raises errors if variables are not in the lhs)
    c = [model.addConstr(x[k] - y[k] == 0.) for k in range(x.size)] 

    # update model to be sure that new variables are immediately visible
    model.update()

    return c