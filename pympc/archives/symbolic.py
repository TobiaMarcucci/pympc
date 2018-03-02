import numpy as np
import sympy as sp

def SymbolicVector(length, name):
    """
    Creates a symbolic column vector.

    Parameters
    ----------
    length : int
        Number of rows of the column vector.
    name : str
        Name of the symbolic vector.

    Returns
    -------
    x : numpy.ndarray
        Column vector (2d matrix) of sympy symbols.
    """
    
    # cosntruct symbolic vector
    x = np.array([[sp.symbols(name + "_" + str(i))] for i in range(length)])
    
    return x

def jacobian(expr, x):
    """
    Differentiates expr with respect to x.

    Parameters
    ----------
    expr : numpy.ndarray
        Column vector (2d matrix) of differentiable sympy expressions.
    x : numpy.ndarray
        Column vector (2d matrix) of sympy symbols.

    Returns
    -------
    J : numpy.ndarray
        Jacobian matrix (2d matrix of sympy expressions).
    """
    
    # check sizes of the input vectors
    a, b = expr.shape
    c, d = x.shape
    if b != 1:
        raise ValueError("can differentiate only column vectors")
    if d != 1:
        raise ValueError("can differentiate only w.r.t. column vectors")
        
    # construct Jacobian
    J = []
    for i in range(a):
        row = []
        for j in range(c):
            row.append(expr[i,0].diff(x[j,0]))
        J.append(row)

    return np.array(J)

def hessian(expr, x):
    """
    Differentiates twice expr with respect to x.

    Parameters
    ----------
    expr : numpy.ndarray
        1-by-1 2d matrix of differentiable sympy expressions.
    x : numpy.ndarray
        Column vector (2d matrix) of sympy symbols.

    Returns
    -------
    H : numpy.ndarray
        Hessian matrix (2d matrix of sympy expressions).
    """
    
    # check sizes of the input vectors
    a, b = expr.shape
    c, d = x.shape
    if a != 1 or b != 1:
        raise ValueError("can differentiate only 1-by-1 arrays")
    if d != 1:
        raise ValueError("can differentiate only w.r.t. column vectors")
        
    # construct Hessian
    H = jacobian(jacobian(expr, x).T, x)

    return H