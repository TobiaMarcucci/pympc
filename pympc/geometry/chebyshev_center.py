import numpy as np
from pympc.algebra import nullspace_basis
from pympc.optimization.gurobi import linear_program

def chebyshev_center(A, b, C=None, d=None, tol=1.e-10):
    """
    Finds the Chebyshev center of the polytope P := {x | A x <= b, C x = d} solving the LP: min_{z, e}  e s.t. F z <= g + F_{row_norm} e.
    If no equalities are provided, we have F = A, z = x, g = b.
    If equalities are present F = A Z, g = b - A Y y, with: Z basis of the nullspace of C, Y orthogonal complement to Z, y = (C Y)^-1 d and x is retrived as x = Z z + Y y.
    Here F_{row_norm} dentes the vector whose ith entry is the 2-norm of the ith row of F.

    Arguments
    ----------
    A : numpy.ndarray
        Left-hand side of the inequalities.
    b : numpy.ndarray
        Right-hand side of the inequalities.
    C : numpy.ndarray
        Left-hand side of the equalities.
    d : numpy.ndarray
        Right-hand side of the equalities.

    Returns
    ----------
    center : numpy.ndarray
        Chebyshev center of the polytope (nan if the P is empty, inf if P is unbounded and the center is at infinity).
    radius : float
        Chebyshev radius of the polytope (nan if the P is empty, inf if it is infinite).
    """
    A_projected = A
    b_projected = b
    if C is not None and d is not None:
        # project in the null space of the equalities
        Z = nullspace_basis(C)
        Y = nullspace_basis(Z.T)
        A_projected = A.dot(Z)
        CY_inv = np.linalg.inv(C.dot(Y))
        y = CY_inv.dot(d)
        y = np.reshape(y, (y.shape[0],1))
        b_projected = b - A.dot(Y.dot(y))
    [n_facets, n_variables] = A_projected.shape
    # check if the problem is trivially unbounded
    A_row_norm = np.linalg.norm(A,axis=1)
    A_zero_rows = np.where(A_row_norm < tol)[0]
    if any(b[A_zero_rows] < 0.):
        radius = np.nan
        center = np.zeros((n_variables,1))
        center[:] = np.nan
        return [center, radius]
    f_lp = np.zeros((n_variables+1, 1))
    f_lp[-1] = 1.
    A_row_norm = np.reshape(np.linalg.norm(A_projected, axis=1), (n_facets, 1))
    A_lp = np.hstack((A_projected, -A_row_norm))
    sol = linear_program(f_lp, A_lp, b_projected)
    center = sol.argmin
    radius = sol.min
    center = center[0:-1]
    radius = -radius
    if C is not None and d is not None:
        # go back to the original coordinates
        center = np.hstack((Z,Y)).dot(np.vstack((center,y)))
    if np.isnan(radius):
        radius = np.inf
    if any(np.isnan(center)):
        center[:] = np.inf
    if radius < tol:
        radius = np.nan
        center[:] = np.nan
    return [center, radius]