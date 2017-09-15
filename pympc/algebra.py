import numpy as np
from copy import copy

def nullspace_basis(A):
    """
    Uses SVD to find a basis of the nullsapce of A.
    """
    V = np.linalg.svd(A)[2].T
    rank = np.linalg.matrix_rank(A)
    Z = V[:,rank:]
    return clean_matrix(Z)

def rangespace_basis(A):
    """
    Uses SVD to find a basis of the rangesapce of A.
    """
    Z = nullspace_basis(A.T)
    return nullspace_basis(Z.T)

def linearly_independent_rows(A, tol=1.e-6):
    """
    Returns the indices of a set of linear independent rows of the matrix A.
    """
    R = linalg.qr(A.T)[1]
    R_diag = np.abs(np.diag(R))
    return list(np.where(R_diag > tol)[0])

def clean_matrix(M, tol=1.e-9):
    M_clean = copy(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.abs(M[i,j]) < tol:
                M_clean[i,j] = 0.
    return M_clean