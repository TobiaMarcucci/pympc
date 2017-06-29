import numpy as np

def nullspace_basis(A):
    """
    Uses singular value decomposition to find a basis of the nullsapce of A.
    """
    V = np.linalg.svd(A)[2].T
    rank = np.linalg.matrix_rank(A)
    Z = V[:,rank:]
    return Z