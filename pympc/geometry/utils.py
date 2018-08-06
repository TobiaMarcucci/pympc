# external imports
import numpy as np

def nullspace_basis(A):
    """
    Uses SVD to find a basis of the nullsapce of A.

    Arguments
    ----------
    A : numpy.ndarray
        Matrix for the nullspace.

    Returns
    ----------
    Z : numpy.ndarray
        Nullspace basis.
    """

    # get singular values
    V = np.linalg.svd(A)[2].T

    # cut to the dimension of the rank
    rank = np.linalg.matrix_rank(A)
    Z = V[:,rank:]

    return Z

def linearly_independent_rows(A, tol=1.e-6):
    """
    uses the QR decomposition to find the indices of a set of linear independent rows of the matrix A.

    Arguments
    ----------
    A : numpy.ndarray
        Matrix for the linear independent rows.
    tol : float
        Threshold value for the diagonal elements of R.

    Returns
    ----------
    independent_rows : list of int
        List of indices of a set of independent rows of A.
    """

    # QR decomposition
    R = np.linalg.qr(A.T)[1]

    # check diagonal elements
    R_diag = np.abs(np.diag(R))
    independent_rows = list(np.where(R_diag > tol)[0])

    return sorted(independent_rows)

def plane_through_points(points):
    """
    Returns the plane a' x = b passing through the points.
    It first adds a random offset to be sure that the matrix of the points is invertible (it wouldn't be the case if the plane we are looking for passes through the origin).
    The vector a has norm equal to one and b is non-negative.

    Arguments
    ----------
    points : list of numpy.ndarray
        List of points that the plane has to fit.

    Returns
    ----------
    a : numpy.ndarray
        Left-hand side of the equality describing the plane.
    d : numpy.ndarray
        Right-hand side of the equality describing the plane.
    """

    # generate random offset
    offset = np.random.rand(points[0].size)
    points = [p + offset for p in points]

    # solve linear system
    P = np.vstack(points)
    a = np.linalg.solve(P, np.ones(points[0].size))

    # go back to the original coordinates
    d = 1. - a.dot(offset)

    # scale and select sign of the result
    a_norm = np.linalg.norm(a)
    d_sign = np.sign(d)
    if d_sign == 0.:
        d_sign = 1.
    a /= a_norm * d_sign
    d /= a_norm * d_sign

    return a, d

def same_rows(A, B, normalize=True):
    """
    Checks if two matrices contain the same rows.
    The order of the rows can be different.
    The option normalize, normalizes the rows of A and B; i.e., if True, checks that set of rows of A is the same of the one of B despite a scaling factor.

    Arguments
    ----------
    A : numpy.ndarray
        First matrix to check.
    B : numpy.ndarray
        Second matrix to check.
    normalize : bool
        If True scales the rows of A and B to have norm equal to one.

    Returns:
    equal : bool
        True if the set of rows of A and B are the same.
    """

    # first check the sizes
    if A.shape[0] != B.shape[0]:
        return False

    # if required, normalize
    if normalize:
        for i in range(A.shape[0]):
            A[i] = A[i] / np.linalg.norm(A[i])
            B[i] = B[i] / np.linalg.norm(B[i])

    # check one row per time
    for a in A:
        i = np.where([np.allclose(a, b) for b in B])[0]
        if len(i) != 1:
            return False
        B = np.delete(B, i, 0)

    return True

def same_vectors(v_list, u_list):
    """
    Tests that two lists of array contain the same elements.
    The order of the elements in the lists can be different.

    Arguments
    ----------
    v_list : list of numpy.ndarray
        First ist of arrays to be checked.
    u_list : list of numpy.ndarray
        Second ist of arrays to be checked.

    Returns:
    equal : bool
        True if the set of arrays oin v_list and u_list are the same.
    """

    # check inputs
    for z_list in [v_list, u_list]:
        if any(len(z.shape) >= 2 for z in z_list):
            raise ValueError('input vectors must be 1-dimensional arrays.')

    # construct matrices
    V = np.vstack(v_list)
    U = np.vstack(u_list)

    return same_rows(V, U, False)