# external imports
import numpy as np
from scipy.spatial import ConvexHull

# pympc imports
from pympc.optimization.pnnls import linear_program
from pympc.geometry.utils import plane_through_points

def convex_hull_method(A, b, resiudal_dimensions):
    """
    Given a bouned polyhedron in the form P := {x | A x <= b}, returns the orthogonal projection to the given dimensions.
    Dividing the space in the residual dimensions y and the dropped dimensions z, we have proj_y(P) := {y | exists z s.t. A_y y + A_z z < b}.
    The projection is returned in both the halfspace representation {x | E x <= f} and the vertices representation {x in conv(vertices)}.
    This is an implementation of the Convex Hull Method for orthogonal projections of polytopes, see, e.g., http://www.ece.drexel.edu/walsh/JayantCHM.pdf.
    The polyhderon is assumed to be bounded and full dimensional.

    Arguments
    ----------
    A : numpy.ndarray
        Left-hand side of the inequalities describing the higher dimensional polytope.
    b : numpy.ndarray
        Right-hand side of the inequalities describing the higher dimensional polytope.
    residual_dimensions : list of int
        Indices of the dimensions onto which the polytope has to be projected.
    
    Returns
    ----------
    E : numpy.ndarray
        Left-hand side of the inequalities describing the projection.
    f : numpy.ndarray
        Right-hand side of the inequalities describing the projection.
    vertices : list of numpy.ndarray
        List of the vertices of the projection.
    """

    # reorder coordinates
    n = len(resiudal_dimensions)
    dropped_dimensions = [i for i in range(A.shape[1]) if i not in resiudal_dimensions]
    A = np.hstack((
        A[:, resiudal_dimensions],
        A[:, dropped_dimensions]
        ))

    # initialize projection
    vertices = _get_two_vertices(A, b, n)
    if n == 1:
        E = np.array([[1.],[-1.]])
        f = np.array([
            [max(v[0,0] for v in vertices)],
            [- min(v[0,0] for v in vertices)]
            ])
        return E, f, vertices
    vertices = _get_inner_simplex(A, b, vertices)

    # expand facets
    hull = ConvexHull(
        np.hstack(vertices).T,
        incremental=True
        )
    hull = _expand_simplex(A, b, hull)
    hull.close()

    # get outputs
    E = hull.equations[:, :-1]
    f = - hull.equations[:, -1:]
    vertices = [np.reshape(v, (v.shape[0], 1)) for v in hull.points]

    return E, f, vertices

def _get_two_vertices(A, b, n):
    """
    Findes two vertices of the projection.

    Arguments
    ----------
    A : numpy.ndarray
        Left-hand side of the inequalities describing the higher dimensional polytope.
    b : numpy.ndarray
        Right-hand side of the inequalities describing the higher dimensional polytope.
    n : int
        Dimensionality of the space onto which the polytope has to be projected.
    
    Returns
    ----------
    vertices : list of numpy.ndarray
        List of two vertices of the projection.
    """

    # select any direction to explore (it has to belong to the projected space, i.e. a_i = 0 for all i > n)
    a = np.vstack((
        np.ones((1,1)),
        np.zeros((A.shape[1]-1, 1))
        ))

    # minimize and maximize in the given direction
    vertices = []
    for f in [a, -a]:
        sol = linear_program(f, A, b)
        vertices.append(sol['argmin'][:n,:])

    return vertices

def _get_inner_simplex(A, b, vertices, tol=1.e-7):
    """
    Constructs a simplex contained in the porjection.

    Arguments
    ----------
    A : numpy.ndarray
        Left-hand side of the inequalities describing the higher dimensional polytope.
    b : numpy.ndarray
        Right-hand side of the inequalities describing the higher dimensional polytope.
    vertices : list of numpy.ndarray
        List of two vertices of the projection.
    tol : float
        Maximal expansion of a facet to consider it a facet of the projection.

    Returns
    ----------
    vertices : list of numpy.ndarray
        List of vertices of the simplex contained in the projection.
    """

    # initialize LPs
    n = vertices[0].shape[0]
    
    # expand increasing at every iteration the dimension of the space
    for i in range(2, n+1):
        a, d = plane_through_points([v[:i,:] for v in vertices])
        f = np.vstack((a, np.zeros((A.shape[1]-i, 1))))
        sol = linear_program(f, A, b)

        # check the length of the expansion wrt to the plane, if zero expand in the opposite direction
        expansion = np.abs(a.T.dot(sol['argmin'][:i, :]) - d) # >= 0
        if expansion < tol:
            f = - f
            sol = linear_program(f, A, b)
        vertices.append(sol['argmin'][:n,:])

    return vertices

def _expand_simplex(A, b, hull, tol=1.e-7):
    """
    Expands the internal simplex to cover all the projection.

    Arguments
    ----------
    A : numpy.ndarray
        Left-hand side of the inequalities describing the higher dimensional polytope.
    b : numpy.ndarray
        Right-hand side of the inequalities describing the higher dimensional polytope.
    hull : instance of ConvexHull
        Convex hull of vertices of the input simplex.
    tol : float
        Maximal expansion of a facet to consider it a facet of the projection.

    Returns
    ----------
    hull : instance of ConvexHull
        Convex hull of vertices of the projection.
    """

    # initialize algorithm's variables
    n = hull.points[0].shape[0]
    a_explored = []

    # start convex-hull method
    convergence = False
    while not convergence:
        convergence = True

        # check if every facet of the inner approximation belongs to the projection
        for i in range(hull.equations.shape[0]):

            # get normalized halfplane {x | a' x <= d} of the ith facet
            a = hull.equations[i:i+1, :-1].T
            d = - hull.equations[i, -1]
            a_norm = np.linalg.norm(a)
            a /= a_norm
            b /= a_norm

            # check it the direction a has been explored so far
            is_explored = any((np.allclose(a, a2) for a2 in a_explored))
            if not is_explored:
                a_explored.append(a)

                # maximize in the direction a
                f = np.vstack((
                    - a,
                    np.zeros((A.shape[1]-n, 1))
                    ))
                sol = linear_program(f, A, b)

                # check if expansion wrt to the halfplane is greater than zero
                expansion = - sol['min'] - d # >= 0
                if expansion > tol:
                    convergence = False
                    hull.add_points(sol['argmin'][:n,:].T)
                    break

    return hull