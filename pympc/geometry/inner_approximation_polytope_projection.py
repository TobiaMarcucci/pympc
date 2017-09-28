import numpy as np
from pympc.optimization.gurobi import linear_program
from scipy.spatial import ConvexHull

class InnerApproximationOfPolytopeProjection:

    def __init__(self, A, b, residual_dimensions):
        """
        It generates and expands an inner approximation for the orthogonal projection of the polytope
        P := {x | A x <= b}
        to the space of the x_i for i in residual_dimensions.
        """
        self.empty = False
        self.A = A
        self.b = b
        self.residual_dimensions = residual_dimensions
        self.hull = None
        # move the variables to drop at the end
        dropped_dimensions = [i for i in range(A.shape[1]) if i not in residual_dimensions]
        self.A_ordered = np.hstack((A[:, residual_dimensions], A[:, dropped_dimensions]))
        return

    def _initialize(self, point=None):
        simplex_vertices = self._first_two_points()
        simplex_vertices = self._inner_simplex(simplex_vertices, point)
        self.hull = ConvexHull(np.hstack(simplex_vertices).T, incremental=True)
        return

    def _first_two_points(self):
        simplex_vertices = []
        a = np.zeros((self.A.shape[1], 1))
        a[0,0] = 1.
        for a in [a, -a]:
            sol = linear_program(a, self.A_ordered, self.b)
            simplex_vertices.append(sol.argmin[:len(self.residual_dimensions),:])
        return simplex_vertices

    def _inner_simplex(self, simplex_vertices, point=None, tol=1.e-7):
        for i in range(2, len(self.residual_dimensions)+1):
            a, d = plane_through_points([v[:i,:] for v in simplex_vertices])
            sign = 1.
            if point is not None: # picks the right sign for a
                sign = np.sign((a.T.dot(point[:i,:]) - d)[0,0])
            a = np.vstack((a, np.zeros((self.A.shape[1]-i, 1))))
            sol = linear_program(-sign*a, self.A_ordered, self.b)
            if -sol.min - sign*d[0,0] < tol:
                if point is not None:
                    raise ValueError('The given point lies outside the projection.')
                a = -a
                sol = linear_program(-a, self.A_ordered, self.b)
            simplex_vertices.append(sol.argmin[:len(self.residual_dimensions),:])
        return simplex_vertices

    def include_point(self, point, tol=1e-7):
        if self.hull is None:
            self._initialize(point)
        residuals = self._hull_residuals(point)
        while max(residuals) > tol:
            index_facet_to_expand = residuals.index(max(residuals))
            facet_to_expand = self._hull_facet(index_facet_to_expand)
            a = np.zeros((self.A.shape[1], 1))
            a[:len(self.residual_dimensions),:] = facet_to_expand[0].T
            sol = linear_program(-a, self.A_ordered, self.b)
            if - sol.min - facet_to_expand[1] < tol:
                raise ValueError('The given point lies outside the projection.')
            new_vertex = sol.argmin[:len(self.residual_dimensions),:].T
            try:
                self.hull.add_points(new_vertex)
            except:
                print 'exception thrown from qhull'
                points = np.vstack((self.hull.points, new_vertex))
                self.hull = ConvexHull(points, incremental=True)
            residuals = self._hull_residuals(point)
        return

    def _hull_residuals(self, point):
        lhs = self.hull.equations[:,:-1]
        rhs = self.hull.equations[:,-1:]
        residuals = (lhs.dot(point) + rhs).flatten().tolist()
        return residuals

    def _hull_facet(self, index):
        lhs = self.hull.equations[index:index+1,:-1]
        rhs = - self.hull.equations[index,-1]
        return [lhs, rhs]

    def applies_to(self, point, tol=1.e-9):
        """
        Determines if the given point belongs to the polytope (returns True or False).
        """
        if self.hull is None:
            return False
        residuals = self._hull_residuals(point)
        is_inside = max(residuals) <= tol
        return is_inside

def plane_through_points(points):
    """
    Returns the plane a^T x = b passing through the input points. It first adds a random offset to be sure that the matrix of the points is invertible (it wouldn't be the case if the plane we are looking for passes through the origin). The a vector has norm equal to one; the scalar b is non-negative.
    """
    offset = np.random.rand(points[0].shape[0], 1)
    points = [p + offset for p in points]
    P = np.hstack(points).T
    a = np.linalg.solve(P, np.ones(offset.shape))
    b = 1. - a.T.dot(offset)
    a_norm = np.linalg.norm(a)
    a /= a_norm
    b /= a_norm
    return a, b