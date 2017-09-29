import numpy as np
from pympc.optimization.gurobi import linear_program
from scipy.spatial import ConvexHull

# class InnerApproximationOfPolytopeProjection:

#     def __init__(self, A, b, residual_dimensions):
#         """
#         It generates and expands an inner approximation for the orthogonal projection of the polytope
#         P := {x | A x <= b}
#         to the space of the x_i for i in residual_dimensions.
#         """
#         self.empty = False
#         self.A = A
#         self.b = b
#         self.residual_dimensions = residual_dimensions
#         self.hull = None
#         # move the variables to drop at the end
#         dropped_dimensions = [i for i in range(A.shape[1]) if i not in residual_dimensions]
#         self.A_ordered = np.hstack((A[:, residual_dimensions], A[:, dropped_dimensions]))
#         return

#     def _initialize(self, point=None):
#         simplex_vertices = self._first_two_points()
#         simplex_vertices = self._inner_simplex(simplex_vertices, point)
#         self.hull = ConvexHull(np.hstack(simplex_vertices).T, incremental=True)
#         return

#     def _first_two_points(self):
#         simplex_vertices = []
#         a = np.zeros((self.A.shape[1], 1))
#         a[0,0] = 1.
#         for a in [a, -a]:
#             sol = linear_program(a, self.A_ordered, self.b)
#             simplex_vertices.append(sol.argmin[:len(self.residual_dimensions),:])
#         return simplex_vertices

#     def _inner_simplex(self, simplex_vertices, point=None, tol=1.e-7):
#         for i in range(2, len(self.residual_dimensions)+1):
#             a, d = plane_through_points([v[:i,:] for v in simplex_vertices])
#             sign = 1.
#             if point is not None: # picks the right sign for a
#                 sign = np.sign((a.T.dot(point[:i,:]) - d)[0,0])
#             a = np.vstack((a, np.zeros((self.A.shape[1]-i, 1))))
#             sol = linear_program(-sign*a, self.A_ordered, self.b)
#             if -sol.min - sign*d[0,0] < tol:
#                 if point is not None:
#                     raise ValueError('The given point lies outside the projection.')
#                 a = -a
#                 sol = linear_program(-a, self.A_ordered, self.b)
#             simplex_vertices.append(sol.argmin[:len(self.residual_dimensions),:])
#         return simplex_vertices

#     def include_point(self, point, tol=1e-7):
#         if self.hull is None:
#             self._initialize(point)
#         residuals = self._hull_residuals(point)
#         while max(residuals) > tol:
#             index_facet_to_expand = residuals.index(max(residuals))
#             facet_to_expand = self._hull_facet(index_facet_to_expand)
#             a = np.zeros((self.A.shape[1], 1))
#             a[:len(self.residual_dimensions),:] = facet_to_expand[0].T
#             sol = linear_program(-a, self.A_ordered, self.b)
#             if - sol.min - facet_to_expand[1] < tol:
#                 raise ValueError('The given point lies outside the projection.')
#             new_vertex = sol.argmin[:len(self.residual_dimensions),:].T
#             # self.hull.add_points(new_vertex)
#             self.hull = ConvexHull(np.vstack((self.hull.points, new_vertex)))
#             residuals = self._hull_residuals(point)
#         return

#     def _hull_residuals(self, point):
#         lhs = self.hull.equations[:,:-1]
#         rhs = self.hull.equations[:,-1:]
#         residuals = (lhs.dot(point) + rhs).flatten().tolist()
#         return residuals

#     def _hull_facet(self, index):
#         lhs = self.hull.equations[index:index+1,:-1]
#         rhs = - self.hull.equations[index,-1]
#         return [lhs, rhs]

#     def applies_to(self, point, tol=1.e-9):
#         """
#         Determines if the given point belongs to the polytope (returns True or False).
#         """
#         if self.hull is None:
#             return False
#         residuals = self._hull_residuals(point)
#         is_inside = max(residuals) <= tol
#         return is_inside


class InnerApproximationOfPolytopeProjection:

    def __init__(self, A, b, residual_dimensions):
        """
        It generates and expands an inner approximation for the orthogonal projection of the polytope
        P := {x | A x <= b}
        to the space of the x_i for i in residual_dimensions.
        """
        # store data
        self.A = A
        self.b = b
        self.residual_dimensions = residual_dimensions
        self.vertices = []

        # move the variables to drop at the end of the lhs matrix
        dropped_dimensions = [i for i in range(A.shape[1]) if i not in residual_dimensions]
        self.A_ordered = np.hstack((A[:, residual_dimensions], A[:, dropped_dimensions]))

        return

    def include_point(self, point, tol=1e-7):

        # if there are no points, initialize the hull
        if not self.vertices:
            self._first_two_points()
            self._inner_simplex(point)

        # loop until the point is included
        while not point_in_convex_hull(point, self.vertices):

            # find best direction of growth
            n, _ = separating_hyperplane_of_maximum_alignment(point, self.vertices)
            a = np.zeros((self.A.shape[1], 1))
            a[:len(self.residual_dimensions),:] = n

            # get the new vetex
            sol = linear_program(-a, self.A_ordered, self.b)
            new_vertex = sol.argmin[:len(self.residual_dimensions),:]
            for v in self.vertices:
                if np.allclose(new_vertex, v):
                    raise ValueError('The given point lies outside the projection.')
            self.vertices.append(new_vertex)

        return

    def _first_two_points(self):
        a = np.zeros((self.A.shape[1], 1))
        a[0,0] = 1.
        for a in [a, -a]:
            sol = linear_program(a, self.A_ordered, self.b)
            self.vertices.append(sol.argmin[:len(self.residual_dimensions),:])
        return

    def _inner_simplex(self, point, tol=1.e-7):
        for i in range(2, len(self.residual_dimensions)+1):
            a, d = plane_through_points([v[:i,:] for v in self.vertices])
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
            self.vertices.append(sol.argmin[:len(self.residual_dimensions),:])
        return

    def applies_to(self, point, tol=1.e-9):
        """
        Determines if the given point belongs to the polytope (returns True or False).
        """
        return point_in_convex_hull(point, self.vertices)

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

def point_in_convex_hull(point, hull_points):
    """
    Checks if the point p is inside the convex hull of the points {v_1, ..., v_n}.
    It solves the linear program
    J^* := max_{a_1, ..., a_n} min({a_1, ..., a_n})
         = - min_{a_1, ..., a_n} max({-a_1, ..., -a_n})
    subject to
    \sum_{i = 1}^n a_i = 1
    \sum_{i = 1}^n a_i v_i = p
    If J^* >= 0 then p \in conv({v_1, ..., v_n}).
    """
        
    # sum of the coefficients equal to one
    n_h = len(hull_points)
    lhs_eq = np.ones((1, n_h))
    rhs_eq = np.ones((1, 1))

    # linear combination of the points of the hull equal to the new point
    lhs_eq = np.vstack((lhs_eq, np.hstack(hull_points)))
    rhs_eq = np.vstack((rhs_eq, point))

    # max function (with slack variable)
    lhs_eq = np.hstack((lhs_eq, np.zeros((lhs_eq.shape[0], 1))))
    lhs_ineq = np.hstack((-np.eye(n_h), np.ones((n_h, 1))))
    rhs_ineq = np.zeros((n_h, 1))

    # cost function
    f = np.zeros((n_h+1, 1))
    f[-1, 0] = 1.

    # solve the linear program
    sol = linear_program(-f, lhs_ineq, rhs_ineq, lhs_eq, rhs_eq)
    if sol.min < 0:
        return True
    else:
        return False

def separating_hyperplane_of_maximum_alignment(point, hull_points):
    """
    Given a set of points {v_1, ..., v_n} and a point p which lies outside their convex hull. It returns the separating hyperplane H := {x | a^T x = b} with the normal a as aligned as possible with the segment connecting p and the centroid c := 1/n * \sum_{i = 1}^n v_i.
    It solves the linear program
    max_{a, b} (p - c)^T a
    subject to
    abs(b) <= 1
    a^T p >= b
    a^T v_i <= b \forall i \in {1, ..., n}
    (The constraints on the norm of b makes the problem bounded. It is necessary to move the origin of the coordinate system in c, in order to be sure that b=0 is always unfeasible.)
    """

    # centroid of the set of points
    n = point.shape[0]
    n_h = len(hull_points)
    HP = np.hstack(hull_points)
    c = (np.sum(HP, axis=1)/n_h).reshape(n,1)

    # move the coordinates
    point = point - c
    HP = HP - np.hstack((c for i in range(n_h)))

    # given point on one side of the plane
    lhs = np.hstack((-point.T, np.ones((1,1))))
    rhs = np.zeros((1, 1))

    # hull points on the other side
    lhs = np.vstack((lhs, np.hstack((HP.T, -np.ones((n_h,1))))))
    rhs = np.vstack((rhs, np.zeros((n_h, 1))))

    # norm of b lower or equal to one
    lhs = np.vstack((lhs,
        np.hstack((
            np.zeros((2, n)),
            np.array([[1.], [-1.]])
            ))
        ))
    rhs = np.vstack((rhs, np.ones((2, 1))))

    # solve the linear program
    f = np.vstack((point, np.zeros((1,1))))
    sol = linear_program(-f, lhs, rhs)
    a_star = sol.argmin[:n,:]
    b_star = sol.argmin[n:,:] + a_star.T.dot(c)

    return a_star, b_star