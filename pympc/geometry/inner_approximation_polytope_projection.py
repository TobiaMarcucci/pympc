import numpy as np
from pympc.optimization.gurobi import linear_program, quadratically_constrained_linear_program
from scipy.spatial import ConvexHull
import scipy.linalg as linalg
import h5py
# from pympc.geometry.polytope import Polytope

class InnerApproximationOfPolytopeProjection:

    def __init__(self, A_hd_unordered, b_hd, residual_dimensions, vertices_ld=None, A_ld=None, b_ld=None):
        """
        Generates and expands an inner approximation for the orthogonal projection of the polytope
        P := {x | A x <= b}
        to the space of the x_i for i in residual_dimensions.
        """

        # higher dimensional polytope
        self.A_hd_unordered = A_hd_unordered
        self.A_hd = self._reorder_coordinates(A_hd_unordered, residual_dimensions)
        self.b_hd = b_hd

        # lower dimensional polytope
        self.residual_dimensions = residual_dimensions
        if vertices_ld is None:
            self.vertices_ld = []
        else:
            self.vertices_ld = vertices_ld
        self.A_ld = A_ld
        self.b_ld = b_ld

        # dimensions
        self.n_hd = A_hd_unordered.shape[1]
        self.n_ld = len(residual_dimensions)

        return

    @staticmethod
    def _reorder_coordinates(A_hd, residual_dimensions):
        dropped_dimensions = [i for i in range(A_hd.shape[1]) if i not in residual_dimensions]
        A_hd_ordered = np.hstack((A_hd[:, residual_dimensions], A_hd[:, dropped_dimensions]))
        return A_hd_ordered

    def include_point(self, point, tol=1e-7):

        # if there are no points, initialize the hull
        if not self.vertices_ld:
            self._first_two_points()
            self._inner_simplex(point)

        # loop until the point is included
        while not point_in_convex_hull(point, self.vertices_ld):

            # find best direction of growth
            n = separating_hyperplane_of_maximum_alignment(point, self.vertices_ld)[0]
            a = np.zeros((self.n_hd, 1))
            a[:self.n_ld,:] = n

            # get the new vetex
            sol = linear_program(-a, self.A_hd, self.b_hd)
            new_vertex = sol.argmin[:self.n_ld,:]
            if min([np.linalg.norm(new_vertex - v) for v in self.vertices_ld]) < tol:
                raise ValueError('The point' + str(new_vertex.flatten()) + ' lies outside the projection.')
            self.vertices_ld.append(new_vertex)

        return

    def _first_two_points(self):
        a = np.zeros((self.n_hd, 1))
        a[0,0] = 1.
        for a in [a, -a]:
            sol = linear_program(a, self.A_hd, self.b_hd)
            self.vertices_ld.append(sol.argmin[:self.n_ld,:])
        return

    def _inner_simplex(self, point, tol=1.e-7):
        for i in range(2, self.n_ld+1):
            a, d = plane_through_points([v[:i,:] for v in self.vertices_ld])
            sign = 1.
            if point is not None: # picks the right sign for a
                sign = np.sign((a.T.dot(point[:i,:]) - d)[0,0])
            a = np.vstack((
                a,
                np.zeros((self.n_hd - i, 1))
                ))
            sol = linear_program(-sign*a, self.A_hd, self.b_hd)
            if -sol.min - sign*d[0,0] < tol:
                if point is not None:
                    raise ValueError('The given point lies outside the projection.')
                a = -a
                sol = linear_program(-a, self.A_hd, self.b_hd)
            self.vertices_ld.append(sol.argmin[:self.n_ld,:])
        return

    def applies_to(self, point, tol=1.e-9):
        """
        Determines if the given point belongs to the polytope (returns True or False).
        """
        return point_in_convex_hull(point, self.vertices_ld)

    def store_halfspaces(self):
        points = np.hstack(self.vertices_ld).T
        hull = ConvexHull(points)
        self.A_ld = hull.equations[:,:-1]
        self.b_ld = - hull.equations[:,-1:]
        return

    def save(self, group_name, super_group=None):

        # open the file
        if super_group is None:
            group = h5py.File(group_name + '.hdf5', 'w')
        else:
            group = super_group.create_group(group_name)

        # write higher dimensional polytope matrices
        A_hd_unordered = group.create_dataset('A_hd_unordered', self.A_hd_unordered.shape)
        b_hd = group.create_dataset('b_hd', self.b_hd.shape)
        A_hd_unordered[...] = self.A_hd_unordered
        b_hd[...] = self.b_hd

        # write residual dimensions
        residual_dimensions = group.create_dataset('residual_dimensions', (len(self.residual_dimensions),), np.int8)
        residual_dimensions[...] = np.array(self.residual_dimensions)

        # write lower dimensional polytope vertices
        vertices_ld = group.create_group('vertices_ld')
        for i, vertex in enumerate(self.vertices_ld):
            dset = vertices_ld.create_dataset(str(i), vertex.shape)
            dset[...] = vertex

        # write lower dimensional polytope halfspaces
        if self.A_ld is not None:
            A_ld = group.create_dataset('A_ld', self.A_ld.shape)
            A_ld[...] = self.A_ld
        else:
            A_ld = group.create_dataset('A_ld', data=h5py.Empty('f'))
        if self.b_ld is not None:
            b_ld = group.create_dataset('b_ld', self.b_ld.shape)
            b_ld[...] = self.b_ld
        else:
            b_ld = group.create_dataset('b_ld', data=h5py.Empty('f'))

        # close the file and return
        if super_group is None:
            group.close()
            return
        else:
            return super_group

def upload_InnerApproximationOfPolytopeProjection(group_name, super_group=None):
    """
    Reads the file group_name.hdf5 and generates a InnerApproximationOfPolytopeProjection from the data therein.
    If a super_group is provided, reads the sub group named group_name which belongs to the super_group.
    """

    # open the file
    if super_group is None:
        polytope_approximation = h5py.File(group_name + '.hdf5', 'r')
    else:
        polytope_approximation = super_group[group_name]

    # read higher dimensional polytope matrices
    A_hd_unordered = np.array(polytope_approximation['A_hd_unordered'])
    b_hd = np.array(polytope_approximation['b_hd'])

    # read residual dimensions
    residual_dimensions = list(polytope_approximation['residual_dimensions'])

    # read lower dimensional polytope vertices
    vertices_ld = []
    for i in range(len(polytope_approximation['vertices_ld'])):
        vertices_ld.append(np.array(polytope_approximation['vertices_ld'][str(i)]))

    # read lower dimensional polytope halfspaces
    if polytope_approximation['A_ld'].shape is None:
        A_ld = None
    else:
        A_ld = np.array(polytope_approximation['A_ld'])
    if polytope_approximation['b_ld'].shape is None:
        b_ld = None
    else:
        b_ld = np.array(polytope_approximation['b_ld'])

    # close the file and return
    if super_group is None:
        controller.close()
    return InnerApproximationOfPolytopeProjection(A_hd_unordered, b_hd, residual_dimensions, vertices_ld, A_ld, b_ld)

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
    return sol.min < 0.


def separating_hyperplane_of_maximum_alignment(point, hull_points):
    """
    Given a set of points {v_1, ..., v_n} and a point p which lies outside their convex hull. It returns the separating hyperplane H := {x | a^T x = b} with the normal a as aligned as possible with the segment connecting p and the centroid c := 1/n * \sum_{i = 1}^n v_i.
    It solves the quadratically-constrained linear program
    max_{a, b} (p - c)^T a
    subject to
    a^T p >= b
    a^T v_i <= b \forall i \in {1, ..., n}
    ||a||_2 <= 1
    (The constraints on the norm of b makes the problem bounded.)
    """

    # centroid of the set of points
    n = point.shape[0]
    n_h = len(hull_points)
    HP = np.hstack(hull_points)
    c = (np.sum(HP, axis=1)/n_h).reshape(n,1)

    # given point on one side of the plane
    A = np.hstack((-point.T, np.ones((1,1))))
    b = np.zeros((1, 1))

    # hull points on the other side
    A = np.vstack((A, np.hstack((HP.T, -np.ones((n_h,1))))))
    b = np.vstack((b, np.zeros((n_h, 1))))

    # ||a||_2 <= 1
    P = linalg.block_diag(np.eye(n), np.zeros((1,1)))
    r = 1.

    # solve the qclp
    f = np.vstack((c - point, np.zeros((1,1))))
    ab_star, _ = quadratically_constrained_linear_program(f=f, A=A, b=b, P=P, r=r)
    a_star = ab_star[:n,:]
    a_star /= np.linalg.norm(a_star) 
    b_star = ab_star[n:,:]/np.linalg.norm(a_star) 

    return a_star, b_star, c