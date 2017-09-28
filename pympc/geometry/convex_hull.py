import time
import numpy as np
import scipy as sp
from itertools import combinations
from pympc.geometry.chebyshev_center import chebyshev_center
from pympc.optimization.gurobi import linear_program
import copy
from scipy.spatial import ConvexHull as ScipyConvexHull

class ConvexHull(object):

    def __init__(self, points):
        self.n_variables = points[0].shape[0]
        vertex_indices = find_vertices_of_a_simplex(points)
        simplex_vertices = [p for i, p in enumerate(points) if i in vertex_indices]
        points_left = [p for i, p in enumerate(points) if i not in vertex_indices]
        self.simplex_halfspaces(simplex_vertices)
        self.points = simplex_vertices
        self.add_points(points_left)
        return

    def simplex_halfspaces(self, simplex_vertices):
        self.halfspaces = []
        for i, v in enumerate(simplex_vertices):
            vertices_on_plane = [u for j, u in enumerate(simplex_vertices) if j!=i]
            a, d = plane_through_points(vertices_on_plane)
            if a.T.dot(v)[0,0] < d[0,0]:
                self.halfspaces.append([a, d, vertices_on_plane])
            else:
                self.halfspaces.append([-a, -d, vertices_on_plane])
        self.get_H_rep()
        return

    def get_H_rep(self):
        # returns a minimal SIMPLICIAL Hrep (i.e. coincident facets are not merged)
        self.A = np.vstack([hs[0].T for hs in self.halfspaces])
        self.b = np.vstack([hs[1] for hs in self.halfspaces])
        return

    def add_point(self, point, tol=1.e-7):
        violations = [(hs[0].T.dot(point) - hs[1])[0,0] > tol for hs in self.halfspaces]
        violated_halfspaces = [self.halfspaces[i] for i, violation in enumerate(violations) if violation]
        self.halfspaces = [self.halfspaces[i] for i, violation in enumerate(violations) if not violation]
        for hs in violated_halfspaces:
            for vertices_on_plane in combinations(hs[2], self.n_variables-1):
                vertices_on_plane = list(vertices_on_plane) + [point]
                a, d = plane_through_points(vertices_on_plane)
                residuals = [(a.T.dot(p) - d)[0,0] for p in self.points] # here only vertices should be multiplied (not all the points)
                if max(residuals) < tol:
                    self.halfspaces.append([a, d, vertices_on_plane])
                elif min(residuals) > -tol:
                    self.halfspaces.append([-a, -d, vertices_on_plane])
        self.points.append(point)
        self.get_H_rep()
        return

    def verified_halfspaces(self, point, tol=1.e-7):
        return [(hs[0].T.dot(point) - hs[1])[0,0] < tol for hs in self.halfspaces]

    def add_points(self, points):
        for p in points:
            self.add_point(p)
        return

    def get_minimal_H_rep(self):
        A = np.zeros((0, self.n_variables))
        b = np.zeros((0, 1))
        for i, hs in enumerate(self.halfspaces):
            A, b = add_facet(hs[0], hs[1], A, b)
        return A, b

    # @staticmethod
    # def check_simplex_vertices(simplex_vertices):
    #     full_rank_matrix = np.hstack([v - simplex_vertices[0] for v in simplex_vertices[1:]])
    #     if full_rank_matrix.shape[0] != full_rank_matrix.shape[1]:
    #         raise ValueError('The number of vertices for the initial simplex has to be equal to ' + str(full_rank_matrix.shape[0]+1) + ', only ' + str(len(simplex_vertices)) + ' provided!')
    #     if np.linalg.matrix_rank(full_rank_matrix) != full_rank_matrix.shape[0]:
    #         raise ValueError('The vertices provided for the construction of the simplex are coplanar!')
    #     return

def add_facet(normal, offset, A, b, tol=1.e-5):
    new_row = np.hstack((normal.T, offset))
    new_row = new_row.flatten()/np.linalg.norm(new_row)
    Ab = np.hstack((A, b))
    duplicate = False
    for row in Ab:
        row = row/np.linalg.norm(row)
        if np.allclose(new_row, row, atol=tol):
            duplicate = True
            break
    if not duplicate:
        A = np.vstack((A, normal.T))
        b = np.vstack((b, offset))
    return A, b

def find_vertices_of_a_simplex(points):
    P = np.hstack([p - points[0] for p in points])
    li_columns = linearly_independent_columns(P)
    return [0] + [i for i in li_columns]

def plane_through_points(points):
    """
    Returns the plane a^T x = b passing through the input points. It first adds a random offset to be shure that the matrix of the points is invertible (it wouldn't be the case if the plane we are looking for passes through the origin). The a vector has norm equal to one; the scalar b is non-negative.
    """
    offset = np.random.rand(points[0].shape[0],1)
    points = [p + offset for p in points]
    P = np.hstack(points).T
    if P.shape[0] != P.shape[1] or np.linalg.matrix_rank(P) != P.shape[0]:
        print 'lhs:', P
        print 'lhs shape:', P.shape
        print 'lhs rank:', np.linalg.matrix_rank(P)
        print 'rhs:', np.ones(offset.shape)
        raise ValueError('aaa')
    a = np.linalg.solve(P, np.ones(offset.shape))
    #print P, np.ones(offset.shape)
    b = 1. - a.T.dot(offset)
    a_norm = np.linalg.norm(a)
    a /= a_norm
    b /= a_norm
    return a, b

def linearly_independent_columns(A):
    li_columns = []
    rank = 0
    A_li = np.zeros((A.shape[0], 0))
    for i in range(A.shape[1]):
        A_li_test = np.hstack((A_li, A[:,i:i+1]))
        if np.linalg.matrix_rank(A_li_test) > rank:
            A_li = A_li_test
            rank += 1
            li_columns.append(i)
    return li_columns



def orthogonal_projection_CHM(A, b, resiudal_dimensions):
    """
    Implementation of the Convex Hull Method for orthogonal projections of polytopes (see, e.g., http://www.ece.drexel.edu/walsh/JayantCHM.pdf).

    Inputs:
        - A, b: H-representation of the polytope P := {x | A x <= b}
        - resiudal_dimensions: list of integers defining the dimensions of the projected polytope.

    Outputs:
        - G, g: non-redundant H-representation of the projection of P
        - v_proj: list of the vertices of the projection of P
    """

    print('\n*** Convex Hull Method for Orthogonal Projections START ***')

    # change of coordinates
    tic = time.time()
    dropped_dimensions = [i for i in range(A.shape[1]) if i not in resiudal_dimensions]
    A = np.hstack((A[:, resiudal_dimensions], A[:, dropped_dimensions]))
    n_proj = len(resiudal_dimensions)

    # projection
    v_proj = first_two_points(A, b, n_proj)
    v_proj = inner_simplex(A, b, v_proj)
    hull = ConvexHull(v_proj)
    print('Inner ' + str(len(hull.points)-1) + 'D simplex found.')
    print('Expansion of the inner polytope started.')
    hull = expand_simplex(A, b, hull)
    G, g = hull.get_minimal_H_rep()

    print('Projection derived ' + str(time.time()-tic) + ' seconds: number of facets is ' + str(hull.A.shape[0]) + ', number of vertices is ' + str(len(hull.points)) + '.')
    print('*** Convex Hull Method for Orthogonal Projections STOP ***\n')
    return G, g, hull.points

def first_two_points(A, b, n_proj):

    v_proj = []
    a = np.zeros((A.shape[1], 1))
    a[0,0] = 1.
    for a in [a, -a]:
        sol = linear_program(a, A, b)
        v_proj.append(sol.argmin[:n_proj,:])

    return v_proj

def inner_simplex(A, b, v_proj, x=None, tol=1.e-7):

    n_proj = v_proj[0].shape[0]
    for i in range(2, n_proj+1):
        a, d = plane_through_points([v[:i,:] for v in v_proj])
        # pick the right sign for a
        sign = 1.
        if x is not None:
            sign = np.sign((a.T.dot(x[:i,:]) - d)[0,0])
        a = np.vstack((a, np.zeros((A.shape[1]-i, 1))))
        sol = linear_program(-sign*a, A, b)
        if -sol.min - sign*d[0,0] < tol:
        #if np.linalg.norm(a.T.dot(sol.argmin) + d) < tol:
        #sol = linear_program(-a, A, b)
        #if -sol.min < d[0,0] + tol:
            if x is not None:
                print 'This is not supposed to happen!'
            a = -a
            sol = linear_program(-a, A, b)
        v_proj.append(sol.argmin[:n_proj,:])

    return v_proj

def expand_simplex(A, b, hull, tol=1.e-7):
    n_proj = hull.points[0].shape[0]
    tested_directions = []
    convergence = False
    residual = np.nan
    while not convergence:
        convergence = True
        for hs in hull.halfspaces:
            if all((hs[0] != a).all() for a in tested_directions):
                tested_directions.append(hs[0])
                a = np.zeros((A.shape[1], 1))
                a[:n_proj,:] = hs[0]
                sol = linear_program(-a, A, b)
                if - sol.min - hs[1][0,0] > tol:
                    residual = '%e' % (- sol.min - hs[1][0,0])
                    convergence = False
                    hull.add_point(sol.argmin[:n_proj,:])
                    break
        print('Facets found so far ' + str(hull.A.shape[0]) + ', vertices found so far ' + str(len(hull.points)) + ', length of the last inflation ' + str(residual) + '.\r'),
    print('\n'),
    return hull


class PolytopeProjectionInnerApproximation:

    def __init__(self, A, b, resiudal_dimensions):

        # data
        self.empty = False
        self.A = A
        self.b = b
        self.resiudal_dimensions = resiudal_dimensions
        self.hull = None

        # put the variables to be eliminated at the end
        dropped_dimensions = [i for i in range(A.shape[1]) if i not in resiudal_dimensions]
        self.A_switched = np.hstack((A[:, resiudal_dimensions], A[:, dropped_dimensions]))

        return

    def _initialize(self, x=None):

        # initialize inner approximation with a simplex
        simplex_vertices = first_two_points(self.A_switched, self.b, len(self.resiudal_dimensions))
        simplex_vertices = inner_simplex(self.A_switched, self.b, simplex_vertices, x)
        # self.hull = ConvexHull(simplex_vertices) # my version
        self.hull = ScipyConvexHull(np.hstack(simplex_vertices).T, incremental=True) # qhull version

        return

    def include_point(self, point, tol=1e-7):

        if self.hull is None:
            self._initialize(point)

        # dimension of the projection space
        n_proj = len(self.resiudal_dimensions)

        # violation of the approximation boundaires
        # residuals = [(hs[0].T.dot(point) - hs[1])[0,0] for hs in self.hull.halfspaces] # my version
        residuals = (self.hull.equations[:,:-1].dot(point) + self.hull.equations[:,-1:]).flatten().tolist() # qhull version

        # # for the plot on the paper
        # import copy
        # from pympc.geometry.polytope import Polytope
        # p_inner_plot = Polytope(self.hull.A, self.hull.b)
        # p_inner_plot.assemble()
        # p_list = [p_inner_plot]

        # expand the most violated boundary until inclusion
        inflations = 0
        time_lp = 0.
        time_ch = 0.
        while max(residuals) > tol:
            inflations += 1
            facet_to_expand = residuals.index(max(residuals))
            a = np.zeros((self.A.shape[1], 1))

            # hs = self.hull.halfspaces[facet_to_expand] # my version
            hs = [self.hull.equations[facet_to_expand:facet_to_expand+1,:-1].T, - self.hull.equations[facet_to_expand:facet_to_expand+1,-1:]] # qhull version

            a[:n_proj,:] = hs[0]
            tic = time.time()
            sol = linear_program(-a, self.A_switched, self.b)
            time_lp += time.time() - tic

            # the point might be outside the projection
            inflation = - sol.min - hs[1][0,0]
            if inflation < tol:
                raise ValueError('The given point lies outside the projection.')
                break

            # add vertex to the hull
            tic = time.time()
            # self.hull.add_point(sol.argmin[:n_proj,:]) # my version
            self.hull.add_points(sol.argmin[:n_proj,:].T) # qhull version
            time_ch += time.time() - tic

            # new residuals
            # residuals = [(hs[0].T.dot(point) - hs[1])[0,0] for hs in self.hull.halfspaces] # my version
            residuals = (self.hull.equations[:,:-1].dot(point) + self.hull.equations[:,-1:]).flatten().tolist() # qhull version

        #     # for the plot on the paper
        #     p_inner_plot = Polytope(self.hull.A, self.hull.b)
        #     p_inner_plot.assemble()
        #     p_list.append(p_inner_plot)
        # return p_list

        print '\nNumber of inflations:', inflations
        print 'Time in linear programming:', time_lp
        print 'Time in convex hull:', time_ch

        return

    def applies_to(self, x, tol=1.e-9):
        """
        Determines if the given point belongs to the polytope (returns True or False).
        """
        if self.hull is None:
            return False
        # is_inside = np.max(self.hull.A.dot(x) - self.hull.b) <= tol # my version
        is_inside = max((self.hull.equations[:,:-1].dot(x) + self.hull.equations[:,-1:]).flatten().tolist()) <= tol # qhull version
        return is_inside