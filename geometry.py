import numpy as np
from optimization.pnnls import linear_program
from pyhull.halfspace import Halfspace, HalfspaceIntersection
import scipy.spatial as spatial
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import copy
import time

class Polytope:
    """
    Defines a polytope as {x | A * x <= b}.

    VARIABLES:
        A: left-hand side of redundant description of the polytope {x | A * x <= b}
        b: right-hand side of redundant description of the polytope {x | A * x <= b}
        n_facets: number of facets of the polyhedron
        n_variables: dimension of the variable x
        assembled: flag that determines when it isn't possible to add constraints
        empty: True if the polytope is empty, False otherwise
        bounded: True if the polytope is bounded, False otherwise (if False a ValueError is thrown)
        center: Chebyshev center of the polytope
        radius: Chebyshev radius of the polytope
        coincident_facets: list of of lists of coincident facets (one list for each facet)
        minimal_facets: list of indices of the non-redundant facets
        A_min: left-hand side of non-redundant facets
        b_min: right-hand side of non-redundant facets
        facet_centers: list of Chebyshev centers of each non-redundant facet (i.e.: A_min[i,:].dot(facet_centers[i]) = b_min[i])
        facet_radii: list of Chebyshev radii of each non-redundant facet
        vertices: list of vertices of the polytope (each one is a 1D array)
    """

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.assembled = False
        return

    def add_facets(self, A, b):
        if self.assembled:
            raise ValueError('Polytope already assembled, cannot add facets!')
        self.A = np.vstack((self.A, A))
        self.b = np.vstack((self.b, b))
        return

    def add_bounds(self, x_max, x_min):
        if self.assembled:
            raise ValueError('Polytope already assembled, cannot add bounds!')
        n_variables = x_max.shape[0]
        A = np.vstack((np.eye(n_variables), -np.eye(n_variables)))
        b = np.vstack((x_max, -x_min))
        self.add_facets(A, b)
        return

    def assemble(self):
        if self.assembled:
            raise ValueError('Polytope already assembled, cannot assemble again!')
        self.assembled = True
        [self.n_facets, self.n_variables] = self.A.shape
        self.normalize()
        self.check_emptiness()
        if self.empty:
            return
        self.check_boundedness()
        if not(self.bounded):
            raise ValueError('Unbounded polyhedron: only polytopes allowed')
        self.find_coincident_facets()
        self.find_minimal_facets()
        self._facet_centers = [None] * len(self.minimal_facets)
        self._facet_radii = [None] * len(self.minimal_facets)
        self._vertices = None
        return

    def normalize(self, toll=1e-9):
        """
        Normalizes the H-polytope dividing each row of A by its norm and each entry of b by the norm of the corresponding row of A.
        """
        for i in range(0, self.n_facets):
            norm_factor = np.linalg.norm(self.A[i,:])
            if norm_factor > toll:
                self.A[i,:] = self.A[i,:]/norm_factor
                self.b[i] = self.b[i]/norm_factor
        return

    def check_emptiness(self):
        """
        Checks if the polytope is empty finding its Chebychev center and radius.
        """
        self.empty = False
        self.center, self.radius = chebyshev_center(self.A, self.b)
        if np.isnan(self.radius):
            self.empty = True
            print('Empty polytope!')
        return

    def check_boundedness(self, toll=1.e-9):
        """
        Checks if the polyhedron is bounded: a polyhedron is unbounded (i.e. a polytope) iff there exists an x != 0 in the recession cone (A*x <= 0). We also have that { exists x != 0 | A*x <= 0 } <=> { exists z < 0 | A^T*z = 0 }. The second condition is tested through a LP.
        """
        self.bounded = True
        # if the Chebyshev radius is infinite
        if np.isinf(self.radius):
            print('Infinite Chebyshev center or radius!')
            self.bounded = False
            return
        # if ker(A) != 0
        rank_A = np.linalg.matrix_rank(self.A, toll)
        dim_null_A = self.A.shape[1] - rank_A
        if dim_null_A > 0:
            self.bounded = False
            return
        # else solve lp
        f = np.zeros((self.A.shape[0],1))
        A = np.eye(self.A.shape[0])
        b = - np.ones((self.A.shape[0],1))
        C = self.A.T
        d = np.zeros((self.A.shape[1],1))
        feasible_point, _ = linear_program(f, A, b, C, d)
        if any(np.isnan(feasible_point)):
            self.bounded = False
            print 'Boundedness test failed!'
        return

    def find_coincident_facets(self, rel_toll=1e-9, abs_toll=1e-9):
        """
        For each facet of the (potentially redundant) polytope finds the set of coincident facets and stores a list whose ith entry is the list of the facetes coincident with the ith facets (the index i itself is included in this list; e.g., if the ith and the jth facets are coincident coincident_facets[i] = coincident_facets[j] = [i,j]).
        """
        # coincident facets indices
        self.coincident_facets = []
        Ab = np.hstack((self.A, self.b))
        for i in range(0, self.n_facets):
            coincident_flag_matrix = np.isclose(Ab, Ab[i,:], rel_toll, abs_toll)
            coincident_flag_vector = np.all(coincident_flag_matrix, axis=1)
            coincident_facets_i = sorted(np.where(coincident_flag_vector)[0].tolist())
            self.coincident_facets.append(coincident_facets_i)
        return

    def find_minimal_facets(self, toll=1e-9):
        """
        Finds the non-redundant facets and derives a minimal representation of the polyhedron solving a LP for each facet. See "Fukuda - Frequently asked questions in polyhedral computation" Sec.2.21.
        """
        # list of non-redundant facets
        self.minimal_facets = range(0, self.n_facets)
        for i in range(0, self.n_facets):
            # remove redundant constraints
            A_reduced = self.A[self.minimal_facets,:]
            # relax the ith constraint
            b_relaxation = np.zeros(np.shape(self.b))
            b_relaxation[i] += 1.
            b_relaxed = (self.b + b_relaxation)[self.minimal_facets];
            # check redundancy
            cost_i = - linear_program(-self.A[i,:], A_reduced, b_relaxed)[1]
            # remove redundant facets from the list
            if cost_i - self.b[i] < toll or np.isnan(cost_i):
                self.minimal_facets.remove(i)
        # list of non redundant facets
        self.lhs_min = self.A[self.minimal_facets,:]
        self.rhs_min = self.b[self.minimal_facets]
        return

    def facet_centers(self, i):
        if self._facet_centers[i] is None:
            A_lp = np.delete(self.lhs_min, i, 0)
            b_lp = np.delete(self.rhs_min, i, 0)
            C_lp = np.reshape(self.lhs_min[i,:], (1, self.n_variables))
            d_lp = self.rhs_min[i,:]
            self._facet_centers[i], self._facet_radii[i] = chebyshev_center(A_lp, b_lp, C_lp, d_lp)
        return self._facet_centers[i]

    def facet_radii(self, i):
        if self._facet_radii[i] is None:
            A_lp = np.delete(self.lhs_min, i, 0)
            b_lp = np.delete(self.rhs_min, i, 0)
            C_lp = np.reshape(self.lhs_min[i,:], (1, self.n_variables))
            d_lp = self.rhs_min[i,:]
            self._facet_centers[i], self._facet_radii[i] = chebyshev_center(A_lp, b_lp, C_lp, d_lp)
        return self._facet_radii[i]

    @property
    def vertices(self):
        """
        Calls qhull for determining the vertices of the polytope (it computes the vertices only when called).
        """
        if self._vertices is None:
            if self.n_variables == 1:
                self._vertices = [np.array([[self.rhs_min[i,0]/self.lhs_min[i,0]]]) for i in [0,1]]
                return self._vertices
            if self.empty:
                self._vertices = []
                return self._vertices
            halfspaces = []
            # change of coordinates because qhull is stupid...
            b_qhull = self.rhs_min - self.lhs_min.dot(self.center)
            for i in range(self.lhs_min.shape[0]):
                halfspace = Halfspace(self.lhs_min[i,:].tolist(), (-b_qhull[i,0]).tolist())
                halfspaces.append(halfspace)
            polyhedron_qhull = HalfspaceIntersection(halfspaces, np.zeros(self.center.shape).flatten().tolist())
            self._vertices = polyhedron_qhull.vertices
            self._vertices += np.repeat(self.center.T, self._vertices.shape[0], axis=0)
        return self._vertices

    def orthogonal_projection(self, dim_proj):
        """
        Projects the polytope in the given directions: from H-rep to V-rep, keeps the component of the vertices in the projected dimensions, from V-rep to H-rep.
        """
        vertices_proj = np.vstack(self.vertices)[:,dim_proj]
        if len(dim_proj) > 1:
            hull = spatial.ConvexHull(vertices_proj)
            lhs = np.array(hull.equations)[:, :-1]
            rhs = - (np.array(hull.equations)[:, -1]).reshape((lhs.shape[0], 1))
        else:
            lhs = np.array([[1.],[-1.]])
            rhs = np.array([[max(vertices_proj)[0]],[-min(vertices_proj)[0]]])
        projected_polytope = Polytope(lhs, rhs)
        projected_polytope.assemble()
        return projected_polytope


    def plot(self, dim_proj=[0,1], largest_ball=False, **kwargs):
        """
        Plots a 2d projection of the polytope.

        INPUTS:
            dim_proj: dimensions in which to project the polytope
        """
        if self.empty:
            print('Cannot plot empty polytope')
            return
        if len(dim_proj) != 2:
            raise ValueError('Only 2d polytopes!')
        # extract vertices components
        vertices_proj = np.vstack(self.vertices)[:,dim_proj]
        hull = spatial.ConvexHull(vertices_proj)
        verts = [hull.points[i].tolist() for i in hull.vertices]
        verts += [verts[0]]
        codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-2) + [Path.CLOSEPOLY]
        path = Path(verts, codes)
        ax = plt.gca()
        patch = patches.PathPatch(path, **kwargs)
        ax.add_patch(patch)
        plt.xlabel(r'$x_' + str(dim_proj[0]+1) + '$')
        plt.ylabel(r'$x_' + str(dim_proj[1]+1) + '$')
        ax.autoscale_view()
        if largest_ball:
            circle = plt.Circle((self.center[0], self.center[1]), self.radius, facecolor='white', edgecolor='black')
            ax.add_artist(circle)
        return

    @staticmethod
    def from_bounds(x_max, x_min):
        """
        Returns a polytope defines through the its bounds.

        INPUTS:
            x_max: upper bound of the variables
            x_min: lower bound of the variables

        OUTPUTS:
            p: polytope
        """
        n = x_max.shape[0]
        A = np.vstack((np.eye(n), -np.eye(n)))
        b = np.vstack((x_max, -x_min))
        p = Polytope(A, b)
        return p

    def applies_to(self, x, tol=1.e-9):
        """
        Determines is a given point belongs to the polytope.

        INPUTS:
            x: tested point

        OUTPUTS:
            is_inside: flag (True if x is in the polytope, False otherwise)
        """

        # check if x is inside the polytope
        is_inside = np.max(self.lhs_min.dot(x) - self.rhs_min) <= tol

        return is_inside


    # def fourier_motzkin_elimination(self, variable_index, tol=1.e-12):
    #     [n_facets, n_variables] = self.lhs_min.shape
    #     lhs_leq = np.zeros((0, n_variables-1))
    #     rhs_leq = np.zeros((0, 1))
    #     lhs_geq = np.zeros((0, n_variables-1))
    #     rhs_geq = np.zeros((0, 1))
    #     lhs = np.zeros((0, n_variables-1))
    #     rhs = np.zeros((0, 1))
    #     for i in range(n_facets):
    #         pivot = self.lhs_min[i, variable_index]
    #         lhs_row = np.hstack((self.lhs_min[i,:variable_index], self.lhs_min[i,variable_index+1:]))
    #         rhs_row = self.rhs_min[i, 0]
    #         if pivot > tol:
    #             lhs_leq = np.vstack((lhs_leq, lhs_row/pivot))
    #             rhs_leq = np.vstack((rhs_leq, rhs_row/pivot))
    #         elif pivot < -tol:
    #             lhs_geq = np.vstack((lhs_geq, lhs_row/pivot))
    #             rhs_geq = np.vstack((rhs_geq, rhs_row/pivot))
    #         else:
    #             lhs = np.vstack((lhs, lhs_row))
    #             rhs = np.vstack((rhs, rhs_row))
    #     for i in range(lhs_leq.shape[0]):
    #         for j in range(lhs_geq.shape[0]):
    #             lhs_row = lhs_leq[i,:] - lhs_geq[j,:]
    #             rhs_row = rhs_leq[i,0] - rhs_geq[j,0]
    #             lhs = np.vstack((lhs, lhs_row))
    #             rhs = np.vstack((rhs, rhs_row))
    #     p = Polytope(lhs,rhs)
    #     p.assemble()
    #     return p


    # def orthogonal_projection_test(self, projection_dimensions, tol=1.e-8):

    #     # find first 2 vertices of the projection minimizing and maximizing in the first direction
    #     growth_direction = np.zeros((self.n_variables, 1))
    #     growth_direction[projection_dimensions[0],0] = 1.
        
    #     v0 = linear_program(-growth_direction, self.lhs_min, self.rhs_min)[0]
    #     v1 = linear_program(growth_direction, self.lhs_min, self.rhs_min)[0]
    #     v_list = [v0[projection_dimensions], v1[projection_dimensions]]

    #     # find inner approximation of the convex hull of the projection
    #     for i in range(2, len(projection_dimensions)+1):

    #         # find the gradient of plane conataining all the vertices in the lower-dimensional space (lhs * x = rhs = 1)
    #         growth_direction = np.zeros((self.n_variables, 1))
    #         v_lower_dim = np.hstack([v[:i] for v in v_list])
    #         gradient_lower_dim = np.sum(np.linalg.inv(v_lower_dim), axis=0)
    #         growth_direction[projection_dimensions[:i],0] = gradient_lower_dim

    #         # move in the (+-) growth_direction to find a vertex
    #         vi = linear_program(growth_direction, self.lhs_min, self.rhs_min)[0]
    #         vi = vi[projection_dimensions]
    #         if any([np.linalg.norm(vi - v) < tol for v in v_list]):
                
    #             vi = linear_program(-growth_direction, self.lhs_min, self.rhs_min)[0]
    #             vi = vi[projection_dimensions]
    #         v_list.append(vi)

    #     # check that the number of vertices is the roght one
    #     if len(v_list) != len(projection_dimensions)+1:
    #         raise ValueError('Not enough vertices from the pre-processing phase!')

    #     # compute the remaining vertices of the projection
    #     # initialize the polytope projection
    #     convergence = False
    #     lhs_projection = []
    #     rhs_projection = []
    #     hull = spatial.ConvexHull(np.hstack(v_list).T, incremental=True)

    #     # check if each facet is a boundary
    #     while not convergence:
    #         internal_facets = 0
    #         v_new = []

    #         # temporary H-rep of the projection
    #         lhs_temp = np.array(hull.equations)[:, :-1]
    #         rhs_temp = - (np.array(hull.equations)[:, -1:])

    #         # temporary H-rep of the projection in the higher dimensional sapce
    #         lhs_temp_higher_dim = np.zeros((lhs_temp.shape[0], self.n_variables))
    #         lhs_temp_higher_dim[:,projection_dimensions] = lhs_temp

    #         for i in range(lhs_temp.shape[0]):
    #             if not any([np.linalg.norm(lhs_temp[i,:] - facet) < tol for facet in lhs_projection]):
    #                 vi, ci = linear_program(-lhs_temp_higher_dim[i,:], self.lhs_min, self.rhs_min)
    #                 vi = vi[projection_dimensions]
    #                 if - rhs_temp[i,0] - ci > tol:
    #                     internal_facets += 1
    #                     if all([np.linalg.norm(vi - v) > tol for v in v_list]):
    #                         v_new.append(vi)
    #                         v_list.append(vi)
    #                 else:
    #                     lhs_projection.append(lhs_temp[i,:])
    #                     rhs_projection.append([rhs_temp[i,0]])

    #         if internal_facets == 0:
    #             convergence = True
    #         else:
    #             hull.add_points(np.hstack(v_new).T)

    #     p = Polytope(np.array(lhs_projection), np.array(rhs_projection))
    #     p.assemble()
    #     p.vertices = hull.points

    #     return p




    # def orthogonal_projection(self, projection_dimensions):
    #     """
    #     Projects the polytope in the given directions: from H-rep to V-rep, keeps the component of the vertices in the projected dimensions, from V-rep to H-rep.
    #     """
    #     remove_dimensions = sorted(list(set(range(self.lhs_min.shape[1])) - set(projection_dimensions)))
    #     p = self
    #     for dimension in reversed(remove_dimensions):
    #         p = p.fourier_motzkin_elimination(dimension)
    #     return p











def chebyshev_center(A, b, C=None, d=None, tol=1.e-10):
    """
    Finds the Chebyshevcenter of the polytope P := {x | A*x <= b, C*x = d} solving the linear program
    min  e
    s.t. F * z <= g + g_{\|}e
    where if an equality is not provided F=A, z=x, g=b; whereas if equalities are present F=A*Z, g=b-A*Y*y, with: Z basis of the nullspace of C, Y orthogonal complement to Z, y=(C*Y)^-1*d and x is retrived as x=Z*z+Y*y.

    INPUTS:
        A: left-hand side of the inequalities
        b: right-hand side of the inequalities
        C: left-hand side of the equalities
        d: right-hand side of the equalities

    OUTPUTS:
        center: Chebyshev center of the polytope (nan if the P is empty, inf if P is unbounded and the center is at infinity)
        radius: Chebyshev radius of the polytope (nan if the P is empty, inf if it is infinite)
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
    center, radius = linear_program(f_lp, A_lp, b_projected)
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

def nullspace_basis(A):
    """
    Uses singular value decomposition to find a basis of the nullsapce of A.
    """
    V = np.linalg.svd(A)[2].T
    rank = np.linalg.matrix_rank(A)
    Z = V[:,rank:]
    return Z








