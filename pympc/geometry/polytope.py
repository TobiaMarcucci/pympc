import numpy as np
import matplotlib.pyplot as plt
from pyhull.halfspace import Halfspace, HalfspaceIntersection
import cdd
from pympc.optimization.pnnls import linear_program
from pympc.geometry.chebyshev_center import chebyshev_center
from pympc.geometry.orthogonal_projection_CHM import orthogonal_projection_CHM
import scipy.spatial as spatial
from matplotlib.path import Path
import matplotlib.patches as patches


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
        return self

    def add_bounds(self, x_min, x_max):
        if self.assembled:
            raise ValueError('Polytope already assembled, cannot add bounds!')
        n_variables = x_max.shape[0]
        A = np.vstack((np.eye(n_variables), -np.eye(n_variables)))
        b = np.vstack((x_max, -x_min))
        self.add_facets(A, b)
        return self

    def assemble(self, redundant=True, vertices=None):
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
        if redundant:
            self.find_minimal_facets()
        else:
            self.minimal_facets = range(self.n_facets)
            self.lhs_min = self.A
            self.rhs_min = self.b
        self._vertices = vertices
        self._facet_centers = [None] * len(self.minimal_facets)
        self._facet_radii = [None] * len(self.minimal_facets)
        self._x_min = None
        self._x_max = None
        return self

    def normalize(self, tol=1e-9):
        """
        Normalizes the H-polytope dividing each row of A by its norm and each entry of b by the norm of the corresponding row of A.
        """
        for i in range(0, self.n_facets):
            norm_factor = np.linalg.norm(self.A[i,:])
            if norm_factor > tol:
                self.A[i,:] = self.A[i,:]/norm_factor
                self.b[i] = self.b[i]/norm_factor
        return self

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

    def check_boundedness(self, tol=1.e-9):
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
        rank_A = np.linalg.matrix_rank(self.A, tol)
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
        sol = linear_program(f, A, b, C, d)
        if any(np.isnan(sol.argmin)):
            self.bounded = False
            print 'Boundedness test failed!'
        return

    def find_coincident_facets(self, rel_tol=1e-9, abs_tol=1e-9):
        """
        For each facet of the (potentially redundant) polytope finds the set of coincident facets and stores a list whose ith entry is the list of the facetes coincident with the ith facets (the index i itself is included in this list; e.g., if the ith and the jth facets are coincident coincident_facets[i] = coincident_facets[j] = [i,j]).
        """
        # coincident facets indices
        self.coincident_facets = []
        Ab = np.hstack((self.A, self.b))
        for i in range(0, self.n_facets):
            coincident_flag_matrix = np.isclose(Ab, Ab[i,:], rel_tol, abs_tol)
            coincident_flag_vector = np.all(coincident_flag_matrix, axis=1)
            coincident_facets_i = sorted(np.where(coincident_flag_vector)[0].tolist())
            self.coincident_facets.append(coincident_facets_i)
        return

    def find_minimal_facets(self, tol=1e-9):
        """
        Finds the non-redundant facets and derives a minimal representation of the polyhedron solving a LP for each facet. See "Fukuda - Frequently asked questions in polyhedral computation" Sec.2.21.
        """
        # list of non-redundant facets
        self.minimal_facets = range(self.n_facets)
        for i in range(self.n_facets):
            # remove redundant constraints
            A_reduced = self.A[self.minimal_facets,:]
            # relax the ith constraint
            b_relaxation = np.zeros(np.shape(self.b))
            b_relaxation[i] += 1.
            b_relaxed = (self.b + b_relaxation)[self.minimal_facets];
            # check redundancy
            sol = linear_program(-self.A[i,:], A_reduced, b_relaxed)
            cost_i = - sol.min
            # remove redundant facets from the list
            if cost_i - self.b[i] < tol or np.isnan(cost_i):
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
            self._vertices = [np.reshape(vertex, (vertex.shape[0],1)) for vertex in self._vertices]
        return self._vertices

    def orthogonal_projection(self, residual_variables, mehtod='convex_hull'):

        if mehtod == 'convex_hull':
            A_proj, b_proj, v_proj = orthogonal_projection_CHM(self.lhs_min, self.rhs_min, residual_variables)
            p_proj = Polytope(A_proj, b_proj)
            p_proj.assemble(redundant=False, vertices=v_proj)

        # elif mehtod == 'block_elimination':
        #     drop_variables = [i+1 for i in range(self.n_variables) if i not in residual_variables]
        #     M = np.hstack((self.rhs_min, -self.lhs_min))
        #     M = cdd.Matrix([list(m) for m in M])
        #     M.rep_type = cdd.RepType.INEQUALITY
        #     M_proj = M.block_elimination(drop_variables)
        #     M_proj.canonicalize()
        #     A_proj = - np.array([list(M_proj[i])[1:] for i in range(M_proj.row_size)])
        #     b_proj = np.array([list(M_proj[i])[:1] for i in range(M_proj.row_size)])
        #     p_proj = Polytope(A_proj, b_proj)
        #     p_proj.assemble()
            
        elif mehtod == 'vertex_enumeration':
            v_proj = np.hstack(self.vertices).T[:,residual_variables]
            if len(residual_variables) > 1:
                hull = spatial.ConvexHull(v_proj)
                A_proj = np.array(hull.equations)[:, :-1]
                b_proj = - np.array(hull.equations)[:, -1:]
            else:
                A_proj = np.array([[1.],[-1.]])
                b_proj = np.array([[max(v_proj)[0]],[-min(v_proj)[0]]])
            p_proj = Polytope(A_proj, b_proj)
            p_proj.assemble()

        return p_proj

    def intersect_with(self, p):
        """
        Checks if the polytope intersect the other polytope p (returns True or False).
        """
        intersection = True
        A = np.vstack((self.lhs_min, p.lhs_min))
        b = np.vstack((self.rhs_min, p.rhs_min))
        f = np.zeros((A.shape[1],1))
        sol = linear_program(f, A, b)
        if any(np.isnan(sol.argmin)):
            intersection = False
        return intersection

    def included_in(self, p, tol=1.e-6):
        """
        Checks if the polytope is a subset of the polytope p (returns True or False).
        """
        inclusion = True
        for i in range(p.lhs_min.shape[0]):
            sol = linear_program(-p.lhs_min[i,:], self.lhs_min, self.rhs_min)
            penetration = - sol.min - p.rhs_min[i]
            if penetration > tol:
                inclusion = False
                break
        return inclusion

    def applies_to(self, x, tol=1.e-9):
        """
        Determines if the given point belongs to the polytope (returns True or False).
        """
        is_inside = np.max(self.lhs_min.dot(x) - self.rhs_min) <= tol
        return is_inside

    @property
    def x_min(self):
        if self._x_min is None:
            self._x_min = []
            for i in range(self.n_variables):
                f = np.zeros((self.n_variables, 1))
                f[i,:] += 1.
                sol = linear_program(f, self.lhs_min, self.rhs_min)
                self._x_min.append([sol.min])
            self._x_min = np.array(self._x_min)
        return self._x_min

    @property
    def x_max(self):
        if self._x_max is None:
            self._x_max = []
            for i in range(self.n_variables):
                f = np.zeros((self.n_variables, 1))
                f[i,:] += 1.
                sol = linear_program(-f, self.lhs_min, self.rhs_min)
                self._x_max.append([-sol.min])
            self._x_max = np.array(self._x_max)
        return self._x_max

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
        vertices_proj = np.hstack(self.vertices).T[:,dim_proj]
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
    def from_bounds(x_min, x_max):
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