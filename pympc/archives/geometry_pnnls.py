import numpy as np
from scipy.optimize import nnls
from optimization import linear_program
# from pyhull.halfspace import Halfspace
# from pyhull.halfspace import HalfspaceIntersection
import scipy.spatial as spatial


class Polytope:
    """
    Defines a polytope as {x | A * x <= b}. A LP is solved to determine if the polytope is really bounded, whereas a PNNLS is solved to check if it is empty. The emptyness could be determined directly from the solution of the LP for the boundedness but the solution PNNLS is accurate to machine precision and hence it is preferable in case of polytopes of small dimensions.

    VARIABLES:
        A: left-hand side of redundant description of the polytope {x | A * x <= b}
        b: right-hand side of redundant description of the polytope {x | A * x <= b}
        assembled: flag that determines when it isn't possible to add constraints
        empty: flag that determines if the polytope is empty
        bounded: flag that determines if the polytope is bounded (if not a ValueError is thrown)
        coincident_facets: list of of lists of coincident facets (one list for each facet)
        vertices: list of vertices of the polytope (each one is a 1D array)
        minimal_facets: list of indices of the non-redundant facets
        A_min: left-hand side of non-redundant facets
        b_min: right-hand side of non-redundant facets
        facet_centers: list of centers of each non-redundant facet (i.e.: A_min[i,:].dot(facet_centers[i]) = b_min[i])
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
        self.empty = False
        self.bounded = True
        self.normalize()
        self.check_emptyness()
        if self.empty:
            return
        self.check_boundedness()
        if not self.bounded:
            raise ValueError('Unbounded polyhedron: only polytopes allowed')
        self.find_coincident_facets()
        self.find_minimal_facets()
        self.find_facet_centers()
        self.remove_lower_dimensional_facets()
        return
        
        # if self.n_facets < self.n_variables+1:
        #     self.bounded = False
        # elif self.n_variables == 1:
        #     self.assemble_1D()
        # elif self.n_variables > 1:
        #     self.assemble_multiD()
        # if self.empty:
        #     print('Empty polytope')


    def normalize(self, toll=1e-9):
        """
        Normalizes the H-polytope dividing each row of A by its norm and each entry of b by the norm of the corresponding row of A.
        """
        for i in range(0, self.n_facets):
            norm_factor = np.linalg.norm(self.A[i,:])
            if norm_factor < toll:
                print('Cannot normalize facet equation: normaliztion factor is ' + str(norm_factor))
            else:
                self.A[i,:] = self.A[i,:]/norm_factor
                self.b[i] = self.b[i]/norm_factor
        return

    def check_emptyness(self, toll=1.e-14):
        """
        Checks if the polytope is empty solving the PNNLS problem
        minimize ||v + A*x - b||_2^2
        s.t.     v >= 0
        this problem can be solved up to machine precision.
        (From "Bemporad - A Multiparametric Quadratic Programming Algorithm With Polyhedral Computations Based on Nonnegative Least Squares", Lemma 2.)
        """
        _, _, residual = pnnls(np.eye(self.n_facets), self.A, self.b)
        if residual > toll:
            self.empty = True
        return

    def check_boundedness(self):
        """
        Checks if the polytope is empty finding an interior point (Chebychev center) solving the linear program
        minimize y
        s.t.     A * x - b <= y
        if the point has infinite components the polytope is empty (this result rely on the particular implementation of "linear_program"; in case of change of the LP solver check that polyhedrons such as A=[[1,0],[-1,0]] b=[[1],[1]] are correctly detected as unbounded).
        """
        f = np.zeros((self.n_variables+1, 1))
        f[-1] = 1.
        A_ip = np.hstack((self.A, -np.ones((self.n_facets, 1))))
        interior_point, _, _ = linear_program(f, A_ip, self.b)
        interior_point = interior_point[:-1]
        if any(np.isinf(interior_point)):
            self.bounded = False
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

    def find_minimal_facets(self, toll=1.e-14):
        non_coincident_facets = range(0, self.n_facets)
        for facets in self.coincident_facets:
            non_coincident_facets = sorted(list(set(non_coincident_facets) - set(facets[1:])))
        self.minimal_facets = []
        for i in non_coincident_facets:
            C_i = np.delete(np.eye(self.n_facets), i, 1)
            _, _, residual = pnnls(C_i, self.A, self.b)
            if residual < toll:
                self.minimal_facets.append(i)
        self.A_min = self.A[self.minimal_facets,:]
        self.b_min = self.b[self.minimal_facets]
        return

    def find_facet_centers(self):
        self.facet_centers = []
        self.facet_dimensions = []
        f = np.zeros((self.n_variables+1, 1))
        f[-1] = 1.
        n_min = len(self.minimal_facets)
        for i in range(0, n_min):
            A_ip = np.hstack((np.delete(self.A_min, i, 0), -np.ones((n_min-1, 1))))
            b_ip = np.delete(self.b_min, i)
            C_ip = np.reshape(self.A_min[i,:], (1, self.n_variables))
            C_ip = np.hstack((C_ip, np.array([[0.]])))
            d_ip = self.b_min[i,:]
            facet_center, min_distance, _ = linear_program(f, A_ip, b_ip, C_ip, d_ip)
            facet_center = facet_center[:-1]
            self.facet_centers.append(facet_center)
            self.facet_dimensions.append(np.absolute(min_distance[0,0]))

    def remove_lower_dimensional_facets(self, toll=1.e-9):
        full_dimensional_facets = np.where(np.array(self.facet_dimensions) > toll)[0]
        self.minimal_facets = [self.minimal_facets[i] for i in full_dimensional_facets]
        self.facet_centers = [self.facet_centers[i] for i in full_dimensional_facets]
        self.facet_dimensions = [self.facet_dimensions[i] for i in full_dimensional_facets]
        self.A_min = self.A[self.minimal_facets,:]
        self.b_min = self.b[self.minimal_facets]
        return




    # def assemble_1D(self):
    #     upper_bounds = []
    #     lower_bounds = []
    #     for i in range(0, self.n_facets):
    #         if self.A[i] > 0:
    #             upper_bounds.append((self.b[i,0]/self.A[i])[0])
    #             lower_bounds.append(-np.inf)
    #         elif self.A[i] < 0:
    #             upper_bounds.append(np.inf)
    #             lower_bounds.append((self.b[i,0]/self.A[i])[0])
    #         else:
    #             raise ValueError('Invalid constraint!')
    #     upper_bound = min(upper_bounds)
    #     lower_bound = max(lower_bounds)
    #     if lower_bound > upper_bound:
    #         self.empty = True
    #         return
    #     self.vertices = [lower_bound, upper_bound]
    #     if any(np.isinf(self.vertices).flatten()):
    #         self.bounded = False
    #         return
    #     self.minimal_facets = sorted([upper_bounds.index(upper_bound), lower_bounds.index(lower_bound)])
    #     self.A_min = self.A[self.minimal_facets,:]
    #     self.b_min = self.b[self.minimal_facets]
    #     if upper_bounds.index(upper_bound) < lower_bounds.index(lower_bound):
    #         self.facet_centers = [upper_bound, lower_bound]
    #     else:
    #         self.facet_centers = [lower_bound, upper_bound]
    #     self.find_coincident_facets()
    #     return

    # def assemble_multiD(self):
    #     interior_point = self.interior_point(self.A, self.b)
    #     if any(np.isnan(interior_point)):
    #         self.empty = True
    #         return
    #     if any(np.isinf(interior_point).flatten()):
    #         self.bounded = False
    #         return
    #     halfspaces = []
    #     for i in range(0, self.n_facets):
    #         halfspace = Halfspace(self.A[i,:].tolist(), (-self.b[i,0]).tolist())
    #         halfspaces.append(halfspace)
    #     polyhedron_qhull = HalfspaceIntersection(halfspaces, interior_point.flatten().tolist())
    #     self.vertices = polyhedron_qhull.vertices
    #     if any(np.isinf(self.vertices).flatten()):
    #         self.bounded = False
    #         return
    #     self.minimal_facets = []
    #     for i in range(0, self.n_facets):
    #         if polyhedron_qhull.facets_by_halfspace[i]:
    #             self.minimal_facets.append(i)
    #     self.A_min = self.A[self.minimal_facets,:]
    #     self.b_min = self.b[self.minimal_facets]
    #     self.facet_centers = []
    #     for facet in self.minimal_facets:
    #         facet_vertices_inidices = polyhedron_qhull.facets_by_halfspace[facet]
    #         facet_vertices = self.vertices[facet_vertices_inidices]
    #         self.facet_centers.append(np.mean(np.vstack(facet_vertices), axis=0))
    #     self.find_coincident_facets()
    #     return

    # @staticmethod
    # def interior_point(A, b):
    #     """
    #     Finds an interior point solving the linear program
    #     minimize y
    #     s.t.     A * x - b <= y
    #     Returns nan if the polyhedron is empty
    #     Might return inf if the polyhedron is unbounded
    #     """
    #     [n_facets, n_variables] = A.shape
    #     cost_gradient_ip = np.zeros((n_variables+1, 1))
    #     cost_gradient_ip[-1] = 1.
    #     A_ip = np.hstack((A, -np.ones((n_facets, 1))))
    #     [interior_point, penetration] = linear_program(cost_gradient_ip, A_ip, b)[0:2]
    #     interior_point = interior_point[0:-1]
    #     if penetration > 0:
    #         interior_point[:] = np.nan
    #     return interior_point

    

    # def check_intersections_with(self, A_2, b_2):
    #     A_12 = np.vstack((self.A, A_2))
    #     b_12 = np.vstack((self.b, b_2))
    #     interior_point = self.interior_point(A_12, b_12)
    #     intersection = True
    #     if any(np.isnan(interior_point)):
    #         intersection = False
    #     return intersection

    # def convex_union_with(self, A_list, b_list):
    #     """
    #     Algorithm 4.1 from "Bemporad et al. - Convexity recognition of the union of polyhedra"
    #     """
    #     if notself.assembled:
    #         raise ValueError('Polytope already assembled, cannot assemble again!')
    #     n_addition = len(A_list)
    #     if len(b_list) != n_addition:
    #         raise ValueError('Inconsistent input dimensions')
    #     polytope_list = []
    #     for i in range(0, n_addition):
    #         polytope_i = Polytope(A_list[i], b_list[i])
    #         polytope_i.assemble()
    #         polytope_list.append(polytope_i)
    #     for i in range(0, n_addition)



    # def plot(self, dim_proj=[0,1], **kwargs):
    #     """
    #     Plots a 2d projection of the polytope.

    #     INPUTS:
    #         dim_proj: dimensions in which to project the polytope

    #     OUTPUTS:
    #         polytope_plot: figure handle
    #     """
    #     if self.empty:
    #         raise ValueError('Empty polytope!')
    #     if len(dim_proj) != 2:
    #         raise ValueError('Only 2d polytopes!')
    #     # extract vertices components
    #     vertices_proj = np.vstack(self.vertices)[:,dim_proj]
    #     hull = spatial.ConvexHull(vertices_proj)
    #     for simplex in hull.simplices:
    #         polytope_plot, = plt.plot(vertices_proj[simplex, 0], vertices_proj[simplex, 1], **kwargs)
    #     plt.xlabel(r'$x_' + str(dim_proj[0]+1) + '$')
    #     plt.ylabel(r'$x_' + str(dim_proj[1]+1) + '$')
    #     return polytope_plot

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

def pnnls(A, B, c):
    """
    Solves the Partial Non-Negative Least Squares problem
    minimize ||A*v + B*u - c||_2^2
    s.t.     v >= 0
    through a NNLS solver.
    (From "Bemporad - A Multiparametric Quadratic Programming Algorithm With Polyhedral Computations Based on Nonnegative Least Squares", Lemma 1.)

    INPUTS:
        A: coefficient matrix for v in the PNNLS problem
        B: coefficient matrix for u in the PNNLS problem
        c: offset term in the PNNLS problem

    OUTPUTS:
        v_star: optimal values of v
        u_star: optimal values of u
        r_star: minimum of least squares
    """
    [n_ineq, n_v] = A.shape
    n_u = B.shape[1]
    B_pinv = np.linalg.pinv(B)
    B_bar = np.eye(n_ineq) - B.dot(B_pinv)
    # print B_bar
    A_bar = B_bar.dot(A)
    b_bar = B_bar.dot(c)
    [v_star, r_star] = nnls(A_bar, b_bar.flatten())
    v_star = np.reshape(v_star, (n_v,1))
    u_star = -B_pinv.dot(A.dot(v_star) - c)
    return [v_star, u_star, r_star]

# def interior_point(A, b):
#     """
#     Finds an interior point solving the linear program
#     minimize y
#     s.t.     A * x - b <= y
#     Returns nan if the polyhedron is empty
#     Might return inf if the polyhedron is unbounded
#     """
#     [n_facets, n_variables] = A.shape
#     cost_gradient_ip = np.zeros((n_variables+1, 1))
#     cost_gradient_ip[-1] = 1.
#     A_ip = np.hstack((A, -np.ones((n_facets, 1))))
#     [interior_point, penetration] = linear_program(cost_gradient_ip, A_ip, b)[0:2]
#     interior_point = interior_point[0:-1]
#     if penetration > 0:
#         interior_point[:] = np.nan
#     penetration
#     return interior_point

# # def linear_program_with_pnnls(f,)














