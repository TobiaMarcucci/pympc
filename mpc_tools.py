import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import time
from utils.ndpiecewise import NDPiecewise
from optimization.pnnls import linear_program
from optimization.drake import quadratic_program
from geometry import Polytope
import copy


def plot_input_sequence(u_sequence, t_s, N, u_max=None, u_min=None):
    """
    Plots the input sequence and its bounds as functions of time.

    INPUTS:
        u_sequence: list with N inputs (2D numpy vectors) of dimension (n_u,1) each
        t_s: sampling time
        N: number of steps
        u_max: upper bound on the input (2D numpy vectors of dimension (n_u,1))
        u_min: lower bound on the input (2D numpy vectors of dimension (n_u,1))
    """

    # dimension of the input
    n_u = u_sequence[0].shape[0]

    # time axis
    t = np.linspace(0,N*t_s,N+1)

    # plot each input element separately
    for i in range(0, n_u):
        plt.subplot(n_u, 1, i+1)

        # plot input sequence
        u_i_sequence = [u_sequence[j][i] for j in range(0,N)]
        input_plot, = plt.step(t, [u_i_sequence[0]] + u_i_sequence, 'b')

        # plot bounds iff provided
        if u_max is not None:
            bound_plot, = plt.step(t, u_max[i,0]*np.ones(t.shape), 'r')
        if u_min is not None:
            bound_plot, = plt.step(t, u_min[i,0]*np.ones(t.shape), 'r')

        # miscellaneous options
        plt.ylabel(r'$u_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            if u_max is not None or u_min is not None:
                plt.legend([input_plot, bound_plot], ['Optimal control', 'Control bounds'], loc=1)
            else:
                plt.legend([input_plot], ['Optimal control'], loc=1)
    plt.xlabel(r'$t$')

    return



def plot_state_trajectory(x_trajectory, t_s, N, x_max=None, x_min=None):
    """
    Plots the state trajectory and its bounds as functions of time.

    INPUTS:
        x_trajectory: list with N+1 states (2D numpy vectors) of dimension (n_x,1) each
        t_s: sampling time
        N: number of steps
        x_max: upper bound on the state (2D numpy vectors of dimension (n_x,1))
        x_min: lower bound on the state (2D numpy vectors of dimension (n_x,1))
    """

    # dimension of the state
    n_x = x_trajectory[0].shape[0]

    # time axis
    t = np.linspace(0,N*t_s,N+1)

    # plot each state element separately
    for i in range(0, n_x):
        plt.subplot(n_x, 1, i+1)

        # plot state trajectory
        x_i_trajectory = [x_trajectory[j][i] for j in range(0,N+1)]
        state_plot, = plt.plot(t, x_i_trajectory, 'b')

        # plot bounds if provided
        if x_max is not None:
            bound_plot, = plt.step(t, x_max[i,0]*np.ones(t.shape),'r')
        if x_min is not None:
            bound_plot, = plt.step(t, x_min[i,0]*np.ones(t.shape),'r')

        # miscellaneous options
        plt.ylabel(r'$x_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            if x_max is not None or x_min is not None:
                plt.legend([state_plot, bound_plot], ['Optimal trajectory', 'State bounds'], loc=1)
            else:
                plt.legend([state_plot], ['Optimal trajectory'], loc=1)
    plt.xlabel(r'$t$')

    return



class DTLinearSystem:
    """
    Discrete time linear systems in the form x_{k+1} = A*x_k + B*u_k.

    VARIABLES:
        A: discrete time state transition matrix
        B: discrete time input to state map
        n_x: number of sates
        n_u: number of inputs
    """

    def __init__(self, A, B):
        self.A = A
        self.B = B
        [self.n_x, self.n_u] = np.shape(B)
        return

    def evolution_matrices(self, N):
        """
        Returns the free and forced evolution matrices for the linear system
        (i.e. [x_1^T, ...,  x_N^T]^T = free_evolution*x_0 + forced_evolution*[u_0^T, ...,  u_{N-1}^T]^T)

        INPUTS:
            N: number of steps

        OUTPUTS:
            free_evolution: free evolution matrix
            forced_evolution: forced evolution matrix
        """

        # free evolution of the system
        free_evolution = np.vstack([np.linalg.matrix_power(self.A,k) for k in range(1, N+1)])

        # forced evolution of the system
        forced_evolution = np.zeros((self.n_x*N,self.n_u*N))
        for i in range(0, N):
            for j in range(0, i+1):
                forced_evolution[self.n_x*i:self.n_x*(i+1),self.n_u*j:self.n_u*(j+1)] = np.linalg.matrix_power(self.A,i-j).dot(self.B)

        return [free_evolution, forced_evolution]

    def simulate(self, x0, N, u_sequence=None):
        """
        Returns the list of states obtained simulating the system dynamics.

        INPUTS:
            x0: initial state of the system
            N: number of steps
            u_sequence: list of inputs [u_1, ..., u_{N-1}]

        OUTPUTS:
            x_trajectory: list of states [x_0, ..., x_N]
        """

        # reshape input list if provided
        if u_sequence is None:
            u_sequence = np.zeros((self.n_u*N, 1))
        else:
            u_sequence = np.vstack(u_sequence)

        # derive evolution matrices
        [free_evolution, forced_evolution] = self.evolution_matrices(N)

        # derive state trajectory includion initial state
        if x0.ndim == 1:
            x0 = np.reshape(x0, (x0.shape[0],1))
        x = free_evolution.dot(x0) + forced_evolution.dot(u_sequence)
        x_trajectory = [x0]
        [x_trajectory.append(x[self.n_x*i:self.n_x*(i+1)]) for i in range(0,N)]

        return x_trajectory

    @staticmethod
    def from_continuous(t_s, A, B):
        """
        Defines a discrete time linear system starting from the continuous time dynamics \dot x = A*x + B*u
        (the exact zero order hold method is used for the discretization).

        INPUTS:
            t_s: sampling time
            A: continuous time state transition matrix
            B: continuous time input to state map

        OUTPUTS:
            sys: discrete time linear system
        """

        # system dimensions
        n_x = np.shape(A)[0]
        n_u = np.shape(B)[1]

        # zero order hold (see Bicchi - Fondamenti di Autometica 2)
        mat_c = np.zeros((n_x+n_u, n_x+n_u))
        mat_c[0:n_x,:] = np.hstack((A, B))
        mat_d = linalg.expm(mat_c*t_s)

        # discrete time dynamics
        A_d = mat_d[0:n_x, 0:n_x]
        B_d = mat_d[0:n_x, n_x:n_x+n_u]

        sys = DTLinearSystem(A_d, B_d)
        return sys

class MPCController:

    def __init__(self, canonical_qp, n_u):
        self.qp = canonical_qp
        self.n_u = n_u

    def feedforward(self, x0):
        u_feedforward = quadratic_program(self.qp.H, (x0.T.dot(self.qp.F)).T, self.qp.G, self.qp.W + self.qp.E.dot(x0))[0]
        if any(np.isnan(u_feedforward).flatten()):
            print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
        return u_feedforward

    def feedback(self, x0):
        u_feedback = self.feedforward(x0)[0:self.n_u]
        return u_feedback

    def compute_explicit_solution(self):

        # start clock
        tic = time.clock()

        # initialize the search with the origin (to which the empty AS is associated)
        active_set = []
        cr0 = CriticalRegion(active_set, self.qp)
        cr_to_be_explored = [cr0]
        explored_cr = []
        tested_active_sets =[cr0.active_set]

        # explore the state space
        while cr_to_be_explored:

            # choose the first candidate in the list and remove it
            cr = cr_to_be_explored[0]
            cr_to_be_explored = cr_to_be_explored[1:]

            # if the CR is not empty, find all the potential neighbors
            if cr.polytope.empty:
                print('Empty critical region detected')
            else:
                [cr_to_be_explored, tested_active_sets] = self.spread_critical_region(cr, cr_to_be_explored, tested_active_sets)
                explored_cr.append(cr)

        # collect all the critical regions and report the result
        self.critical_regions = NDPiecewise(explored_cr)
        toc = time.clock()
        print('\nExplicit solution successfully computed in ' + str(toc-tic) + ' s:')
        print('parameter space partitioned in ' + str(len(self.critical_regions)) + ' critical regions.')

    def spread_critical_region(self, cr, cr_to_be_explored, tested_active_sets):

        # for all the facets of the CR and all candidate active sets across each facet
        for facet_index in range(0, len(cr.polytope.minimal_facets)):
            for active_set in cr.candidate_active_sets[facet_index]:

                # add active set to the list of tested be sure to not explore an active set twice
                if active_set not in tested_active_sets:
                    tested_active_sets.append(active_set)

                    # check LICQ for the given active set
                    licq_flag = self.licq_check(self.qp.G, active_set)

                    # if LICQ holds, determine the critical region
                    if licq_flag:
                        cr_to_be_explored.append(CriticalRegion(active_set, self.qp))

                    # if LICQ doesn't hold, correct the active set and determine the critical region
                    else:
                        print('LICQ does not hold for the active set ' + str(active_set))
                        active_set = self.active_set_if_not_licq(active_set, facet_index, cr)
                        if active_set and active_set not in tested_active_sets:
                            print('    corrected active set ' + str(active_set))
                            cr_to_be_explored.append(CriticalRegion(active_set, self.qp))
                        else:
                            print('    unfeasible critical region detected')
        return [cr_to_be_explored, tested_active_sets]

    def active_set_if_not_licq(self, candidate_active_set, facet_index, cr, dist=1e-6, lambda_bound=1e6, toll=1e-6):
        """
        Returns the active set of a critical region in case that licq does not hold (Theorem 4 revisited)

        INPUTS:
            candidate_active_set: candidate active set for which LICQ doesn't hold
            facet_index: index of this active set hypothesis in the parent's list of neighboring active sets
            cr: critical region from which the new active set is generated

        OUTPUTS:
            active_set: real active set of the child critical region ([] if the region is unfeasible)
        """

        # differences between the active set of the parent and the candidate active set
        active_set_change = list(set(cr.active_set).symmetric_difference(set(candidate_active_set)))

        # if there is more than one change, nothing can be done...
        if len(active_set_change) > 1:
            print 'Cannot solve degeneracy with multiple active set changes! The solution of a QP is required...'
            active_set = self.solve_qp_beyond_facet(facet_index, cr)

        # if there is one change solve the lp from Theorem 4
        else:
            active_set = self.solve_lp_on_facet(candidate_active_set, facet_index, cr)

        return active_set

    def solve_qp_beyond_facet(self, facet_index, cr, dist=1e-8, toll=1.e-6):

        """
        Solves a QP a step of length "dist" beyond the facet wich index is "facet_index"
        to determine the active set in that region.

        INPUTS:
            facet_index: index of this active set hypothesis in the parent's list of neighboring active sets
            cr: critical region from which the new active set is generated

        OUTPUTS:
            active_set: real active set of the child critical region ([] if the region is unfeasible)
        """
        
        # relate step distance with polytope dimensions
        dist = min(cr.polytope.facet_radii(facet_index)/100., cr.polytope.radius/100., dist)

        # relates step length to polytope dimension
        dist = min(dist, cr.polytope.radius, cr.polytope.facet_radii(facet_index))

        # center of the facet in the parameter space
        x_center = cr.polytope.facet_centers(facet_index)

        # solve the QP inside the new critical region to derive the active set
        x_beyond = x_center + dist*cr.polytope.lhs_min[facet_index,:].reshape(x_center.shape)
        x_beyond = x_beyond.reshape(x_center.shape[0],1)
        z = quadratic_program(self.qp.H, np.zeros((self.qp.H.shape[0],1)), self.qp.G, self.qp.W + self.qp.S.dot(x_beyond))[0]

        # new active set for the child
        constraints_residuals = self.qp.G.dot(z) - self.qp.W - self.qp.S.dot(x_beyond)
        active_set = [i for i in range(0,self.qp.G.shape[0]) if constraints_residuals[i] > -toll]

        return active_set

    def solve_lp_on_facet(self, candidate_active_set, facet_index, cr, toll=1e-6):
        """
        Solves a LP on the center of the facet wich index is "facet_index" to determine
        the active set in that region (Theorem 4)

        INPUTS:
            candidate_active_set: candidate active set for which LICQ doesn't hold
            facet_index: index of this active set hypothesis in the parent's list of neighboring active sets
            cr: critical region from which the new active set is generated

        OUTPUTS:
            active_set: real active set of the child critical region ([] if the region is unfeasible)
        """

        # differences between the active set of the parent and the candidate active set
        active_set_change = list(set(cr.active_set).symmetric_difference(set(candidate_active_set)))

        # compute optimal solution in the center of the shared facet
        x_center = cr.polytope.facet_centers(facet_index)
        z_center = cr.z_optimal(x_center)

        # solve lp from Theorem 4
        G_A = self.qp.G[candidate_active_set,:]
        n_lam = G_A.shape[0]
        cost = np.zeros((n_lam,1))
        cost[candidate_active_set.index(active_set_change[0])] = -1.
        cons_lhs = np.vstack((G_A.T, -G_A.T, -np.eye(n_lam)))
        cons_rhs = np.vstack((-self.qp.H.dot(z_center), self.qp.H.dot(z_center), np.zeros((n_lam,1))))
        lambda_sol = linear_program(cost, cons_lhs, cons_rhs)[0]

        # if the solution in unbounded the region is unfeasible
        active_set = []
        if any(np.isnan(lambda_sol)):
            return active_set

        # if the solution in bounded look at the indices of the solution to derive the active set
        for i in range(0, n_lam):
            if lambda_sol[i] > toll:
                active_set += [candidate_active_set[i]]
        return active_set

    def feedforward_explicit(self, x0):

        # check that the explicit solution is available
        if self.critical_regions is None:
            raise ValueError('Explicit solution not computed yet! First run .compute_explicit_solution() ...')

        # find the CR to which the given state belongs
        cr_where_x0 = self.critical_regions.lookup(x0)

        # derive explicit solution
        if cr_where_x0 is not None:
            u_feedforward = cr_where_x0.u_offset + cr_where_x0.u_linear.dot(x0)

        # if unfeasible return nan
        else:
            #print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            u_feedforward = np.zeros((self.qp.G.shape[1], 1))
            u_feedforward[:] = np.nan

        return u_feedforward

    def feedback_explicit(self, x0):
        # select only the first control of the feedforward
        u_feedback = self.feedforward_explicit(x0)[0:self.n_u]
        return u_feedback

    def optimal_value_function(self, x0):

        # check that the explicit solution is available
        if self.critical_regions is None:
            raise ValueError('Explicit solution not computed yet! First run .compute_explicit_solution() ...')

        # find the CR to which the given state belongs
        cr_where_x0 = self.critical_regions.lookup(x0)

        # derive explicit solution
        if cr_where_x0 is not None:
            V = cr_where_x0.V_offset + cr_where_x0.V_linear.dot(x0) + .5*x0.T.dot(cr_where_x0.V_quadratic).dot(x0)
            V = V[0]
        else:
            V = np.nan

        return V

    def merge_critical_regions(self):
        self.u_offset_list = []
        self.u_linear_list = []
        self.cr_families = []
        for cr in self.critical_regions:
            cr_family = np.where(np.isclose(cr.u_offset[0], self.u_offset_list))[0]
            if cr_family and all(np.isclose(cr.u_linear[0,:], self.u_linear_list[cr_family[0]])):
                self.cr_families[cr_family[0]].append(cr)
            else:
                self.cr_families.append([cr])
                self.u_offset_list.append(cr.u_offset[0])
                self.u_linear_list.append(cr.u_linear[0,:])
        print 'Critical regions merged in ', str(len(self.cr_families)), ' sets.'
        return


    def plot_merged_state_partition(self):
        self.merge_critical_regions()

        first_input_index = np.where(np.isclose(self.qp.G[:,1:], 0.).all(axis=1))[0]
        asets = [[]]
        for n in range(1, len(first_input_index)+1):
            asets_n = itertools.combinations(first_input_index, n)
            asets += [list(c) for c in asets_n]

        colors = ['b','g','r','c','m','y']*len(self.cr_families)
        fig, ax = plt.subplots()
        for i, family in enumerate(self.cr_families):
            for cr in family:
                cr.polytope.plot(color=colors[i])

        #         circle = plt.Circle((cr.polytope.center[0], cr.polytope.center[1]), cr.polytope.radius)
        #         ax.add_artist(circle)
        #         plt.text(cr.polytope.center[0], cr.polytope.center[1], str(cr.active_set))
                # plt.text(cr.polytope.center[0], cr.polytope.center[1], str(cr.u_linear[0,:])+str(cr.u_offset[0]))
        #         for j in range(0, len(cr.polytope.minimal_facets)):
        #             plt.text(cr.polytope.facet_centers(j)[0], cr.polytope.facet_centers(j)[1], str(cr.polytope.minimal_facets[j]))
        for family in self.cr_families:
            for cr in family: 
                #if cr.active_set in asets:
                if cr.active_set in asets:
                    p = Polytope(cr.polytope.A[first_input_index,:], cr.polytope.b[first_input_index])
                    p.add_bounds(np.array([[.4],[1.]]), np.array([[-.4],[-1.]]))
                    p.assemble()
                    p.plot(color='black')

        return


    def plot_state_partition(self):
        colors = ['b','g','r','c','m','y']*len(self.critical_regions)
        plt.figure()
        for i, cr in enumerate(self.critical_regions):
            cr.polytope.plot(color=colors[i])
            #plt.savefig('/Users/tobia/Google Drive/UNIPI/PhD/box_atlas/presentation/figures/cr' + str(i+2) + '.pdf')
        return

    def plot_optimal_control_law(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        vertices = np.zeros((0,2))
        for cr in self.critical_regions:
            vertices = np.vstack((vertices, cr.polytope.vertices))
        x_max = max([vertex[0] for vertex in vertices])
        x_min = min([vertex[0] for vertex in vertices])
        y_max = max([vertex[1] for vertex in vertices])
        y_min = min([vertex[1] for vertex in vertices])
        x = np.arange(x_min, x_max, (x_max-x_min)/100.)
        y = np.arange(y_min, y_max, (y_max-y_min)/100.)
        X, Y = np.meshgrid(x, y)
        zs = np.array([self.feedback_explicit(np.array([[x],[y]])) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(r'$u^*_0$')
        return

    def plot_optimal_value_function(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        vertices = np.zeros((0,2))
        for cr in self.critical_regions:
            vertices = np.vstack((vertices, cr.polytope.vertices))
        x_max = max([vertex[0] for vertex in vertices])
        x_min = min([vertex[0] for vertex in vertices])
        y_max = max([vertex[1] for vertex in vertices])
        y_min = min([vertex[1] for vertex in vertices])
        x = np.arange(x_min, x_max, (x_max-x_min)/100.)
        y = np.arange(y_min, y_max, (y_max-y_min)/100.)
        X, Y = np.meshgrid(x, y)
        zs = np.array([self.optimal_value_function(np.array([[x],[y]])) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(r'$V^*$')
        return


        



    @staticmethod
    def licq_check(G, active_set, max_cond=1e9):
        """
        Checks if LICQ holds for the given active set

        INPUTS:
            G: gradient of the constraints
            active_set: active set
            max_cond: maximum condition number of the squared active constraints

        OUTPUTS:
            licq -> flag (True if licq holds, False if licq doesn't hold)
        """

        # select active constraints
        G_A = G[active_set,:]

        # check condion number of the squared active constraints
        licq = True
        cond = np.linalg.cond(G_A.dot(G_A.T))
        if cond > max_cond:
            licq = False

        return licq

    # def compute_approximated_explicit_solution(self):

    #     self.qp.find_feasible_region()

    #     first_block_indices = np.where(np.isclose(self.qp.G[:,1:], 0.).all(axis=1))[0]
    #     if first_block_indices != range(0, len(first_block_indices)):
    #         raise ValueError('Constraint matrix not in lower triangular form!')
        
    #     x_beyond_facet =
    #     first_block_active_set = 
    #     first_block_inactive_set = sorted(list(set(first_block_indices) - set(first_block_active_set)))
        
    #     # remove rows associated with incative contraints
    #     modified_qp = copy.copy(self.qp)
    #     modified_qp.G[first_block_inactive_set,:] = []
    #     modified_qp.W[first_block_inactive_set,:] = []
    #     modified_qp.S[first_block_inactive_set,:] = []

    #     # turn active inequaliteis into euqualities
    #     modified_qp.G = np.vstack((- selg.qp.G[first_block_active_set,:], modified_qp.G))
    #     modified_qp.W = np.vstack((- selg.qp.W[first_block_active_set,:], modified_qp.W))
    #     modified_qp.S = np.vstack((- selg.qp.S[first_block_active_set,:], modified_qp.S))

    #     # get active set
    #     f = x_beyond_facet.T.dot(modified_qp.F)
    #     b = modified_qp.S.dot(x_beyond_facet) + modified_qp.W
    #     z_star = quadratic_program(modified_qp.H, f, modified_qp.G, b)[0]
    #     active_set = ???
    #     inactive_set = sorted(list(set(range(0, qp.G.shape[0])) - set(active_set)))

    #     # multipliers explicit solution
    #     [G_A, W_A, S_A] = [qp.G[active_set,:], qp.W[active_set,:], qp.S[active_set,:]]
    #     [G_I, W_I, S_I] = [qp.G[inactive_set,:], qp.W[inactive_set,:], qp.S[inactive_set,:]]
    #     H_A = np.linalg.inv(G_A.dot(qp.H_inv.dot(G_A.T)))
    #     lambda_A_offset = - H_A.dot(W_A)
    #     lambda_A_linear = - H_A.dot(S_A)

    #     # primal variables explicit solution
    #     z_offset = - qp.H_inv.dot(G_A.T.dot(lambda_A_offset))
    #     z_linear = - qp.H_inv.dot(G_A.T.dot(lambda_A_linear))
        
    #     # first input primal boundaries
    #     fipb_lhs = self.qp.G[first_block_inactive_set,:].dot(z_linear) - self.qp.S[first_block_inactive_set,:]
    #     fipb_rhs = - self.qp.G[first_block_inactive_set,:].dot(z_offset) + self.qp.W[first_block_inactive_set,:]





class CriticalRegion:
    """
    Implements the algorithm from Tondel et al. "An algorithm for multi-parametric quadratic programming and explicit MPC solutions"

    VARIABLES:
        n_constraints: number of contraints in the qp
        n_parameters: number of parameters of the qp
        active_set: active set inside the critical region
        inactive_set: list of indices of non active contraints inside the critical region
        polytope: polytope describing the ceritical region in the parameter space
        weakly_active_constraints: list of indices of constraints that are weakly active iside the entire critical region
        candidate_active_sets: list of lists of active sets, its ith element collects the set of all the possible
            active sets that can be found crossing the ith minimal facet of the polyhedron
        z_linear: linear term in the piecewise affine primal solution z_opt = z_linear*x + z_offset
        z_offset: offset term in the piecewise affine primal solution z_opt = z_linear*x + z_offset
        u_linear: linear term in the piecewise affine primal solution u_opt = u_linear*x + u_offset
        u_offset: offset term in the piecewise affine primal solution u_opt = u_linear*x + u_offset
        lambda_A_linear: linear term in the piecewise affine dual solution (only active multipliers) lambda_A = lambda_A_linear*x + lambda_A_offset
        lambda_A_offset: offset term in the piecewise affine dual solution (only active multipliers) lambda_A = lambda_A_linear*x + lambda_A_offset
    """

    def __init__(self, active_set, qp):

        # store active set
        print 'Computing critical region for the active set ' + str(active_set)
        [self.n_constraints, self.n_parameters] = qp.S.shape
        self.active_set = active_set
        self.inactive_set = sorted(list(set(range(0, self.n_constraints)) - set(active_set)))

        # find the polytope
        self.polytope(qp)
        if self.polytope.empty:
            return

        # find candidate active sets for the neighboiring regions
        minimal_coincident_facets = [self.polytope.coincident_facets[i] for i in self.polytope.minimal_facets]
        self.candidate_active_sets = self.candidate_active_sets(active_set, minimal_coincident_facets)

        # find weakly active constraints
        self.find_weakly_active_constraints()

        # expand the candidates if there are weakly active constraints
        if self.weakly_active_constraints:
            self.candidate_active_set = self.expand_candidate_active_sets(self.candidate_active_set, self.weakly_active_constraints)

        return

    def polytope(self, qp):
        """
        Stores a polytope that describes the critical region in the parameter space.
        """

        # multipliers explicit solution
        [G_A, W_A, S_A] = [qp.G[self.active_set,:], qp.W[self.active_set,:], qp.S[self.active_set,:]]
        [G_I, W_I, S_I] = [qp.G[self.inactive_set,:], qp.W[self.inactive_set,:], qp.S[self.inactive_set,:]]
        H_A = np.linalg.inv(G_A.dot(qp.H_inv.dot(G_A.T)))
        self.lambda_A_offset = - H_A.dot(W_A)
        self.lambda_A_linear = - H_A.dot(S_A)

        # primal variables explicit solution
        self.z_offset = - qp.H_inv.dot(G_A.T.dot(self.lambda_A_offset))
        self.z_linear = - qp.H_inv.dot(G_A.T.dot(self.lambda_A_linear))

        # primal original variables explicit solution
        self.u_offset = self.z_offset
        self.u_linear = self.z_linear - np.linalg.inv(qp.H).dot(qp.F.T)

        # optimal value function explicit solution
        self.V_offset = .5*self.u_offset.T.dot(qp.H).dot(self.u_offset)
        self.V_linear = self.u_offset.T.dot(qp.H).dot(self.u_linear) + self.u_offset.T.dot(qp.F.T)
        self.V_quadratic = self.u_linear.T.dot(qp.H).dot(self.u_linear) + qp.Q + 2.*qp.F.dot(self.u_linear)

        # equation (12) (modified: only inactive indices considered)
        lhs_type_1 = G_I.dot(self.z_linear) - S_I
        rhs_type_1 = - G_I.dot(self.z_offset) + W_I

        # equation (13)
        lhs_type_2 = - self.lambda_A_linear
        rhs_type_2 = self.lambda_A_offset

        # gather facets of type 1 and 2 to define the polytope (note the order: the ith facet of the cr is generated by the ith constraint)
        lhs = np.zeros((self.n_constraints, self.n_parameters))
        rhs = np.zeros((self.n_constraints, 1))
        lhs[self.inactive_set + self.active_set, :] = np.vstack((lhs_type_1, lhs_type_2))
        rhs[self.inactive_set + self.active_set] = np.vstack((rhs_type_1, rhs_type_2))

        # construct polytope
        self.polytope = Polytope(lhs, rhs)
        self.polytope.assemble()

        return

    def find_weakly_active_constraints(self, toll=1e-8):
        """
        Stores the list of constraints that are weakly active in the whole critical region
        enumerated in the as in the equation G z <= W + S x ("original enumeration")
        (by convention weakly active constraints are included among the active set,
        so that only constraints of type 2 are anlyzed)
        """

        # equation (13), again...
        lhs_type_2 = - self.lambda_A_linear
        rhs_type_2 = self.lambda_A_offset

        # weakly active constraints are included in the active set
        self.weakly_active_constraints = []
        for i in range(0, len(self.active_set)):

            # to be weakly active in the whole region they can only be in the form 0^T x <= 0
            if np.linalg.norm(lhs_type_2[i,:]) + np.absolute(rhs_type_2[i,:]) < toll:
                print('Weakly active constraint detected!')
                self.weakly_active_constraints.append(self.active_set[i])

        return

    @staticmethod
    def candidate_active_sets(active_set, minimal_coincident_facets):
        """
        Computes one candidate active set for each non-redundant facet of a critical region
        (Theorem 2 and Corollary 1).

        INPUTS:
        active_set: active set of the parent critical region
        minimal_coincident_facets: list of facets coincident to the minimal facets
            (i.e.: [coincident_facets[i] for i in minimal_facets])

        OUTPUTS:
            candidate_active_sets: list of the candidate active sets for each minimal facet
        """

        # initialize list of condidate active sets
        candidate_active_sets = []

        # cross each non-redundant facet of the parent CR
        for coincident_facets in minimal_coincident_facets:

            # add or remove each constraint crossed to the active set of the parent CR
            candidate_active_set = set(active_set).symmetric_difference(set(coincident_facets))
            candidate_active_sets.append([sorted(list(candidate_active_set))])

        return candidate_active_sets

    @staticmethod
    def expand_candidate_active_sets(candidate_active_sets, weakly_active_constraints):
        """
        Expands the candidate active sets if there are some weakly active contraints (Theorem 5).

        INPUTS:
            candidate_active_sets: list of the candidate active sets for each minimal facet
            weakly_active_constraints: list of weakly active constraints (in the "original enumeration")

        OUTPUTS:
            candidate_active_sets: list of the candidate active sets for each minimal facet
        """

        # determine every possible combination of the weakly active contraints
        wac_combinations = []
        for n in range(1, len(weakly_active_constraints)+1):
            wac_combinations_n = itertools.combinations(weakly_active_constraints, n)
            wac_combinations += [list(c) for c in wac_combinations_n]

        # for each minimal facet of the CR add or remove each combination of wakly active constraints
        for i in range(0, len(candidate_active_sets)):
            active_set = candidate_active_sets[i][0]
            for combination in wac_combinations:
                further_active_set = set(active_set).symmetric_difference(combination)
                candidate_active_sets[i].append(sorted(list(further_active_set)))

        return candidate_active_sets

    def z_optimal(self, x):
        """
        Returns the explicit solution of the mpQP as a function of the parameter.

        INPUTS:
            x: value of the parameter

        OUTPUTS:
            z_optimal: solution of the QP
        """

        z_optimal = self.z_offset + self.z_linear.dot(x).reshape(self.z_offset.shape)
        return z_optimal

    def lambda_optimal(self, x):
        """
        Returns the explicit value of the multipliers of the mpQP as a function of the parameter.

        INPUTS:
            x: value of the parameter

        OUTPUTS:
            lambda_optimal: optimal multipliers
        """

        lambda_A_optimal = self.lambda_A_offset + self.lambda_A_linear.dot(x)
        lambda_optimal = np.zeros(len(self.active_set + self.inactive_set))
        for i in range(0, len(self.active_set)):
            lambda_optimal[self.active_set[i]] = lambda_A_optimal[i]
        return lambda_optimal

    def applies_to(self, x):
        """
        Determines is a given point belongs to the critical region.

        INPUTS:
            x: value of the parameter

        OUTPUTS:
            is_inside: flag (True if x is in the CR, False otherwise)
        """

        # check if x is inside the polytope
        is_inside = self.polytope.applies_to(x)

        return is_inside


class ApproximatedCriticalRegion:

    def __init__(self, active_set, qp, feasible_region):

        # store active set
        print 'Computing approximated critical region for the active set ' + str(active_set)
        [self.n_constraints, self.n_parameters] = qp.S.shape
        self.active_set = active_set
        self.inactive_set = sorted(list(set(range(0, self.n_constraints)) - set(active_set)))

        # find the polytope
        self.polytope(qp, feasible_region)
        if self.polytope.empty:
            return


    def polytope(self, qp, feasible_region):

        # multipliers explicit solution
        [G_A, W_A, S_A] = [qp.G[self.active_set,:], qp.W[self.active_set,:], qp.S[self.active_set,:]]
        [G_I, W_I, S_I] = [qp.G[self.inactive_set,:], qp.W[self.inactive_set,:], qp.S[self.inactive_set,:]]
        H_A = np.linalg.inv(G_A.dot(qp.H_inv.dot(G_A.T)))
        self.lambda_A_offset = - H_A.dot(W_A)
        self.lambda_A_linear = - H_A.dot(S_A)

        # primal variables explicit solution
        self.z_offset = - qp.H_inv.dot(G_A.T.dot(self.lambda_A_offset))
        self.z_linear = - qp.H_inv.dot(G_A.T.dot(self.lambda_A_linear))

        # primal original variables explicit solution
        self.u_offset = self.z_offset
        self.u_linear = self.z_linear - np.linalg.inv(qp.H).dot(qp.F.T)

        # first input primal boundaries
        lhs_type_1 = G_I.dot(self.z_linear) - S_I
        rhs_type_1 = - G_I.dot(self.z_offset) + W_I

        # construct polytope
        self.polytope = Polytope(lhs, rhs)
        self.polytope.assemble()

        return