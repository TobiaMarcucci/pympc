import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from optimization.gurobi import quadratic_program
from geometry import Polytope
from mpcqp import CanonicalMPCQP
from optimization.mpqpsolver import MPQPSolver

class MPCController:
    """
    VARIABLES:
        sys:
        N:
        Q:
        R:
        P:
        X:
        U:
        X_N:
        H:
        F:
        Y:
        G:
        W:
        E:
    """
    def __init__(self, sys, N, Q, R, P=None, X=None, U=None, X_N=None):
        self.sys = sys
        self.N = N
        self.Q = Q
        self.R = R
        if P is None:
            self.P = Q
        else:
            self.P = P
        self.X = X
        self.U = U
        self.X_N = X_N
        self.critical_regions = None
        self.constraint_blocks()
        self.cost_blocks()
        self.canonical_qp = CanonicalMPCQP(self.H, self.F, self.Y, self.G, self.W, self.E)
        return

    def constraint_blocks(self):
        # compute each constraint
        [G_u, W_u, E_u] = self.input_constraint_blocks()
        [G_x, W_x, E_x] = self.state_constraint_blocks()
        [G_xN, W_xN, E_xN] = self.terminal_constraint_blocks()
        # gather constraints
        self.G = np.vstack((G_u, G_x, G_xN))
        self.W = np.vstack((W_u, W_x, W_xN))
        self.E = np.vstack((E_u, E_x, E_xN))
        # permutation of the rows to put G in lower triangular form
        self.constraint_permutation()
        # remove redundant constraints
        constraint_polytope = Polytope(np.hstack((self.G, -self.E)), self.W)
        constraint_polytope.assemble()
        self.G = constraint_polytope.lhs_min[:,:self.sys.n_u*self.N]
        self.E = - constraint_polytope.lhs_min[:,self.sys.n_u*self.N:]
        self.W = constraint_polytope.rhs_min
        return

    def input_constraint_blocks(self):
        if self.U is None:
            G_u = np.array([]).reshape((0, self.sys.n_u*self.N))
            W_u = np.array([]).reshape((0, 1))
            E_u = np.array([]).reshape((0, self.sys.n_x))
        else:
            G_u = linalg.block_diag(*[self.U.lhs_min for i in range(0, self.N)])
            W_u = np.vstack([self.U.rhs_min for i in range(0, self.N)])
            E_u = np.zeros((W_u.shape[0], self.sys.n_x))
        return [G_u, W_u, E_u]

    def state_constraint_blocks(self):
        if self.X is None:
            G_x = np.array([]).reshape((0, self.sys.n_u*self.N))
            W_x = np.array([]).reshape((0, 1))
            E_x = np.array([]).reshape((0, self.sys.n_x))
        else:
            [free_evolution, forced_evolution] = self.sys.evolution_matrices(self.N)
            lhs_x_diag = linalg.block_diag(*[self.X.lhs_min for i in range(0, self.N)])
            G_x = lhs_x_diag.dot(forced_evolution)
            W_x = np.vstack([self.X.rhs_min for i in range(0, self.N)])
            E_x = - lhs_x_diag.dot(free_evolution)
        return [G_x, W_x, E_x]

    def terminal_constraint_blocks(self):
        if self.X_N is None:
            G_xN = np.array([]).reshape((0, self.sys.n_u*self.N))
            W_xN = np.array([]).reshape((0, 1))
            E_xN = np.array([]).reshape((0, self.sys.n_x))
        else:
            forced_evolution = self.sys.evolution_matrices(self.N)[1]
            G_xN = self.X_N.lhs_min.dot(forced_evolution[-self.sys.n_x:,:])
            W_xN = self.X_N.rhs_min
            E_xN = - self.X_N.lhs_min.dot(np.linalg.matrix_power(self.sys.A, self.N))
        return [G_xN, W_xN, E_xN]

    def constraint_permutation(self):
        """
        puts the constraint gradient G in the lower block triangular form.
        """
        n_u = len(self.U.minimal_facets)
        n_x = len(self.X.minimal_facets)
        row_permutation = []
        for i in range(0, self.N):
            row_permutation += range(i*n_u, (i+1)*n_u) + range(self.N*n_u + i*n_x, self.N*n_u + (i+1)*n_x)
        self.G[0:len(row_permutation)] = self.G[row_permutation]
        self.W[0:len(row_permutation)] = self.W[row_permutation]
        self.E[0:len(row_permutation)] = self.E[row_permutation]
        return

    def cost_blocks(self):
        # quadratic term in the state sequence
        if self.N == 1:
            H_x = np.zeros((0, 0))
        else:
            H_x = linalg.block_diag(*[self.Q for i in range(0, self.N-1)])
        H_x = linalg.block_diag(H_x, self.P)
        # quadratic term in the input sequence
        H_u = linalg.block_diag(*[self.R for i in range(0, self.N)])
        # evolution of the system
        [free_evolution, forced_evolution] = self.sys.evolution_matrices(self.N)
        # quadratic term
        self.H = 2*(H_u+forced_evolution.T.dot(H_x.dot(forced_evolution)))
        # linear term
        F = 2*forced_evolution.T.dot(H_x.T).dot(free_evolution)
        self.F = F.T
        # quadratic term in the initial state
        self.Y = 2.*(self.Q + free_evolution.T.dot(H_x.T).dot(free_evolution))
        return

    def feedforward(self, x0):
        u_feedforward = quadratic_program(self.H, (x0.T.dot(self.F)).T, self.G, self.W + self.E.dot(x0))[0]
        if any(np.isnan(u_feedforward).flatten()):
            print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
        return u_feedforward

    def feedback(self, x0):
        u_feedback = self.feedforward(x0)[0:self.sys.n_u]
        return u_feedback

    def compute_explicit_solution(self):
        mpqp_solution = MPQPSolver(self.canonical_qp)
        self.critical_regions = mpqp_solution.critical_regions
        return

    def feedforward_explicit(self, x0):

        # check that the explicit solution is available
        if self.critical_regions is None:
            raise ValueError('Explicit solution not computed yet! First run .compute_explicit_solution().')

        # find the CR to which the given state belongs
        cr_where_x0 = self.critical_regions.lookup(x0)

        # derive explicit solution
        if cr_where_x0 is not None:
            u_feedforward = cr_where_x0.u_offset + cr_where_x0.u_linear.dot(x0)

        # if unfeasible return nan
        else:
            print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            u_feedforward = np.zeros((self.G.shape[1], 1))
            u_feedforward[:] = np.nan

        return u_feedforward

    def feedback_explicit(self, x0):

        # select only the first control of the feedforward
        u_feedback = self.feedforward_explicit(x0)[0:self.sys.n_u]

        return u_feedback

    def optimal_value_function(self, x0):

        # check that the explicit solution is available
        if self.critical_regions is None:
            raise ValueError('Explicit solution not computed yet! First run .compute_explicit_solution().')

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

    def plot_state_partition(self, active_set=False, facet_index=False, **kwargs):

        if self.critical_regions is None:
            raise ValueError('Explicit solution not computed yet! First run .compute_explicit_solution().')

        fig, ax = plt.subplots()
        x_min, x_max, y_min, y_max = [0.]*4
        for cr in self.critical_regions:
            cr.polytope.plot(facecolor=np.random.rand(3,1), **kwargs)
            x_min = min(x_min, ax.get_xlim()[0])
            x_max = max(x_max, ax.get_xlim()[1])
            y_min = min(y_min, ax.get_ylim()[0])
            y_max = max(y_max, ax.get_ylim()[1])
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            if active_set:
                plt.text(cr.polytope.center[0], cr.polytope.center[1], str(cr.active_set))
            if facet_index:
                for j in range(0, len(cr.polytope.minimal_facets)):
                    plt.text(cr.polytope.facet_centers(j)[0], cr.polytope.facet_centers(j)[1], str(cr.polytope.minimal_facets[j]))
        return

    def plot_merged_state_partition(self, active_set=False, first_input=False, facet_index=False, **kwargs):
        self.merge_critical_regions()
        fig, ax = plt.subplots()
        x_min, x_max, y_min, y_max = [0.]*4
        for i, family in enumerate(self.cr_families):
            color = np.random.rand(3,1)
            for cr in family:
                cr.polytope.plot(facecolor=color, **kwargs)
                x_min = min(x_min, ax.get_xlim()[0])
                x_max = max(x_max, ax.get_xlim()[1])
                y_min = min(y_min, ax.get_ylim()[0])
                y_max = max(y_max, ax.get_ylim()[1])
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                if active_set:
                    plt.text(cr.polytope.center[0], cr.polytope.center[1], str(cr.active_set))
                if first_input:
                    plt.text(cr.polytope.center[0], cr.polytope.center[1], str(cr.u_linear[0,:])+str(cr.u_offset[0]))
                if facet_index:
                    for j in range(0, len(cr.polytope.minimal_facets)):
                        plt.text(cr.polytope.facet_centers(j)[0], cr.polytope.facet_centers(j)[1], str(cr.polytope.minimal_facets[j]))

    def plot_optimal_value_function(self):
        if self.critical_regions is None:
            raise ValueError('Explicit solution not computed yet! First run .compute_explicit_solution().')
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
        cp = plt.contour(X, Y, Z)
        plt.colorbar(cp)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title(r'$V^*(x)$')
        return
