import time
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from optimization.gurobi import quadratic_program
from geometry import Polytope
from mpcqp import CanonicalMPCQP
from optimization.mpqpsolver import MPQPSolver
import gurobipy as grb

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
        for cr in self.critical_regions:
            cr.polytope.plot(facecolor=np.random.rand(3,1), **kwargs)
            ax.autoscale_view()
            if active_set:
                plt.text(cr.polytope.center[0], cr.polytope.center[1], str(cr.active_set))
            if facet_index:
                for j in range(0, len(cr.polytope.minimal_facets)):
                    plt.text(cr.polytope.facet_centers(j)[0], cr.polytope.facet_centers(j)[1], str(cr.polytope.minimal_facets[j]))
        return

    def plot_merged_state_partition(self, active_set=False, first_input=False, facet_index=False, **kwargs):
        self.merge_critical_regions()
        fig, ax = plt.subplots()
        for i, family in enumerate(self.cr_families):
            color = np.random.rand(3,1)
            for cr in family:
                cr.polytope.plot(facecolor=color, **kwargs)
                ax.autoscale_view()
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







class MPCHybridController:
    """
    """
    def __init__(self, sys, N, Q, R, P=None, X_N=None):
        self.sys = sys
        self.N = N
        self.Q = Q
        self.R = R
        if P is None:
            self.P = Q
        else:
            self.P = P
        self.X_N = X_N
        return

    def feedforward(self, x0):

        # gurobi model
        model = grb.Model()

        # variables
        lb_x = [[-grb.GRB.INFINITY]*(self.N+1)]*self.sys.n_x
        lb_u = [[-grb.GRB.INFINITY]*self.N]*self.sys.n_u
        lb_z = [[[-grb.GRB.INFINITY]*self.N]*self.sys.n_sys]*self.sys.n_x
        x = model.addVars(self.N+1, self.sys.n_x, lb=lb_x, name='x')
        u = model.addVars(self.N, self.sys.n_u, lb=lb_u, name='u')
        z = model.addVars(self.N, self.sys.n_sys, self.sys.n_x, lb=lb_z, name='z')
        d = model.addVars(self.N, self.sys.n_sys, vtype=grb.GRB.BINARY, name='d')
        model.update()

        # numpy variables (list of numpy matrices ordered in time)
        x_np = [np.array([[x[i,j]] for j in range(self.sys.n_x)]) for i in range(self.N+1)]
        u_np = [np.array([[u[i,j]] for j in range(self.sys.n_u)]) for i in range(self.N)]

        # set objective
        V = 0.
        for i in range(self.N):
            V += .5*(x_np[i].T.dot(self.Q).dot(x_np[i]) + u_np[i].T.dot(self.R).dot(u_np[i]))
        V += .5*x_np[self.N].T.dot(self.P).dot(x_np[self.N])
        model.setObjective(V[0,0])

        # initial condition
        model.addConstrs((x[0,i] == x0[i,0] for i in range(self.sys.n_x)))

        # disjuction
        for i in range(self.N):
            model.addConstr(np.sum([d[i,j] for j in range(self.sys.n_sys)]) == 1.)

        # domains
        for i in range(self.N):
            for j in range(self.sys.n_sys):
                expr_x = self.sys.X_list[j].lhs_min.dot(x_np[i]) - self.sys.X_list[j].rhs_min - self.sys.big_M_domains*(1. - d[i,j])
                expr_u = self.sys.U_list[j].lhs_min.dot(u_np[i]) - self.sys.U_list[j].rhs_min - self.sys.big_M_domains*(1. - d[i,j])
                model.addConstrs((expr_x[k,0] <= 0. for k in range(len(self.sys.X_list[j].minimal_facets))))
                model.addConstrs((expr_u[k,0] <= 0. for k in range(len(self.sys.U_list[j].minimal_facets))))

        # state transition
        for i in range(self.N):
            for j in range(self.sys.n_x):
                expr = 0.
                for k in range(self.sys.n_sys):
                    expr += z[i,k,j]
                model.addConstr(x[i+1,j] == expr)
        
        # relaxation of the dynamics
        for i in range(self.N):
            for j in range(self.sys.n_sys):
                expr = self.sys.affine_systems[j].A.dot(x_np[i]) + self.sys.affine_systems[j].B.dot(u_np[i]) + self.sys.affine_systems[j].c
                for k in range(self.sys.n_x):
                    model.addConstr(z[i,j,k] >= self.sys.small_m_dynamics*d[i,j])
                    model.addConstr(z[i,j,k] <= self.sys.big_M_dynamics*d[i,j])
                    model.addConstr(z[i,j,k] >= expr[k,0] - self.sys.big_M_dynamics*(1 - d[i,j]))
                    model.addConstr(z[i,j,k] <= expr[k,0] - self.sys.small_m_dynamics*(1 - d[i,j]))

        # terminal constraint
        if self.X_N is not None:
            expr = self.X_N.lhs_min.dot(x_np[self.N]) - self.X_N.rhs_min
            model.addConstrs((expr[i,0] <= 0. for i in range(len(self.X_N.minimal_facets))))

        # run optimization
        model.setParam('OutputFlag', False)
        model.optimize()

        # return active set
        if model.status != grb.GRB.Status.OPTIMAL:
            print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            u_feedforward = [np.full((self.sys.n_u,1), np.nan) for i in range(self.N)]
        else:
            u_feedforward = [np.array([[model.getAttr('x', u)[i,j]] for j in range(self.sys.n_u)]) for i in range(self.N)]
        return u_feedforward

    def feedback(self, x0):
        u_feedforward = self.feedforward(x0)
        u_feedback = u_feedforward[0]
        return u_feedback

    def backward_reachability_analysis(self, switching_sequence):
        tic = time.time()
        if self.X_N is None:
            print('A terminal constraint is needed for the backward reachability analysis!')
            return
        N = len(switching_sequence)
        X = self.X_N

        A_sequence = [self.sys.affine_systems[switch].A for switch in switching_sequence]
        B_sequence = [self.sys.affine_systems[switch].B for switch in switching_sequence]
        c_sequence = [self.sys.affine_systems[switch].c for switch in switching_sequence]

        U_sequence = [self.sys.U_list[switch] for switch in switching_sequence]
        X_sequence = [self.sys.X_list[switch] for switch in switching_sequence]

        for i in range(N-1,-1,-1):
            lhs_x = X.lhs_min.dot(A_sequence[i])
            lhs_u = X.lhs_min.dot(B_sequence[i])
            lhs = np.hstack((lhs_x, lhs_u))
            rhs = X.rhs_min - X.lhs_min.dot(c_sequence[i])
            feasible_set = Polytope(lhs, rhs)
            lhs = linalg.block_diag(X_sequence[i].lhs_min, U_sequence[i].lhs_min)
            rhs = np.vstack((X_sequence[i].rhs_min, U_sequence[i].rhs_min))
            feasible_set.add_facets(lhs, rhs)
            feasible_set.assemble()
            X = feasible_set.orthogonal_projection(range(self.sys.n_x))
        toc = time.time()
        print('Feasible set computed in ' + str(toc-tic) + ' s')
        return X

    def plot_feasible_set(self, switching_sequence, **kwargs):
        X = self.backward_reachability_analysis(switching_sequence)
        X.plot(**kwargs)
        plt.text(X.center[0], X.center[1], str(switching_sequence))
        return



    @staticmethod
    def condense_qp(switching_sequence, sys, Q, R, P, X_N):
        H, F, M, Y, D, L = MPCHybridController.condense_cost(switching_sequence, sys, Q, R, P)
        G, W, E = MPCHybridController.condense_constraints(switching_sequence, sys, X_N)
        if None in [G, W, E]:
            mpqp = None
        else:
            mpqp = multiparametric_qp(H, Y, F, M, D, L, G, E, W)

        return mpqp

    @staticmethod
    def condense_cost(switching_sequence, sys, Q, R, P):
        """
        Puts the cost function in the form
        .5 u'Hu + x0'Fu + Mu + .5x0'Yx0 + Dx0 + L
        """

        # cost function
        N = len(switching_sequence)
        if N == 1:
            H_x = np.zeros((0, 0))
        else:
            H_x = linalg.block_diag(*[Q for i in range(N-1)])
        H_x = linalg.block_diag(H_x, P)

        # quadratic term in the input sequence
        H_u = linalg.block_diag(*[R for i in range(N)])

        # evolution of the system
        free_evolution, forced_evolution, offset_evolution = sys.evolution_matrices(switching_sequence)

        # quadratic term
        H = 2*(H_u+forced_evolution.T.dot(H_x.dot(forced_evolution)))

        # linear term
        F = 2*forced_evolution.T.dot(H_x.T).dot(free_evolution)
        F = F.T

        # quadratic term in the initial state
        Y = 2.*(Q + free_evolution.T.dot(H_x.T).dot(free_evolution))

        # offset terms
        M = 2.*offset_evolution.T.dot(H_x).dot(forced_evolution)
        D = 2.*offset_evolution.T.dot(H_x).dot(free_evolution)
        L = offset_evolution.T.dot(H_x).dot(offset_evolution)

        return H, F, M, Y, D, L

    @staticmethod
    def condense_constraints(switching_sequence, sys, X_N=None):

        N = len(switching_sequence)
        
        # compute each constraint
        G_u, W_u, E_u = MPCHybridController.condense_input_constraints(switching_sequence, sys)
        G_x, W_x, E_x = MPCHybridController.condense_state_constraints(switching_sequence, sys)
        G_xN, W_xN, E_xN = MPCHybridController.condense_terminal_constraints(switching_sequence, sys, X_N)
        # gather constraints
        G = np.vstack((G_u, G_x, G_xN))
        W = np.vstack((W_u, W_x, W_xN))
        E = np.vstack((E_u, E_x, E_xN))
        # remove redundant constraints
        constraint_polytope = Polytope(np.hstack((G, -E)), W)
        constraint_polytope.assemble()
        if not constraint_polytope.empty:
            G = constraint_polytope.lhs_min[:,:sys.n_u*N]
            E = - constraint_polytope.lhs_min[:,sys.n_u*N:]
            W = constraint_polytope.rhs_min
        else:
            G = None
            W = None
            E = None

        return G, W, E

    @staticmethod
    def condense_input_constraints(switching_sequence, sys):

        N = len(switching_sequence)
        U_sequence = [sys.U_list[switching_sequence[i]] for i in range(N)]

        G_u = linalg.block_diag(*[U.lhs_min for U in U_sequence])
        W_u = np.vstack([U.rhs_min for U in U_sequence])
        E_u = np.zeros((W_u.shape[0], sys.n_x))

        return G_u, W_u, E_u

    @staticmethod
    def condense_state_constraints(switching_sequence, sys):

        N = len(switching_sequence)
        # note that we cannot constraint x0, so we start the loop from 1...
        X_sequence = [sys.X_list[switching_sequence[i]] for i in range(1, N)]

        free_evolution, forced_evolution, offset_evolution = sys.evolution_matrices(switching_sequence)
        lhs_x_diag = linalg.block_diag(*[X.lhs_min for X in X_sequence])

        G_x = lhs_x_diag.dot(forced_evolution[:-sys.n_x,:])
        W_x = np.vstack([X.rhs_min for X in X_sequence])
        W_x -= lhs_x_diag.dot(offset_evolution[:-sys.n_x,:])
        E_x = - lhs_x_diag.dot(free_evolution[:-sys.n_x,:])

        return G_x, W_x, E_x

    @staticmethod
    def condense_terminal_constraints(switching_sequence, sys, X_N):

        N = len(switching_sequence)

        if X_N is None:
            G_xN = np.zeros((0, sys.n_u*N))
            W_xN = np.zeros((0, 1))
            E_xN = np.zeros((0, sys.n_x))

        else:

            free_evolution, forced_evolution, offset_evolution = sys.evolution_matrices(switching_sequence)
            G_xN = X_N.lhs_min.dot(forced_evolution[-sys.n_x:,:])
            W_xN = X_N.rhs_min - X_N.lhs_min.dot(offset_evolution[-sys.n_x:,:])
            E_xN = - X_N.lhs_min.dot(free_evolution[-sys.n_x:,:])

        return G_xN, W_xN, E_xN
 





    # def plot_feasible_set(self, switching_sequence, **kwargs):
    #     tic = time.time()
    #     mpqp = self.condense_qp(switching_sequence, self.sys, self.Q, self.R, self.P, self.X_N)
    #     if mpqp is not None:
    #         X = self.sys.X_list[switching_sequence[0]]
    #         lhs = np.vstack((mpqp.feasible_set.lhs_min, X.lhs_min))
    #         rhs = np.vstack((mpqp.feasible_set.rhs_min, X.rhs_min))
    #         p = Polytope(lhs,rhs)
    #         p.assemble()
    #         toc = time.time()
    #         print('Feasible set computed in ' + str(toc-tic) + ' s')
    #         p.plot(**kwargs)
    #         plt.text(p.center[0], p.center[1], str(switching_sequence))
    #     else:
    #         print('Unfeasible switching sequence!')
    #     return

    




class multiparametric_qp:
    """
    """

    def __init__(self, F_uu, F_xx, F_xu, F_u, F_x, F, C_u, C_x, C):
        """
        QP in the form:
        min  .5 u' F_uu u + .5x' F_xx x  + x' F_xu u + F_u u + F_x x + F
        s.t. C_u u <= C_x x + C
        """

        self.F_uu = F_uu
        self.F_xx = F_xx
        self.F_xu = F_xu
        self.F_u = F_u
        self.F_x = F_x
        self.F = F

        self.C_u = C_u
        self.C_x = C_x
        self.C = C

        self.remove_linear_terms()

        self._feasible_set = None

        return

    def remove_linear_terms(self):
        """
        Applies the change of variables z = u + F_uu^-1 (F_xu' x + F_u')
        """

        F_uu_inv = np.linalg.inv(self.F_uu)

        self.F_uu_quad = self.F_uu
        self.F_xx_quad = (self.F_xx - self.F_xu.dot(F_uu_inv).dot(self.F_xu.T))
        self.F_x_quad = self.F_x - self.F_u.dot(F_uu_inv).dot(self.F_xu.T)
        self.F_quad = self.F - .5*self.F_u.dot(F_uu_inv).dot(self.

            F_u.T)

        self.C_u_quad = self.C_u
        self.C_x_quad = self.C_x + self.C_u.dot(F_uu_inv).dot(self.F_xu.T)
        self.C_quad = self.C + self.C_u.dot(F_uu_inv).dot(self.F_u.T)

        return

    @property
    def feasible_set(self):
        if self._feasible_set is None:
            augmented_polytope = Polytope(np.hstack((- self.C_x, self.C_u)), self.C)
            augmented_polytope.assemble()
            self._feasible_set = augmented_polytope.orthogonal_projection(range(self.C_x.shape[1]))
        return self._feasible_set




















