import time
import sys, os
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import gurobipy as grb
from contextlib import contextmanager
from optimization.pnnls import linear_program
from optimization.gurobi import quadratic_program, real_variable
from geometry.polytope import Polytope
from dynamical_systems import DTAffineSystem, DTPWASystem
from optimization.mpqpsolver import MPQPSolver, CriticalRegion
# from scipy.spatial import ConvexHull
from pympc.geometry.convex_hull import ConvexHull
import cdd
from pympc.geometry.convex_hull import PolytopeProjectionInnerApproximation




class MPCController:

    def __init__(self, sys, N, objective_norm, Q, R, P=None, X=None, U=None, X_N=None):
        self.sys = sys
        self.N = N
        self.objective_norm = objective_norm
        self.Q = Q
        self.R = R
        if P is None:
            self.P = Q
        else:
            self.P = P
        self.X = X
        self.U = U
        if X_N is None and X is not None:
            self.X_N = X
        else:
            self.X_N = X_N
        self.condense_program()
        self.critical_regions = None
        return

    def condense_program(self):
        c = np.zeros((self.sys.n_x, 1))
        a_sys = DTAffineSystem(self.sys.A, self.sys.B, c)
        sys_list = [a_sys]*self.N
        X_list = [self.X]*self.N
        U_list = [self.U]*self.N
        switching_sequence = [0]*self.N
        pwa_sys = DTPWASystem.from_orthogonal_domains(sys_list, X_list, U_list)
        self.condensed_program = OCP_condenser(pwa_sys, self.objective_norm, self.Q, self.R, self.P, self.X_N, switching_sequence)
        self.remove_intial_state_contraints()
        return

    def remove_intial_state_contraints(self, tol=1e-10):
        C_u_rows_norm = list(np.linalg.norm(self.condensed_program.C_u, axis=1))
        intial_state_contraints = [i for i, row_norm in enumerate(C_u_rows_norm) if row_norm < tol]
        if len(intial_state_contraints) > self.X.lhs_min.shape[0]:
            raise ValueError('Wrong number of zero rows in the constrinats')
        self.condensed_program.C_u = np.delete(self.condensed_program.C_u,intial_state_contraints, 0)
        self.condensed_program.C_x = np.delete(self.condensed_program.C_x,intial_state_contraints, 0)
        self.condensed_program.C = np.delete(self.condensed_program.C,intial_state_contraints, 0)
        return

    def feedforward(self, x0):
        u_feedforward, cost = self.condensed_program.solve(x0)
        u_feedforward = [u_feedforward[self.sys.n_u*i:self.sys.n_u*(i+1),:] for i in range(self.N)]
        # if any(np.isnan(u_feedforward).flatten()):
        #     print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
        return u_feedforward, cost

    def feedback(self, x0):
        return self.feedforward(x0)[0][0]

    def get_explicit_solution(self):
        """
        Attention: since the method remove_intial_state_contraints() modifies the variables of condensed_program, I have to call remove_linear_terms() again!
        """
        self.condensed_program.remove_linear_terms()
        mpqp_solution = MPQPSolver(self.condensed_program)
        self.critical_regions = mpqp_solution.critical_regions
        return

    def feedforward_explicit(self, x0):
        if self.critical_regions is None:
            raise ValueError('Explicit solution not available, call .get_explicit_solution()')
        cr_x0 = self.critical_regions.lookup(x0)
        if cr_x0 is not None:
            u_feedforward = cr_x0.u_offset + cr_x0.u_linear.dot(x0)
            u_feedforward = [u_feedforward[self.sys.n_u*i:self.sys.n_u*(i+1),:] for i in range(self.N)]
            cost = .5*x0.T.dot(cr_x0.V_quadratic).dot(x0) + cr_x0.V_linear.dot(x0) + cr_x0.V_offset
            cost = cost[0,0]
        else:
            # print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            u_feedforward = [np.full((self.sys.n_u, 1), np.nan) for i in range(self.N)]
            cost = np.nan
        return u_feedforward, cost

    def feedback_explicit(self, x0):
            return self.feedforward(x0)[0][0]

    def optimal_value_function(self, x0):
        if self.critical_regions is None:
            raise ValueError('Explicit solution not available, call .get_explicit_solution()')
        cr_x0 = self.critical_regions.lookup(x0)
        if cr_x0 is not None:
            cost = .5*x0.T.dot(cr_x0.V_quadratic).dot(x0) + cr_x0.V_linear.dot(x0) + cr_x0.V_offset
        else:
            #print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            cost = np.nan
        return cost

    # def group_critical_regions(self):
    #     self.u_offset_list = []
    #     self.u_linear_list = []
    #     self.cr_families = []
    #     for cr in self.critical_regions:
    #         cr_family = np.where(np.isclose(cr.u_offset[0], self.u_offset_list))[0]
    #         if cr_family and all(np.isclose(cr.u_linear[0,:], self.u_linear_list[cr_family[0]])):
    #             self.cr_families[cr_family[0]].append(cr)
    #         else:
    #             self.cr_families.append([cr])
    #             self.u_offset_list.append(cr.u_offset[0])
    #             self.u_linear_list.append(cr.u_linear[0,:])
    #     print 'Critical regions grouped in ', str(len(self.cr_families)), ' sets.'
    #     return



class MPCHybridController:

    def __init__(self, sys, N, objective_norm, Q, R, P, X_N):
        self.sys = sys
        self.N = N
        self.objective_norm = objective_norm
        self.Q = Q
        self.R = R
        self.P = P
        self.X_N = X_N
        self.compute_M_domains()
        self.compute_M_dynamics()
        return

    def compute_M_domains(self):
        """
        Denoting with s the number of affine systems, M_domains is a list with s elements, each element has other s elements, each one of these is a bigM vector.
        """
        self.M_domains = []
        for i, domain_i in enumerate(self.sys.domains):
            M_i = []
            for j, domain_j in enumerate(self.sys.domains):
                M_ij = []
                if i != j:
                    for k in range(domain_i.lhs_min.shape[0]):
                        sol = linear_program(-domain_i.lhs_min[k,:], domain_j.lhs_min, domain_j.rhs_min)
                        M_ijk = (- sol.min - domain_i.rhs_min[k])[0]
                        M_ij.append(M_ijk)
                M_ij = np.reshape(M_ij, (len(M_ij), 1))
                M_i.append(M_ij)
            self.M_domains.append(M_i)
        return

    def compute_M_dynamics(self):
        self.M_dynamics = []
        self.m_dynamics = []
        for i in range(self.sys.n_sys):
            M_i = []
            m_i = []
            lhs_i = np.hstack((self.sys.affine_systems[i].A, self.sys.affine_systems[i].B))
            rhs_i = self.sys.affine_systems[i].c
            for domain_j in self.sys.domains:
                M_ij = []
                m_ij = []
                for k in range(lhs_i.shape[0]):
                    sol = linear_program(-lhs_i[k,:], domain_j.lhs_min, domain_j.rhs_min)
                    M_ijk = (- sol.min + rhs_i[k])[0]
                    M_ij.append(M_ijk)
                    sol = linear_program(lhs_i[k,:], domain_j.lhs_min, domain_j.rhs_min)
                    m_ijk = (sol.min + rhs_i[k])[0]
                    m_ij.append(m_ijk)
                M_ij = np.reshape(M_ij, (len(M_ij), 1))
                m_ij = np.reshape(m_ij, (len(m_ij), 1))
                M_i.append(M_ij)
                m_i.append(np.array(m_ij))
            self.M_dynamics.append(M_i)
            self.m_dynamics.append(m_i)
        return

    def feedforward(self, x0, u_ws=None, x_ws=None, ss_ws=None):

        # gurobi model
        model = grb.Model()

        # variables
        x, model = real_variable(model, [self.N+1, self.sys.n_x])
        u, model = real_variable(model, [self.N, self.sys.n_u])
        z, model = real_variable(model, [self.N, self.sys.n_sys, self.sys.n_x])
        d = model.addVars(self.N, self.sys.n_sys, vtype=grb.GRB.BINARY, name='d')
        model.update()

        # warm start
        if x_ws is not None:
            for i in range(self.sys.n_x):
                for k in range(self.N+1):
                    x[k,i].setAttr('Start', x_ws[k][i,0])
            if ss_ws is not None:
                for i in range(self.sys.n_x):
                    for j in range(self.sys.n_sys):
                        for k in range(self.N):
                            if j == ss_ws[k]:
                                z[k,j,i].setAttr('Start', x_ws[k+1][i,0])
                            else:
                                z[k,j,i].setAttr('Start', 0.)
        if u_ws is not None:
            for i in range(self.sys.n_u):
                for k in range(self.N):
                    u[k,i].setAttr('Start', u_ws[k][i,0])
        if ss_ws is not None:
            for j in range(self.sys.n_sys):
                for k in range(self.N):
                    if j == ss_ws[k]:
                        d[k,j].setAttr('Start', 1)
                    else:
                        d[k,j].setAttr('Start', 0)

        # numpy variables (list of numpy matrices ordered in time)
        x_np = [np.array([[x[k,i]] for i in range(self.sys.n_x)]) for k in range(self.N+1)]
        u_np = [np.array([[u[k,i]] for i in range(self.sys.n_u)]) for k in range(self.N)]

        # set objective
        model = self.mip_objective(model, x_np, u_np)

        # initial condition
        model.addConstrs((x[0,i] == x0[i,0] for i in range(self.sys.n_x)))

        # set constraints
        model = self.mip_constraints(model, x_np, u_np, z, d)

        # set parameters
        time_limit = 600.
        model.setParam('OutputFlag', False)
        model.setParam('TimeLimit', time_limit)
        # model.setParam(grb.GRB.Param.OptimalityTol, 1.e-9)
        # model.setParam(grb.GRB.Param.FeasibilityTol, 1.e-9)
        # model.setParam(grb.GRB.Param.IntFeasTol, 1.e-9)
        # model.setParam(grb.GRB.Param.MIPGap, 1.e-9)

        # run optimization
        model.optimize()

        # return solution
        if model.status != grb.GRB.Status.OPTIMAL:
            # print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            u_feedforward = [np.full((self.sys.n_u,1), np.nan) for k in range(self.N)]
            x_trajectory = [np.full((self.sys.n_x,1), np.nan) for k in range(self.N+1)]
            cost = np.nan
            switching_sequence = [np.nan]*self.N
        else:
            if model.status == grb.GRB.Status.TIME_LIMIT:
                print('The solution of the MIQP excedeed the time limit of ' + str(time_limit))
            cost = model.objVal
            u_feedforward = [np.array([[model.getAttr('x', u)[k,i]] for i in range(self.sys.n_u)]) for k in range(self.N)]
            x_trajectory = [np.array([[model.getAttr('x', x)[k,i]] for i in range(self.sys.n_x)]) for k in range(self.N+1)]
            d_star = [np.array([[model.getAttr('x', d)[k,i]] for i in range(self.sys.n_sys)]) for k in range(self.N)]
            switching_sequence = [np.where(np.isclose(d, 1.))[0][0] for d in d_star]
        return u_feedforward, x_trajectory, tuple(switching_sequence), cost

    def mip_objective(self, model, x_np, u_np, x_ws=None, u_ws=None):

        # linear objective
        if self.objective_norm == 'one':
            phi = model.addVars(self.N+1, self.sys.n_x, name='phi')
            if x_ws is not None:
                for i in range(self.sys.n_x):
                    for k in range(self.N):
                        phi[k,i].setAttr('Start', self.Q[i,:].dot(x_ws[k]))
                    phi[self.N,i].setAttr('Start', self.P[i,:].dot(x_ws[self.N]))
            psi = model.addVars(self.N, self.sys.n_u, name='psi')
            if u_ws is not None:
                for i in range(self.sys.n_u):
                    for k in range(self.N):
                        psi[k,i].setAttr('Start', self.R[i,:].dot(u_ws[k]))
            model.update()
            V = 0.
            for k in range(self.N+1):
                for i in range(self.sys.n_x):
                    V += phi[k,i]
            for k in range(self.N):
                for i in range(self.sys.n_u):
                    V += psi[k,i]
            model.setObjective(V)
            for k in range(self.N):
                for i in range(self.sys.n_x):
                    model.addConstr(phi[k,i] >= self.Q[i,:].dot(x_np[k])[0])
                    model.addConstr(phi[k,i] >= - self.Q[i,:].dot(x_np[k])[0])
                for i in range(self.sys.n_u):
                    model.addConstr(psi[k,i] >= self.R[i,:].dot(u_np[k])[0])
                    model.addConstr(psi[k,i] >= - self.R[i,:].dot(u_np[k])[0])
            for i in range(self.sys.n_x):
                model.addConstr(phi[self.N,i] >= self.P[i,:].dot(x_np[self.N])[0])
                model.addConstr(phi[self.N,i] >= - self.P[i,:].dot(x_np[self.N])[0])

       # quadratic objective
        elif self.objective_norm == 'two':
            V = 0.
            for k in range(self.N):
                V += x_np[k].T.dot(self.Q).dot(x_np[k]) + u_np[k].T.dot(self.R).dot(u_np[k])
            V += x_np[self.N].T.dot(self.P).dot(x_np[self.N])
            model.setObjective(V[0,0])

        return model

    def mip_constraints(self, model, x_np, u_np, z, d):

        with suppress_stdout():

            # disjuction
            for k in range(self.N):
                model.addConstr(np.sum([d[k,i] for i in range(self.sys.n_sys)]) == 1.)

            # relaxation of the domains
            for k in range(self.N):
                for i in range(self.sys.n_sys):
                    expr_xu = self.sys.domains[i].lhs_min.dot(np.vstack((x_np[k], u_np[k]))) - self.sys.domains[i].rhs_min
                    expr_M = np.sum([self.M_domains[i][j]*d[k,j] for j in range(self.sys.n_sys) if j != i], axis=0)
                    expr = expr_xu - expr_M
                    model.addConstrs((expr[j][0] <= 0. for j in range(len(expr))))

            # state transition
            for k in range(self.N):
                for j in range(self.sys.n_x):
                    expr = 0.
                    for i in range(self.sys.n_sys):
                        expr += z[k,i,j]
                    model.addConstr(x_np[k+1][j,0] == expr)

            # relaxation of the dynamics, part 1
            for k in range(self.N):
                for i in range(self.sys.n_sys):
                    expr_M = self.M_dynamics[i][i]*d[k,i]
                    expr_m = self.m_dynamics[i][i]*d[k,i]
                    for j in range(self.sys.n_x):
                        model.addConstr(z[k,i,j] <= expr_M[j,0])
                        model.addConstr(z[k,i,j] >= expr_m[j,0])

            # relaxation of the dynamics, part 2
            for k in range(self.N):
                for i in range(self.sys.n_sys):
                    expr = self.sys.affine_systems[i].A.dot(x_np[k]) + self.sys.affine_systems[i].B.dot(u_np[k]) + self.sys.affine_systems[i].c
                    expr_M = expr - np.sum([self.M_dynamics[i][j]*d[k,j] for j in range(self.sys.n_sys) if j != i], axis=0)
                    expr_m = expr - np.sum([self.m_dynamics[i][j]*d[k,j] for j in range(self.sys.n_sys) if j != i], axis=0)
                    for j in range(self.sys.n_x):
                        model.addConstr(z[k,i,j] >= expr_M[j,0])
                        model.addConstr(z[k,i,j] <= expr_m[j,0])

            # terminal constraint
            expr = self.X_N.lhs_min.dot(x_np[self.N]) - self.X_N.rhs_min
            model.addConstrs((expr[i,0] <= 0. for i in range(len(self.X_N.minimal_facets))))

        return model

    def feedback(self, x0):
        return self.feedforward(x0)[0][0]

    def condense_program(self, switching_sequence):
        tic = time.time()
        # print('\nCondensing the OCP for the switching sequence ' + str(switching_sequence) + ' ...')
        if len(switching_sequence) != self.N:
            raise ValueError('Switching sequence not coherent with the controller horizon.')
        prog = OCP_condenser(self.sys, self.objective_norm, self.Q, self.R, self.P, self.X_N, switching_sequence)
        # print('... OCP condensed in ' + str(time.time() -tic ) + ' seconds.\n')
        return prog

# class FeasibleSetLibrary:
#     """
#     library[switching_sequence]
#     - program
#     - feasible_set
#     """

#     def __init__(self, controller):
#         self.controller = controller
#         self.library = dict()
#         return

#     def sample_policy(self, n_samples, X=None):
#         for i in range(n_samples):
#             print('Sample ' + str(i) + ':'),
#             x = self.random_sample(X)
#             if not self.sampling_rejection(x):
#                 ss = self.controller.feedforward(x)[2]
#                 if not any(np.isnan(ss)):
#                     print('new switching sequence ' + str(ss) + '.')
#                     self.library[ss] = dict()
#                     prog = self.controller.condense_program(ss)
#                     self.library[ss]['program'] = prog
#                     self.library[ss]['feasible_set'] = prog.feasible_set
#                 else:
#                     print('unfeasible.')
#             else:
#                 print('rejected.')
#         return

#     def random_sample(self, X=None):
#         if X is None:
#             x = np.random.rand(self.controller.sys.n_x, 1)
#             x = np.multiply(x, (self.controller.sys.x_max - self.controller.sys.x_min)) + self.controller.sys.x_min
#         else:
#             is_inside = False
#             while not is_inside:
#                 x = np.random.rand(self.controller.sys.n_x,1)
#                 x = np.multiply(x, (self.controller.sys.x_max - self.controller.sys.x_min)) + self.controller.sys.x_min
#                 is_inside = X.applies_to(x)
#         return x

#     def sampling_rejection(self, x):
#         for ss_value in self.library.values():
#             if ss_value['feasible_set'].applies_to(x):
#                 return True
#         return False

#     def get_feasible_switching_sequences(self, x):
#         return [ss for ss, ss_values in self.library.items() if ss_values['feasible_set'].applies_to(x)]

#     def feedforward(self, x, given_ss=None):
#         fss = self.get_feasible_switching_sequences(x)
#         if given_ss is not None:
#             fss.insert(0, given_ss)
#         if not fss:
#             V_star = np.nan
#             u_star = [np.full((self.controller.sys.n_u, 1), np.nan) for i in range(self.controller.N)]
#             ss_star = [np.nan]*self.controller.N
#             return u_star, V_star, ss_star
#         else:
#             V_star = np.inf
#             for ss in fss:
#                 u, V = self.library[ss]['program'].solve(x)
#                 if V < V_star:
#                     V_star = V
#                     u_star = [u[i*self.controller.sys.n_u:(i+1)*self.controller.sys.n_u,:] for i in range(self.controller.N)]
#                     ss_star = ss
#         return u_star, V_star, ss_star

#     def feedback(self, x, given_ss=None):
#         u_star, V_star, ss_star = self.feedforward(x, given_ss)
#         return u_star[0], ss_star

#     def add_shifted_switching_sequences(self, terminal_domain):
#         for ss in self.library.keys():
#             for shifted_ss in self.shift_switching_sequence(ss, terminal_domain):
#                 if not self.library.has_key(shifted_ss):
#                     self.library[shifted_ss] = dict()
#                     self.library[shifted_ss]['program'] = self.controller.condense_program(shifted_ss)
#                     self.library[shifted_ss]['feasible_set'] = EmptyFeasibleSet()

#     @staticmethod
#     def shift_switching_sequence(ss, terminal_domain):
#         return [ss[i:] + (terminal_domain,)*i for i in range(1,len(ss))]

#     def plot_partition(self):
#         for ss_value in self.library.values():
#             color = np.random.rand(3,1)
#             fs = ss_value['feasible_set']
#             if not fs.empty:
#                 fs.plot(facecolor=color, alpha=.5)
#         return



class FeasibleSetLibrary:
    """
    library[switching_sequence]
    - program
    - feasible_set
    """

    def __init__(self, controller):
        self.controller = controller
        self.library = dict()
        return

    def sample_policy(self, n_samples, state_domains=None):
        n_rejected = 0
        n_included = 0
        n_new_ss = 0
        n_unfeasible = 0
        for i in range(n_samples):
            print('Sample ' + str(i) + ': ')
            x = self.random_sample(state_domains)
            if not self.sampling_rejection(x):
                print('solving MIQP... '),
                tic = time.time()
                ss = self.controller.feedforward(x)[2]
                print('solution found in ' + str(time.time()-tic) + ' s.')
                if not any(np.isnan(ss)):
                    if self.library.has_key(ss):
                        n_included += 1
                        print('included.')
                        print('including sample in inner approximation... '),
                        tic = time.time()
                        self.library[ss]['feasible_set'].include_point(x)
                        print('sample included in ' + str(time.time()-tic) + ' s.')
                    else:
                        n_new_ss += 1
                        print('new switching sequence ' + str(ss) + '.')
                        self.library[ss] = dict()
                        print('condensing QP... '),
                        tic = time.time()
                        prog = self.controller.condense_program(ss)
                        print('QP condensed in ' + str(time.time()-tic) + ' s.')
                        self.library[ss]['program'] = prog
                        lhs = np.hstack((-prog.C_x, prog.C_u))
                        rhs = prog.C
                        residual_dimensions = range(prog.C_x.shape[1])
                        print('constructing inner simplex... '),
                        tic = time.time()
                        feasible_set = PolytopeProjectionInnerApproximation(lhs, rhs, residual_dimensions)
                        print('inner simplex constructed in ' + str(time.time()-tic) + ' s.')
                        print('including sample in inner approximation... '),
                        tic = time.time()
                        feasible_set.include_point(x)
                        print('sample included in ' + str(time.time()-tic) + ' s.')
                        self.library[ss]['feasible_set'] = feasible_set
                else:
                    n_unfeasible += 1
                    print('unfeasible.')
            else:
                n_rejected += 1
                print('rejected.')
        print('\nTotal number of samples: ' + str(n_samples) + ', switching sequences found: ' + str(n_new_ss) + ', included samples: ' + str(n_included) + ', rejected samples: ' + str(n_rejected) + ', unfeasible samples: ' + str(n_unfeasible) + '.')
        return

    def random_sample(self, state_domains=None):
        if state_domains is None:
            x = np.random.rand(self.controller.sys.n_x, 1)
            x = np.multiply(x, (self.controller.sys.x_max - self.controller.sys.x_min)) + self.controller.sys.x_min
        else:
            is_inside = False
            while not is_inside:
                x = np.random.rand(self.controller.sys.n_x,1)
                x = np.multiply(x, (self.controller.sys.x_max - self.controller.sys.x_min)) + self.controller.sys.x_min
                for X in state_domains:
                    if X.applies_to(x):
                        is_inside = True
                        break
        return x

    def sampling_rejection(self, x):
        for ss_value in self.library.values():
            if ss_value['feasible_set'].applies_to(x):
                return True
        return False

    def get_feasible_switching_sequences(self, x):
        return [ss for ss, ss_values in self.library.items() if ss_values['feasible_set'].applies_to(x)]

    def feedforward(self, x, given_ss=None):
        V_star = np.nan
        u_star = [np.full((self.controller.sys.n_u, 1), np.nan) for i in range(self.controller.N)]
        ss_star = [np.nan]*self.controller.N
        fss = self.get_feasible_switching_sequences(x)
        if given_ss is not None:
            fss.insert(0, given_ss)
        if not fss:
            return u_star, V_star, ss_star
        else:
            for ss in fss:
                u, V = self.library[ss]['program'].solve(x)
                if V < V_star or (np.isnan(V_star) and not np.isnan(V)):
                    V_star = V
                    u_star = [u[i*self.controller.sys.n_u:(i+1)*self.controller.sys.n_u,:] for i in range(self.controller.N)]
                    ss_star = ss
        return u_star, V_star, ss_star

    def feedback(self, x, given_ss=None):
        u_star, V_star, ss_star = self.feedforward(x, given_ss)
        return u_star[0], ss_star

    def add_shifted_switching_sequences(self, terminal_domain):
        for ss in self.library.keys():
            for shifted_ss in self.shift_switching_sequence(ss, terminal_domain):
                if not self.library.has_key(shifted_ss):
                    self.library[shifted_ss] = dict()
                    self.library[shifted_ss]['program'] = self.controller.condense_program(shifted_ss)
                    self.library[shifted_ss]['feasible_set'] = EmptyFeasibleSet()

    @staticmethod
    def shift_switching_sequence(ss, terminal_domain):
        return [ss[i:] + (terminal_domain,)*i for i in range(1,len(ss))]

    def plot_partition(self):
        for ss_value in self.library.values():
            color = np.random.rand(3,1)
            fs = ss_value['feasible_set']
            if not fs.empty:
                p = Polytope(fs.hull.A, fs.hull.b)
                p.assemble()#redundant=False, vertices=fs.hull.points)
                p.plot(facecolor=color, alpha=.5)
        return



class EmptyFeasibleSet:

    def __init__(self):
        self.empty = True
        return

    def applies_to(self, x):
        return False


class parametric_lp:

    def __init__(self, F_u, F_x, F, C_u, C_x, C):
        """
        LP in the form:
        min  \sum_i | (F_u u + F_x x + F)_i |
        s.t. C_u u <= C_x x + C
        """
        self.F_u = F_u
        self.F_x = F_x
        self.F = F
        self.C_u = C_u
        self.C_x = C_x
        self.C = C
        self.add_slack_variables()
        return

    def add_slack_variables(self):
        """
        Reformulates the LP as:
        min f^T z
        s.t. A z <= B x + c
        """
        n_slack = self.F.shape[0]
        n_u = self.F_u.shape[1]
        self.f = np.vstack((
            np.zeros((n_u,1)),
            np.ones((n_slack,1))
            ))
        self.A = np.vstack((
            np.hstack((self.C_u, np.zeros((self.C_u.shape[0], n_slack)))),
            np.hstack((self.F_u, -np.eye(n_slack))),
            np.hstack((-self.F_u, -np.eye(n_slack)))
            ))
        self.B = np.vstack((self.C_x, -self.F_x, self.F_x))
        self.c = np.vstack((self.C, -self.F, self.F))
        self.n_var = n_u + n_slack
        self.n_cons = self.A.shape[0]
        return

    def solve(self, x0, u_length=None):
        x0 = np.reshape(x0, (x0.shape[0], 1))
        sol = linear_program(self.f, self.A, self.B.dot(x0)+self.c)
        u_star = sol.argmin[0:self.F_u.shape[1]]
        if u_length is not None:
            if not float(u_star.shape[0]/u_length).is_integer():
                raise ValueError('Uncoherent dimension of the input u_length.')
            u_star = [u_star[i*u_length:(i+1)*u_length,:] for i in range(u_star.shape[0]/u_length)]
        return u_star, sol.min


class parametric_qp:

    def __init__(self, F_uu, F_xu, F_xx, F_u, F_x, F, C_u, C_x, C):
        """
        Multiparametric QP in the form:
        min  .5 u' F_{uu} u + x0' F_{xu} u + F_u' u + .5 x0' F_{xx} x0 + F_x' x0 + F
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

    def solve(self, x0, u_length=None):
        x0 = np.reshape(x0, (x0.shape[0], 1))
        H = self.F_uu
        f = x0.T.dot(self.F_xu) + self.F_u.T
        A = self.C_u
        b = self.C + self.C_x.dot(x0)
        u_star, cost = quadratic_program(H, f, A, b)
        cost += .5*x0.T.dot(self.F_xx).dot(x0) + self.F_x.T.dot(x0) + self.F
        if u_length is not None:
            if not float(u_star.shape[0]/u_length).is_integer():
                raise ValueError('Uncoherent dimension of the input u_length.')
            u_star = [u_star[i*u_length:(i+1)*u_length,:] for i in range(u_star.shape[0]/u_length)]
        return u_star, cost[0,0]

    def get_active_set(self, x, u, tol=1.e-6):
        u = np.vstack(u)
        return tuple(np.where((self.C_u.dot(u) - self.C - self.C_x.dot(x)) > -tol)[0])

    def remove_linear_terms(self):
        """
        Applies the change of variables z = u + F_uu^-1 (F_xu' x + F_u')
        that puts the cost function in the form
        V = 1/2 z' H z + 1/2 x' F_xx_q x + F_x_q' x + F_q
        and the constraints in the form:
        G u <= W + S x
        """
        self.H_inv = np.linalg.inv(self.F_uu)
        self.H = self.F_uu
        self.F_xx_q = self.F_xx - self.F_xu.dot(self.H_inv).dot(self.F_xu.T)
        self.F_x_q = self.F_x - self.F_xu.dot(self.H_inv).dot(self.F_u)
        self.F_q = self.F - .5*self.F_u.T.dot(self.H_inv).dot(self.F_u)
        self.G = self.C_u
        self.S = self.C_x + self.C_u.dot(self.H_inv).dot(self.F_xu.T)
        self.W = self.C + self.C_u.dot(self.H_inv).dot(self.F_u)
        return

    @property
    def feasible_set(self):
        if self._feasible_set is None:
            augmented_polytope = Polytope(np.hstack((- self.C_x, self.C_u)), self.C)
            augmented_polytope.assemble()
            self._feasible_set = augmented_polytope.orthogonal_projection(range(self.C_x.shape[1]))
        return self._feasible_set


    def get_z_sensitivity(self, active_set):
        # clean active set
        G_A = self.G[active_set,:]
        if active_set and np.linalg.matrix_rank(G_A) < G_A.shape[0]:
            lir = linearly_independent_rows(G_A)
            active_set = [active_set[i] for i in lir]

        # multipliers explicit solution
        inactive_set = sorted(list(set(range(self.C.shape[0])) - set(active_set)))
        [G_A, W_A, S_A] = [self.G[active_set,:], self.W[active_set,:], self.S[active_set,:]]
        [G_I, W_I, S_I] = [self.G[inactive_set,:], self.W[inactive_set,:], self.S[inactive_set,:]]
        H_A = np.linalg.inv(G_A.dot(self.H_inv).dot(G_A.T))
        lambda_A_offset = - H_A.dot(W_A)
        lambda_A_linear = - H_A.dot(S_A)

        # primal variables explicit solution
        z_offset = - self.H_inv.dot(G_A.T.dot(lambda_A_offset))
        z_linear = - self.H_inv.dot(G_A.T.dot(lambda_A_linear))
        return z_offset, z_linear

    def get_u_sensitivity(self, active_set):
        z_offset, z_linear = self.get_z_sensitivity(active_set)

        # primal original variables explicit solution
        u_offset = z_offset - self.H_inv.dot(self.F_u)
        u_linear = z_linear - self.H_inv.dot(self.F_xu.T)
        return u_offset, u_linear

    def get_cost_sensitivity(self, x_list, active_set):
        z_offset, z_linear = self.get_z_sensitivity(active_set)

        # optimal value function explicit solution: V_star = .5 x' V_quadratic x + V_linear x + V_offset
        V_quadratic = z_linear.T.dot(self.H).dot(z_linear) + self.F_xx_q
        V_linear = z_offset.T.dot(self.H).dot(z_linear) + self.F_x_q.T
        V_offset = self.F_q + .5*z_offset.T.dot(self.H).dot(z_offset)

        # tangent approximation
        plane_list = []
        for x in x_list:
            A = x.T.dot(V_quadratic) + V_linear
            b = -.5*x.T.dot(V_quadratic).dot(x) + V_offset
            plane_list.append([A, b])

        return plane_list

    def solve_free_x(self):
        H = np.vstack((
            np.hstack((self.F_uu, self.F_xu.T)),
            np.hstack((self.F_xu, self.F_xx))
            ))
        f = np.vstack((self.F_u, self.F_x))
        A = np.hstack((self.C_u, -self.C_x))
        b = self.C
        z_star, cost = quadratic_program(H, f, A, b)
        u_star = z_star[0:self.F_uu.shape[0],:]
        x_star = z_star[self.F_uu.shape[0]:,:]
        return u_star, x_star, cost

### AUXILIARY FUNCTIONS

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def linearly_independent_rows(A, tol=1.e-6):
    R = linalg.qr(A.T)[1]
    R_diag = np.abs(np.diag(R))
    return list(np.where(R_diag > tol)[0])

def OCP_condenser(sys, objective_norm, Q, R, P, X_N, switching_sequence):
    tic = time.time()
    N = len(switching_sequence)
    Q_bar = linalg.block_diag(*[Q for i in range(N)] + [P])
    R_bar = linalg.block_diag(*[R for i in range(N)])
    G, W, E = constraint_condenser(sys, X_N, switching_sequence)
    if objective_norm == 'one':
        F_u, F_x, F = linear_objective_condenser(sys, Q_bar, R_bar, switching_sequence)
        parametric_program = parametric_lp(F_u, F_x, F, G, E, W)
    elif objective_norm == 'two':
        F_uu, F_xu, F_xx, F_u, F_x, F = quadratic_objective_condenser(sys, Q_bar, R_bar, switching_sequence)
        parametric_program = parametric_qp(F_uu, F_xu, F_xx, F_u, F_x, F, G, E, W)
    # print 'total condensing time is', str(time.time()-tic),'s.\n'
    return parametric_program

def constraint_condenser(sys, X_N, switching_sequence):
    N = len(switching_sequence)
    D_sequence = [sys.domains[switching_sequence[i]] for i in range(N)]
    lhs_x = linalg.block_diag(*[D.lhs_min[:,:sys.n_x] for D in D_sequence] + [X_N.lhs_min])
    lhs_u = linalg.block_diag(*[D.lhs_min[:,sys.n_x:] for D in D_sequence])
    lhs_u = np.vstack((lhs_u, np.zeros((X_N.lhs_min.shape[0], lhs_u.shape[1]))))
    rhs = np.vstack([D.rhs_min for D in D_sequence] + [X_N.rhs_min])
    A_bar, B_bar, c_bar = sys.condense(switching_sequence)
    G = (lhs_x.dot(B_bar) + lhs_u)
    W = rhs - lhs_x.dot(c_bar)
    E = - lhs_x.dot(A_bar)
    # # the following might be super slow (and is not necessary)
    # n_ineq_before = G.shape[0]
    # tic = time.time()
    # p = Polytope(np.hstack((G, -E)), W)
    # p.assemble()
    # if not p.empty:
    #     G = p.lhs_min[:,:sys.n_u*N]
    #     E = - p.lhs_min[:,sys.n_u*N:]
    #     W = p.rhs_min
    #     n_ineq_after = G.shape[0]
    # else:
    #     G = None
    #     W = None
    #     E = None
    # print '\n' + str(n_ineq_before - n_ineq_after) + 'on' + str(n_ineq_before) + 'redundant inequalities removed in', str(time.time()-tic),'s,',
    return G, W, E

def linear_objective_condenser(sys, Q_bar, R_bar, switching_sequence):
    """
    \sum_i | (F_u u + F_x x + F)_i |
    """
    A_bar, B_bar, c_bar = sys.condense(switching_sequence)
    F_u = np.vstack((Q_bar.dot(B_bar), R_bar))
    F_x = np.vstack((Q_bar.dot(A_bar), np.zeros((R_bar.shape[0], A_bar.shape[1]))))
    F = np.vstack((Q_bar.dot(c_bar), np.zeros((R_bar.shape[0], 1))))
    return F_u, F_x, F

def quadratic_objective_condenser(sys, Q_bar, R_bar, switching_sequence):
    """
    .5 u' F_{uu} u + x0' F_{xu} u + F_u' u + .5 x0' F_{xx} x0 + F_x' x0 + F
    """
    A_bar, B_bar, c_bar = sys.condense(switching_sequence)
    F_uu = 2*(R_bar + B_bar.T.dot(Q_bar).dot(B_bar))
    F_xu = 2*A_bar.T.dot(Q_bar).dot(B_bar)
    F_xx = 2.*A_bar.T.dot(Q_bar).dot(A_bar)
    F_u = 2.*B_bar.T.dot(Q_bar).dot(c_bar)
    F_x = 2.*A_bar.T.dot(Q_bar).dot(c_bar)
    F = c_bar.T.dot(Q_bar).dot(c_bar)
    return F_uu, F_xu, F_xx, F_u, F_x, F

def remove_initial_state_constraints(prog, tol=1e-10):
    C_u_rows_norm = list(np.linalg.norm(prog.C_u, axis=1))
    intial_state_contraints = [i for i, row_norm in enumerate(C_u_rows_norm) if row_norm < tol]
    prog.C_u = np.delete(prog.C_u,intial_state_contraints, 0)
    prog.C_x = np.delete(prog.C_x,intial_state_contraints, 0)
    prog.C = np.delete(prog.C,intial_state_contraints, 0)
    prog.remove_linear_terms()
    return prog


def explict_solution_from_hybrid_condensing(prog, tol=1e-10):
    porg = remove_initial_state_constraints(prog)
    mpqp_solution = MPQPSolver(prog)
    critical_regions = mpqp_solution.critical_regions
    return critical_regions