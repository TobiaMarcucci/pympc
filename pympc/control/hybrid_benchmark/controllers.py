# external imports
import numpy as np
import gurobipy as grb
from scipy.linalg import block_diag
from copy import copy, deepcopy

# internal inputs
from pympc.control.hybrid_benchmark.utils import (graph_representation,
                                                  big_m,
                                                  add_vars,
                                                  add_linear_inequality,
                                                  add_linear_equality,
                                                  add_rotated_socc,
                                                  infeasible_mode_sequences
                                                  )
from pympc.optimization.programs import linear_program

class HybridModelPredictiveController(object):

    def __init__(self, S, N, Q, R, P, X_N, method='Big-M'):

        # store inputs
        self.S = S
        self.N = N
        self.Q = Q
        self.R = R
        self.P = P
        self.X_N = X_N

        # mpMIQP
        self.prog = self.build_mpmiqp(method)
        # self.partial_mode_sequence = []

    def build_mpmiqp(self, method):
        if method == 'Traditional formulation':
            return self.bild_miqp_bemporad_morari()
        if method == 'Big-M':
            return self.bild_miqp_bigm()
        if method == 'Convex hull':
            return self.bild_miqp_convex_hull()
        if method == 'Convex hull, lifted constraints':
            return self.build_miqp_convex_hull_lifted_constraints()
        else:
            raise ValueError('unknown method ' + method + '.')

    def add_reachability_constraints(self, t_max):

        imss = infeasible_mode_sequences(self.S, t_max)
        for ms in imss:
            for t in range(self.N-len(ms)):
                self.prog.addConstr(sum([self.prog.getVarByName('d%d[%d]'%(t+tau,m)) for tau, m in enumerate(ms)]) <= len(ms)-1.)

    def bild_miqp_bemporad_morari(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # big-M dynamics
        alpha = []
        beta = []
        for i, S_i in enumerate(self.S.affine_systems):
            alpha_i = []
            beta_i = []
            A_i = np.hstack((S_i.A, S_i.B))
            for j, S_j in enumerate(self.S.affine_systems):
                alpha_ij = []
                beta_ij = []
                D_j = self.S.domains[j]
                for k in range(S_i.nx):
                    sol = linear_program(A_i[k], D_j.A, D_j.b, D_j.C, D_j.d)
                    alpha_ij.append(sol['min'] + S_i.c[k])
                    sol = linear_program(-A_i[k], D_j.A, D_j.b, D_j.C, D_j.d)
                    beta_ij.append(- sol['min'] + S_i.c[k])
                alpha_i.append(np.array(alpha_ij))
                beta_i.append(np.array(beta_ij))
            alpha.append(alpha_i)
            beta.append(beta_i)
        alpha = np.minimum.reduce([np.minimum.reduce(alpha_i) for alpha_i in alpha])
        beta = np.maximum.reduce([np.maximum.reduce(beta_i) for beta_i in beta])

        # big-M domains
        gamma = []
        for i, D_i in enumerate(self.S.domains):
            gamma_i = []
            for j, D_j in enumerate(self.S.domains):
                gamma_ij = []
                for k in range(D_i.A.shape[0]):
                    sol = linear_program(-D_i.A[k], D_j.A, D_j.b, D_j.C, D_j.d)
                    gamma_ij.append(- sol['min'] - D_i.b[k])
                gamma_i.append(np.array(gamma_ij))
            gamma.append(gamma_i)
        gamma = [np.maximum.reduce(gamma_i) for gamma_i in gamma]

        # initialize program
        prog = grb.Model()
        obj = 0.

        # loop over the time horizon
        for t in range(self.N):

            # initial conditions
            if t == 0:
                x = add_vars(prog, nx, name='x0')

            # stage variables
            else:
                x = x_next
            x_next = add_vars(prog, nx, name='x%d'%(t+1))
            z = [add_vars(prog, nx) for i in range(nm)]
            u = add_vars(prog, nu, name='u%d'%t)
            d = add_vars(prog, nm, lb=[0.]*nm, name='d%d'%t)
            prog.update()

            # constrained dynamics
            add_linear_equality(prog, x_next, sum(z))
            xu = np.concatenate((x, u))
            for i in range(nm):
                Ai = self.S.affine_systems[i].A
                Bi = self.S.affine_systems[i].B
                ci = self.S.affine_systems[i].c
                dyn_i = Ai.dot(x) + Bi.dot(u) + ci
                add_linear_inequality(prog, alpha*d[i], z[i])
                add_linear_inequality(prog, z[i], beta*d[i])
                add_linear_inequality(prog, alpha*(1.-d[i]), dyn_i - z[i])
                add_linear_inequality(prog, dyn_i - z[i], beta*(1.-d[i]))
                add_linear_inequality(prog, self.S.domains[i].A.dot(xu), self.S.domains[i].b + gamma[i]*(1.-d[i]))

            # constraints on the binaries
            prog.addConstr(sum(d) == 1.)
            # prog.addSOS(d, [1.]*nm, grb.GRB.SOS_TYPE1)

            # stage cost to the objective
            obj += .5 * (x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))

        # terminal constraint
        add_linear_inequality(prog, self.X_N.A.dot(x_next), self.X_N.b)

        # terminal cost
        obj += .5 * x_next.dot(self.P).dot(x_next)
        prog.setObjective(obj)

        return prog

    def bild_miqp_bigm(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # graph of the dynamics and big-Ms
        P = graph_representation(self.S)
        m = big_m(P)

        # graph of the dynamics and big-Ms, final stage
        P_N = [copy(Pi) for Pi in P]
        [Pi.add_inequality(self.X_N.A, self.X_N.b, range(nx+nu, 2*nx+nu)) for Pi in P_N]
        P_N_indices = [i for i, Pi in enumerate(P_N) if not Pi.empty]
        m_N = big_m([P_N[i] for i in P_N_indices])

        # initialize program
        prog = grb.Model()
        obj = 0.

        # loop over the time horizon
        for t in range(self.N):

            # initial conditions
            if t == 0:
                x = add_vars(prog, nx, name='x0')

            # stage variables
            else:
                x = x_next
            x_next = add_vars(prog, nx, name='x%d'%(t+1))
            u = add_vars(prog, nu, name='u%d'%t)
            d = add_vars(prog, nm, lb=[0.]*nm, name='d%d'%t)
            prog.update()

            # constrained dynamics
            xux = np.concatenate((x, u, x_next))
            if t < self.N-1:
                for i in range(nm):
                    sum_mi = sum(m[i][j] * d[j] for j in range(nm) if j != i)
                    add_linear_inequality(prog, P[i].A.dot(xux), P[i].b + sum_mi)
            else:
            	for i in P_N_indices:
                    sum_mi = sum(m_N[i][j] * d[j] for j in P_N_indices if j != i)
                    add_linear_inequality(prog, P_N[i].A.dot(xux), P_N[i].b + sum_mi)

            # constraints on the binaries
            prog.addConstr(sum(d) == 1.)
            # prog.addSOS(grb.GRB.SOS_TYPE1, d, [1.]*d.size)

            # stage cost to the objective
            obj += .5 * (x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))

        # terminal cost
        obj += .5 * x_next.dot(self.P).dot(x_next)
        prog.setObjective(obj)

        return prog

    def bild_miqp_convex_hull(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # graph of the dynamics
        P = graph_representation(self.S)

        # initialize program
        prog = grb.Model()
        obj = 0.

        # loop over the time horizon
        for t in range(self.N):

            # initial conditions
            if t == 0:
                x = add_vars(prog, nx, name='x0')

            # stage variables
            else:
                x = x_next
            x_next = add_vars(prog, nx, name='x%d'%(t+1))
            u = add_vars(prog, nu, name='u%d'%t)
            d = add_vars(prog, nm, lb=[0.]*nm, name='d%d'%t)

            # auxiliary continuous variables for the convex-hull method
            y = [add_vars(prog, nx) for i in range(nm)]
            z = [add_vars(prog, nx) for i in range(nm)]
            v = [add_vars(prog, nu) for i in range(nm)]
            prog.update()

            # constrained dynamics
            for i in range(nm):
                yvzi = np.concatenate((y[i], v[i], z[i]))
                add_linear_inequality(prog, P[i].A.dot(yvzi), P[i].b * d[i])

            # recompose the state and input (convex hull method)
            add_linear_equality(prog, x, sum(y))
            add_linear_equality(prog, x_next, sum(z))
            add_linear_equality(prog, u, sum(v))

            # constraints on the binaries
            prog.addConstr(sum(d) == 1.)
            # prog.addSOS(grb.GRB.SOS_TYPE1, d, [1.]*d.size)

            # stage cost to the objective
            obj += .5 * (x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))

        # terminal constraint
        for i in range(nm):
            add_linear_inequality(prog, self.X_N.A.dot(z[i]), self.X_N.b * d[i])

        # terminal cost
        obj += .5 * x_next.dot(self.P).dot(x_next)
        prog.setObjective(obj)

        return prog

    def build_miqp_convex_hull_lifted_constraints(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # graph of the dynamics
        P = graph_representation(self.S)

        # initialize program
        prog = grb.Model()
        obj = 0.
        
        # loop over time
        for t in range(self.N):

            # initial conditions
            if t == 0:
                x = add_vars(prog, nx, name='x0')

            # stage variables
            else:
                x = x_next
            x_next = add_vars(prog, nx, name='x%d'%(t+1))
            u = add_vars(prog, nu, name='u%d'%t)
            d = add_vars(prog, nm, lb=[0.]*nm, name='d%d'%t)

            # auxiliary continuous variables for the improved convex-hull method
            y = [add_vars(prog, nx) for i in range(nm)]
            z = [add_vars(prog, nx) for i in range(nm)]
            v = [add_vars(prog, nu) for i in range(nm)]
            s = add_vars(prog, nm, lb=[0.]*nm)
            prog.update()

            # constrained dynamics
            for i in range(nm):
                yvzi = np.concatenate((y[i], v[i], z[i]))
                add_linear_inequality(prog, P[i].A.dot(yvzi), P[i].b * d[i])

                # stage cost
                if t < self.N - 1:
                    add_rotated_socc(
                        prog,
                        block_diag(self.Q, self.R),
                        np.concatenate((y[i], v[i])),
                        s[i],
                        d[i]
                        )

                # final time step
                else:

                    # terminal cost
                    add_rotated_socc(
                        prog,
                        block_diag(self.Q, self.R, self.P),
                        np.concatenate((y[i], v[i], z[i])),
                        s[i],
                        d[i]
                        )

                    # terminal constraint
                    add_linear_inequality(prog, self.X_N.A.dot(z[i]), self.X_N.b * d[i])

            # update cost
            obj += sum(s)

            # recompose the state and input (convex hull method)
            add_linear_equality(prog, x, sum(y))
            add_linear_equality(prog, x_next, sum(z))
            add_linear_equality(prog, u, sum(v))

            # constraints on the binaries
            prog.addConstr(sum(d) == 1.)
            # prog.addSOS(grb.GRB.SOS_TYPE1, d, [1.]*d.size)

        # set cost
        prog.setObjective(obj)

        return prog

    def set_initial_condition(self, x0):
        for k in range(self.S.nx):
            self.prog.getVarByName('x0[%d]'%k).LB = x0[k]
            self.prog.getVarByName('x0[%d]'%k).UB = x0[k]
        self.prog.update()

    def _reset_initial_condition(self):
    	for k in range(self.S.nx):
    		self.prog.getVarByName('x0[%d]'%k).LB = -grb.GRB.INFINITY
    		self.prog.getVarByName('x0[%d]'%k).UB = grb.GRB.INFINITY

    def set_binaries(self, identifier):
    	'''
    	`identifier` is a dictionary of dictionaries.
    	`identifier[time][mode]`, if the keys exist, gives the value for d[time][mode].
    	'''
    	self._reset_binaries()
    	for k, v in identifier.items():
    		self.prog.getVarByName('d%d[%d]'%k).LB = v
    		self.prog.getVarByName('d%d[%d]'%k).UB = v
    	self.prog.update()

    def _reset_binaries(self):
    	for t in range(self.N):
    		for k in range(self.S.nm):
    			self.prog.getVarByName('d%d[%d]'%(t, k)).LB = 0.
    			self.prog.getVarByName('d%d[%d]'%(t, k)).UB = grb.GRB.INFINITY

    def _set_type_binaries(self, d_type):
        for t in range(self.N):
            for k in range(self.S.nm):
                self.prog.getVarByName('d%d[%d]'%(t,k)).VType = d_type

    def reset_program(self):
        self._set_type_binaries('C')
        self.reset_binaries()
        self.reset_initial_condition()
        self.prog.update()

    def solve_relaxation(self, x0, identifier):

        # reset program
        self.prog.reset()

        # set up miqp
        self._set_type_binaries('C')
        self.set_initial_condition(x0)
    	self.set_binaries(identifier)
        
        # parameters
        self.prog.setParam('OutputFlag', 0)
        # self.prog.setParam('Method', 0)
        # self.prog.setParam('BarConvTol', 1.e-8)

        # run the optimization
        self.prog.optimize()

        # elaborate result
        solution = self.organize_result()
        objective = solution['objective']
        feasible = objective is not None
        if feasible:
            integer_feasible = all(np.isclose(max(dt), 1., atol=1.e-6) for dt in solution['binaries'])
        else:
        	integer_feasible = None

        return solution, feasible, objective, integer_feasible

    def branching_rule(self, identifier, sol):

    	t = np.argmin([max(dt) - min(dt) for dt in sol['binaries']])
    	free_bin_id = range(self.S.nm)
    	[free_bin_id.remove(k[1]) for k in identifier.keys() if k[0] == t]
    	free_bin_val = [sol['binaries'][t][i] for i in free_bin_id]
    	free_bin_id_ordered = [free_bin_id[i] for i in np.argsort(free_bin_val)]

    	n = len(free_bin_id)
    	free_bin_id_1 = free_bin_id_ordered[(n+1)/2:]
    	free_bin_id_2 = free_bin_id_ordered[:(n+1)/2]

    	branch_1 = {}
    	branch_2 = {}
    	for i in free_bin_id_1:
    		branch_1[(t,i)] = 0.
    	for i in free_bin_id_2:
    		branch_2[(t,i)] = 0.

    	return [branch_1, branch_2]

    # def update_mode_sequence(self, partial_mode_sequence):

    #     # loop over the time horizon
    #     for t in range(self.N):

    #         # earse and write
    #         if t < len(partial_mode_sequence) and t < len(self.partial_mode_sequence):
    #             if partial_mode_sequence[t] != self.partial_mode_sequence[t]:
    #                 self.prog.getVarByName('d%d[%d]'%(t,self.partial_mode_sequence[t])).LB = 0.
    #                 self.prog.getVarByName('d%d[%d]'%(t,partial_mode_sequence[t])).LB = 1.

    #         # erase only
    #         elif t >= len(partial_mode_sequence) and t < len(self.partial_mode_sequence):
    #             self.prog.getVarByName('d%d[%d]'%(t,self.partial_mode_sequence[t])).LB = 0.

    #         # write only
    #         elif t < len(partial_mode_sequence) and t >= len(self.partial_mode_sequence):
    #             self.prog.getVarByName('d%d[%d]'%(t,partial_mode_sequence[t])).LB = 1.

    #     # update partial mode sequence
    #     self.partial_mode_sequence = partial_mode_sequence

    # def feedforward_relaxation(self, x0, partial_mode_sequence):

    #     # reset program
    #     self.prog.reset()

    #     # set up miqp
    #     self.set_type_auxliaries('C')
    #     self.update_mode_sequence(partial_mode_sequence)
    #     self.set_initial_condition(x0)

    #     # parameters
    #     self.prog.setParam('OutputFlag', 0)
    #     self.prog.setParam('Method', 0)
    #     self.prog.setParam('BarConvTol', 1.e-8)

    #     # run the optimization
    #     self.prog.optimize()

    #     return self.organize_result()

    def feedforward(self, x0):

        # reset program
        self.prog.reset()

        # set up miqp
        self._reset_binaries()
        self._set_type_binaries('B')
        self.set_initial_condition(x0)

        # parameters
        self.prog.setParam('OutputFlag', 1)
        # self.prog.setParam('Threads', 0)

        # run the optimization
        self.prog.optimize()
        # print self.prog.Runtime
        sol = self.organize_result()

        return sol['input'], sol['state'], sol['mode_sequence'], sol['objective']

    def feedback(self, x):
        u_feedforward = self.feedforward(x)[0]
        if u_feedforward is None:
            return None
        return u_feedforward[0]

    def organize_result(self):
    	sol = {'state': None, 'input': None, 'mode_sequence': None, 'objective': None, 'binaries': None}
        if self.prog.status in [2, 9, 11] and self.prog.SolCount > 0: # optimal, interrupted, time limit
            sol['state'] = [np.array([self.prog.getVarByName('x%d[%d]'%(t,k)).x for k in range(self.S.nx)]) for t in range(self.N+1)]
            sol['input'] = [np.array([self.prog.getVarByName('u%d[%d]'%(t,k)).x for k in range(self.S.nu)]) for t in range(self.N)]
            sol['binaries'] = [[self.prog.getVarByName('d%d[%d]'%(t,k)).x for k in range(self.S.nm)] for t in range(self.N)]
            sol['mode_sequence'] = [dt.index(max(dt)) for dt in sol['binaries']]
            sol['objective'] = self.prog.objVal
        return sol