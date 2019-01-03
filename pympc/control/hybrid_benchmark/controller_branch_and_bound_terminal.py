# external imports
import numpy as np
import gurobipy as grb
from scipy.linalg import block_diag
from copy import copy

# internal inputs
from pympc.control.hybrid_benchmark.utils import (add_vars,
                                                  add_linear_inequality,
                                                  add_linear_equality,
                                                  add_stage_cost,
                                                  add_terminal_cost
                                                  )
from pympc.optimization.programs import linear_program

class HybridModelPredictiveController(object):

    def __init__(self, S, N, Q, R, P, X_N, method='BM', norm='two'):

        # check that all the domains are in hrep (if not use two inequalities)
        assert max([D.d.size for D in S.domains]) == 0

        # store inputs
        self.S = S
        self.N = N
        self.Q = Q
        self.R = R
        self.P = P
        self.X_N = X_N

        # build mixed integer program
        self.prog = self.bild_mip()

        # bounds on the terminal state
        self.x_N_min, self.x_N_max = self.bounds_terminal_state()

    def bild_mip(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

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
            v = [add_vars(prog, nu) for i in range(nm)]
            z = [add_vars(prog, nx) for i in range(nm)]

            # constrained dynamics
            for i in range(nm):

                # enforce dynamics
                Si = self.S.affine_systems[i]
                add_linear_equality(prog, z[i], Si.A.dot(y[i]) + Si.B.dot(v[i]) + Si.c * d[i])

                # enforce state and input constraints
                Di = self.S.domains[i]
                yvi = np.concatenate((y[i], v[i]))
                add_linear_inequality(prog, Di.A.dot(yvi), Di.b * d[i])

            # recompose variables
            add_linear_equality(prog, x, sum(y))
            add_linear_equality(prog, u, sum(v))
            add_linear_equality(prog, x_next, sum(z))

            # constraints on the binaries
            prog.addConstr(sum(d) == 1.)

            # stage cost
            obj += add_stage_cost(prog, self.Q, self.R, x, u, 'two')

        # auxiliary variable to drop the terminal stage
        xN_delta = add_vars(prog, nx, lb=[0.]*nx, ub=[0.]*nx, name='xN_delta')

        # terminal constraint
        add_linear_inequality(prog, self.X_N.A.dot(xN_delta - x_next), self.X_N.b)

        # terminal cost
        obj += add_terminal_cost(prog, self.P, xN_delta - x_next, 'two')
        prog.setObjective(obj)

        return prog

    def bounds_terminal_state(self):
        x_min = np.full(self.S.nx, np.inf)
        x_max = - np.full(self.S.nx, np.inf)
        for k in range(self.S.nx):
            for i in range(self.S.nm):

                Si = self.S.affine_systems[i]
                Di = self.S.domains[i]
                f = np.concatenate((Si.A[k], Si.B[k]))

                # update lower bound
                sol = linear_program(f, Di.A, Di.b)
                x_min[k] = min(x_min[k], sol['min'] + Si.c[k])

                # update upper bound
                sol = linear_program(-f, Di.A, Di.b)
                x_max[k] = max(x_max[k], - sol['min'] + Si.c[k])

        return x_min, x_max

    def set_initial_condition(self, x0):
        for k in range(self.S.nx):
            self.prog.getVarByName('x0[%d]'%k).LB = x0[k]
            self.prog.getVarByName('x0[%d]'%k).UB = x0[k]
        self.prog.update()

    def drop_terminal_conditions(self):
        for k in range(self.S.nx):
            self.prog.getVarByName('xN_delta[%d]'%k).LB = self.x_N_min[k]
            self.prog.getVarByName('xN_delta[%d]'%k).UB = self.x_N_max[k]
        self.prog.update()

    def enforce_terminal_conditions(self):
        for k in range(self.S.nx):
            self.prog.getVarByName('xN_delta[%d]'%k).LB = 0.
            self.prog.getVarByName('xN_delta[%d]'%k).UB = 0.
        self.prog.update()

    def set_bounds_binaries(self, identifier):
        if type(identifier) in [list, tuple]:
            identifier = {(t,m): 1. for t, m in enumerate(identifier)}
        self.reset_bounds_binaries()
        for k, v in identifier.items():
            self.prog.getVarByName('d%d[%d]'%k).LB = v
            self.prog.getVarByName('d%d[%d]'%k).UB = v
        self.prog.update()

    def reset_bounds_binaries(self):
        for t in range(self.N):
            for k in range(self.S.nm):
                self.prog.getVarByName('d%d[%d]'%(t, k)).LB = 0.
                self.prog.getVarByName('d%d[%d]'%(t, k)).UB = grb.GRB.INFINITY

    def set_type_binaries(self, d_type):
        for t in range(self.N):
            for k in range(self.S.nm):
                self.prog.getVarByName('d%d[%d]'%(t,k)).VType = d_type

    def solve_relaxation(self, x0, identifier, objective_cutoff=np.inf):

        # reset program (gurobi does not try to use the last solve to warm start)
        self.prog.reset()

        # set up miqp
        self.set_type_binaries('C')
        self.set_initial_condition(x0)
        self.set_bounds_binaries(identifier)

        # drop or enforce the terminal conditions
        if len(identifier) < self.N:
            self.drop_terminal_conditions()
        else:
            self.enforce_terminal_conditions()

        # parameters
        self.prog.setParam('OutputFlag', 0)
        self.prog.setParam('Cutoff', grb.GRB.INFINITY)
        if not np.isinf(objective_cutoff):
            self.prog.setParam('Cutoff', objective_cutoff) # DO NOT SET Cutoff TO np.inf! IT IS DIFFERENT FROM SETTING IT TO grb.GRB.INFINITY!!
        # self.prog.setParam('Method', 0)
        # self.prog.setParam('BarConvTol', 1.e-8)

        # run the optimization
        self.prog.optimize()

        # elaborate result
        sol = self.organize_result()
        objective = sol['objective']

        # double check some stuff
        tol = np.ones(self.S.nx)*1.e-3
        if objective is not None:
            assert np.all(sol['state'][-1] <= self.x_N_max + tol)
            assert np.all(sol['state'][-1] >= self.x_N_min - tol)
            if len(identifier) < self.N:
                xN_delta = np.array([self.prog.getVarByName('xN_delta[%d]'%k).x for k in range(self.S.nx)])
                assert np.allclose(sol['state'][-1], xN_delta, rtol=1.e-3)

        feasible = objective is not None
        integer_feasible = feasible and len(identifier) == self.N

        return feasible, objective, integer_feasible, sol

    def explore_in_chronological_order(self, node):
        t = len(node.identifier)
        branches = [{(t,mode): 1.} for mode in np.argsort(node.extra_data['binaries'][t])]
        return branches

    def feedforward(self, x0):

        # reset program
        self.prog.reset()

        # set up miqp
        self.reset_bounds_binaries()
        self.set_type_binaries('B')
        self.set_initial_condition(x0)
        self.enforce_terminal_conditions()

        # parameters
        self.prog.setParam('OutputFlag', 1)

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