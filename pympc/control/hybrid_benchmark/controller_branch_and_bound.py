# external imports
import numpy as np
import gurobipy as grb
from scipy.linalg import block_diag
from copy import copy
from collections import OrderedDict

# internal inputs
from pympc.control.hybrid_benchmark.utils import (add_vars,
                                                  add_linear_inequality,
                                                  add_linear_equality
                                                  )
from pympc.control.hybrid_benchmark.branch_and_bound_with_warm_start import Node, branch_and_bound, best_first
from pympc.optimization.programs import linear_program

class HybridModelPredictiveController(object):

    def __init__(self, S, N, Q, R):

        # check that all the domains are in hrep (if not use two inequalities)
        assert max([D.d.size for D in S.domains]) == 0

        # store inputs
        self.S = S
        self.N = N
        self.Q = Q
        self.R = R

        # build mixed integer program
        self.prog = self.bild_mip()

    def bild_mip(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # initialize program
        prog = grb.Model()
        obj = 0.

        # initial state
        x_next = add_vars(prog, nx, name='x0')

        # loop over the time horizon
        for t in range(self.N):

            # stage variables
            x = x_next
            u = add_vars(prog, nu, name='u%d'%t)
            d = add_vars(prog, nm, lb=[0.]*nm, name='d%d'%t)
            x_next = add_vars(prog, nx, name='x%d'%(t+1))

            # constraints on the binaries
            prog.addConstr(sum(d) == 1.)

            # auxiliary continuous variables for the convex-hull method
            x_i = [add_vars(prog, nx) for i in range(nm)]
            u_i = [add_vars(prog, nu) for i in range(nm)]
            x_next_i = [add_vars(prog, nx) for i in range(nm)]

            # recompose variables
            add_linear_equality(prog, x, sum(x_i))
            add_linear_equality(prog, u, sum(u_i))
            add_linear_equality(prog, x_next, sum(x_next_i))

            # constrained dynamics
            for i in range(nm):

                # enforce dynamics
                Si = self.S.affine_systems[i]
                add_linear_equality(prog, x_next_i[i], Si.A.dot(x_i[i]) + Si.B.dot(u_i[i]) + Si.c * d[i])

                # enforce state and input constraints
                Di = self.S.domains[i]
                xu_i = np.concatenate((x_i[i], u_i[i]))
                add_linear_inequality(prog, Di.A.dot(xu_i), Di.b * d[i])

            # stage cost
            obj += .5 * (x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))

        # set cost
        prog.setObjective(obj)

        return prog

    def set_initial_condition(self, x0):
        for k in range(self.S.nx):
            self.prog.getVarByName('x0[%d]'%k).LB = x0[k]
            self.prog.getVarByName('x0[%d]'%k).UB = x0[k]
        self.prog.update()

    def set_bounds_binaries(self, identifier):
        if type(identifier) in [list, tuple]:
            identifier = OrderedDict([((t,m), 1.) for t, m in enumerate(identifier)])
        self.reset_bounds_binaries()
        for k, v in identifier.items():
            self.prog.getVarByName('d%d[%d]'%k).LB = v
            self.prog.getVarByName('d%d[%d]'%k).UB = v
        self.prog.update()

    def reset_bounds_binaries(self):
        for t in range(self.N):
            for k in range(self.S.nm):
                self.prog.getVarByName('d%d[%d]'%(t, k)).LB = 0.
                self.prog.getVarByName('d%d[%d]'%(t, k)).UB = 1.
        self.prog.update()

    def set_type_binaries(self, d_type):
        for t in range(self.N):
            for k in range(self.S.nm):
                self.prog.getVarByName('d%d[%d]'%(t,k)).VType = d_type
        self.prog.update()

    def solve_relaxation(self, x0, identifier):

        # reset program (gurobi does not try to use the last solve to warm start)
        self.prog.reset()

        # set up miqp
        self.set_type_binaries('C')
        self.set_initial_condition(x0)
        self.set_bounds_binaries(identifier)

        # parameters
        self.prog.setParam('OutputFlag', 0)

        # run the optimization
        self.prog.optimize()

        # elaborate result
        sol = self.organize_result()
        objective = sol['objective']
        feasible = objective is not None
        integer_feasible = feasible and len(identifier) == self.N

        return feasible, objective, integer_feasible, sol

    def explore_in_chronological_order(self, node):
        t = len(node.identifier)
        branches = [{(t,mode): 1.} for mode in np.argsort(node.extra_data['binaries'][t])]
        return branches

    def feedforward_gurobi(self, x0):

        # reset program
        self.prog.reset()

        # set up miqp
        self.reset_bounds_binaries()
        self.set_type_binaries('B')
        self.set_initial_condition(x0)

        # parameters
        self.prog.setParam('OutputFlag', 1)

        # run the optimization
        self.prog.optimize()
        sol = self.organize_result()

        return sol['input'], sol['state'], sol['mode_sequence'], sol['objective']

    def feedback_gurobi(self, x):
        u_feedforward = self.feedforward_gurobi(x)[0]
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

    def propagate_bounds(self, feasible_leaves, x0, u0):
        delta = .5 * (x0.dot(self.Q.dot(x0)) + u0.dot(self.R.dot(u0)))
        warm_start = []
        for leaf in feasible_leaves:
            identifier = self.get_new_identifier(leaf.identifier)
            lower_bound = leaf.lower_bound - delta
            warm_start.append(Node(None, identifier, lower_bound))
        return warm_start

    def get_new_identifier(self, old_id):
        new_id = copy(old_id)
        for t in range(self.N):
            for i in range(self.S.nm):
                if new_id.has_key((t, i)):
                    if t == 0:
                        new_id.pop((t, i))
                    else:
                        new_id[(t-1, i)] = new_id.pop((t, i))
        return new_id

    def feedforward(self, x, **kwargs):
        def solver(identifier):
            return self.solve_relaxation(x, identifier)
        sol, feasible_leaves = branch_and_bound(solver,
            best_first,
            self.explore_in_chronological_order,
            **kwargs
        )
        new_warm_start = self.propagate_bounds(feasible_leaves, sol['state'][0], sol['input'][0])
        return sol['input'], sol['state'], sol['mode_sequence'], sol['objective'], new_warm_start
