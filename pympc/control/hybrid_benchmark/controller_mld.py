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

    def __init__(self, MLD, N, P, Q, R, C=None, c=None, D=None, d=None, tol=1.e-5):

        if C is None:
            C = np.eye(MLD.nx)
        if c is None:
            c = np.zeros(MLD.nx)
        if D is None:
            D = np.eye(MLD.nuc + MLD.nub)
        if d is None:
            d = np.zeros(MLD.nuc + MLD.nub)

        # store inputs
        self.MLD = MLD
        self.N = N
        self.P = P
        self.Q = Q
        self.R = R
        self.C = C
        self.c = c
        self.D = D
        self.d = d

        # checks

        assert self.P.shape[0] == self.P.shape[1]
        assert self.Q.shape[0] == self.Q.shape[1]
        assert self.R.shape[0] == self.R.shape[1]
        assert np.all(np.linalg.eigvals(self.P) > tol)
        assert np.all(np.linalg.eigvals(self.Q) > tol)
        assert np.all(np.linalg.eigvals(self.R) > tol)

        assert self.Q.shape[0] == self.C.shape[0]
        assert self.C.shape[1] == self.MLD.nx
        assert self.c.size == self.Q.shape[0]

        assert self.R.shape[0] == self.D.shape[0]
        assert self.D.shape[1] == self.MLD.nuc + self.MLD.nub
        assert self.d.size == self.R.shape[0]

        # build mixed integer program
        self.prog = self.bild_mip()

    def bild_mip(self):

        # initialize program
        prog = grb.Model()
        var = OrderedDict()
        con = OrderedDict()
        obj = 0.

        # initial state (initialized to zero)
        var['x_0'] = add_vars(prog, self.MLD.nx)
        con['initial_conditions'] = add_linear_equality(prog, var['x_0'], [0.]*self.MLD.nx)

        # loop over the time horizon
        for t in range(self.N):

            # stage variables
            var['uc_%d'%t] = add_vars(prog, self.MLD.nuc)
            var['ub_%d'%t] = add_vars(prog, self.MLD.nub)
            var['sc_%d'%t] = add_vars(prog, self.MLD.nsc)
            var['sb_%d'%t] = add_vars(prog, self.MLD.nsb)
            var['x_%d'%(t+1)] = add_vars(prog, self.MLD.nx)

            # bounds on the binaries
            con['lb_u_%d'%t] = add_linear_inequality(prog, [0.]*self.MLD.nub, var['ub_%d'%t])
            con['lb_s_%d'%t] = add_linear_inequality(prog, [0.]*self.MLD.nsb, var['sb_%d'%t])
            con['ub_u_%d'%t] = add_linear_inequality(prog, var['ub_%d'%t], [1.]*self.MLD.nub)
            con['ub_s_%d'%t] = add_linear_inequality(prog, var['sb_%d'%t], [1.]*self.MLD.nsb)

            # dynamics
            con['mld_dyn_%d'%t] = add_linear_equality(
                prog,
                var['x_%d'%(t+1)],
                self.MLD.A.dot(var['x_%d'%t]) +
                self.MLD.Buc.dot(var['uc_%d'%t]) + self.MLD.Bub.dot(var['ub_%d'%t]) +
                self.MLD.Bsc.dot(var['sc_%d'%t]) + self.MLD.Bsb.dot(var['sb_%d'%t]) +
                self.MLD.b
                )

            # constraints
            con['mld_ineq_%d'%t] = add_linear_inequality(
                prog,
                self.MLD.F.dot(var['x_%d'%t]) +
                self.MLD.Guc.dot(var['uc_%d'%t]) + self.MLD.Gub.dot(var['ub_%d'%t]) +
                self.MLD.Gsc.dot(var['sc_%d'%t]) + self.MLD.Gsb.dot(var['sb_%d'%t]),
                self.MLD.g
                )

            # stage cost
            u_t = np.concatenate((var['uc_%d'%t], var['ub_%d'%t]))
            var['y_%d'%t] = add_vars(prog, self.C.shape[0])
            con['y_%d'%t] = add_linear_equality(
                prog,
                var['y_%d'%t],
                self.C.dot(var['x_%d'%t]) + self.c
                )
            var['v_%d'%t] = add_vars(prog, self.D.shape[0])
            con['v_%d'%t] = add_linear_equality(
                prog,
                var['v_%d'%t],
                self.D.dot(u_t) + self.d
                )
            obj += var['y_%d'%t].dot(self.Q).dot(var['y_%d'%t]) + var['v_%d'%t].dot(self.R).dot(var['v_%d'%t])

        # terminal cost
        var['y_%d'%self.N] = add_vars(prog, self.C.shape[0])
        con['y_%d'%self.N] = add_linear_equality(
            prog,
            var['y_%d'%self.N],
            self.C.dot(var['x_%d'%self.N]) + self.c
            )
        obj += var['y_%d'%self.N].dot(self.P).dot(var['y_%d'%self.N])

        # set cost
        prog.setObjective(obj)

        # store variables and constraints
        self.variables = var
        self.constraints = con
        self.objective = obj

        return prog

    def set_initial_condition(self, x0):
        for k, xk in enumerate(x0):
            self.constraints['initial_conditions'][k].RHS = xk
        self.prog.update()

    def set_bounds_binaries(self, identifier):
        '''
        identifier is a dictionary with tuples as keys.
        a key is in the form ('u',22,4) where
        'u' are binary inputs, 's' are binary slacks
        22 is the time step
        4 denotes the 4th element of u
        '''
        self.reset_bounds_binaries()
        for k, v in identifier.items():
            self.constraints['lb_%s_%d'%k[:2]][k[2]].RHS = - v # in the form -sb <= -lb
            self.constraints['ub_%s_%d'%k[:2]][k[2]].RHS = v
        self.prog.update()

    def reset_bounds_binaries(self):
        for t in range(self.N):
            for k in range(self.MLD.nub):
                self.constraints['lb_u_%d'%t][k].RHS = 0.
                self.constraints['ub_u_%d'%t][k].RHS = 1.
            for k in range(self.MLD.nsb):
                self.constraints['lb_s_%d'%t][k].RHS = 0.
                self.constraints['ub_s_%d'%t][k].RHS = 1.
        self.prog.update()

    def set_type_binaries(self, d_type):
        for t in range(self.N):
            for k in range(self.MLD.nub):
                self.variables['ub_%d'%t][k].VType = d_type
            for k in range(self.MLD.nsb):
                self.variables['sb_%d'%t][k].VType = d_type
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
        sol = self.organize_result()

        # integer feasibility
        integer_feasible = sol['feasible'] and len(identifier) == self.N*(self.MLD.nub+self.MLD.nsb)

        self.test_solution(x0, identifier, sol)

        return sol['feasible'], sol['objective'], integer_feasible, sol

    def organize_result(self):

        # initialize solution
        sol = {}

        # optimal
        if self.prog.status == 2:
            sol['feasible'] = True
            sol['objective'] = self.prog.objVal
            sol['primal'] = OrderedDict()
            for k, v in self.variables.items():
                sol['primal'][k] = np.array([vk.x for vk in v])
            sol['dual'] = OrderedDict()
            for k, v in self.constraints.items():
                sol['dual'][k] = - np.array([vk.Pi for vk in v])

        #infeasible, infeasible_or_unbounded, or numeric_errors
        elif self.prog.status in [3, 4, 12]:
            sol['feasible'] = False
            sol['objective'] = None
            sol['primal'] = None
            sol['dual'] = self.get_farkas_proof()

        return sol

    def get_farkas_proof(self):

        # rerun the optimization
        self.prog.setParam('InfUnbdInfo', 1)
        self.prog.setObjective(0.)
        self.prog.optimize()

        # be sure that the problem is actually infeasible
        assert self.prog.status == 3

        # retrieve the proof of infeasibility
        farkas_proof = OrderedDict()
        i = 0
        for k, v in self.constraints.items():
            farkas_proof[k] = np.array(self.prog.FarkasDual[i:i+len(v)])
            i += len(v)

        # reset objective
        self.prog.setObjective(self.objective)

        return farkas_proof

    def explore_in_chronological_order(self, node):

        # idices of the last binary fixed in time
        t = max([0] + [k[1] for k in node.identifier.keys()])
        index_u = max([0] + [k[2]+1 for k in node.identifier.keys() if k[:2] == ('u',t)])
        index_s = max([0] + [k[2]+1 for k in node.identifier.keys() if k[:2] == ('s',t)])

        # try to fix one more ub at time t
        if index_u < self.MLD.nub:
            branches = [{('u',t,index_u): 0.}, {('u',t,index_u): 1.}]

        # try to fix one more sb at time t
        elif index_s < self.MLD.nsb:
            branches = [{('s',t,index_s): 0.}, {('s',t,index_s): 1.}]

        # if everything is fixed at time t, move to time t+1
        else:
            if self.MLD.nub > 0:
                branches = [{('u',t+1,0): 0.}, {('u',t+1,0): 1.}]
            else:
                branches = [{('s',t+1,0): 0.}, {('s',t+1,0): 1.}]

        return branches

    def feedforward(self, x, **kwargs):
        def solver(identifier):
            return self.solve_relaxation(x, identifier)
        sol, optimal_leaves = branch_and_bound(
            solver,
            best_first,
            self.explore_in_chronological_order,
            **kwargs
        )
        return sol, optimal_leaves

    def test_solution(self, x0, identifier, sol, tol=1.e-5):

        # rename
        primal = sol['primal']
        dual = sol['dual']

        # # reorganize primal variables
        # x = [primal['x_%d'%t] for t in range(self.N+1)]
        # uc = [primal['uc_%d'%t] for t in range(self.N)]
        # ub = [primal['ub_%d'%t] for t in range(self.N)]
        # sc = [primal['sc_%d'%t] for t in range(self.N)]
        # sb = [primal['sb_%d'%t] for t in range(self.N)]
        # u = [np.concatenate((uc[t], ub[t], sc[t], sb[t])) for t in range(self.N)]
        # y = [primal['y_%d'%t] for t in range(self.N+1)]
        # v = [primal['v_%d'%t] for t in range(self.N)]

        # reorganize multipliers
        alpha = [dual['initial_conditions']] + [dual['mld_dyn_%d'%t] for t in range(self.N)]
        beta = [dual['mld_ineq_%d'%t] for t in range(self.N)]
        gamma = [dual['y_%d'%t] for t in range(self.N+1)]
        delta = [dual['v_%d'%t] for t in range(self.N)]
        pi_lb = [np.concatenate((dual['lb_u_%d'%t], dual['lb_s_%d'%t])) for t in range(self.N)]
        pi_ub = [np.concatenate((dual['ub_u_%d'%t], dual['ub_s_%d'%t])) for t in range(self.N)]

        # reorganize matrices
        B = np.hstack((self.MLD.Buc, self.MLD.Bub, self.MLD.Bsc, self.MLD.Bsb))
        G = np.hstack((self.MLD.Guc, self.MLD.Gub, self.MLD.Gsc, self.MLD.Gsb))
        D = np.hstack((self.D, np.zeros((self.D.shape[0], self.MLD.nsc+self.MLD.nsb))))
        W = np.vstack((
            np.hstack((np.zeros((self.MLD.nub,self.MLD.nuc)), np.eye(self.MLD.nub), np.zeros((self.MLD.nub,self.MLD.nsc)), np.zeros((self.MLD.nub,self.MLD.nsb)))),
            np.hstack((np.zeros((self.MLD.nsb,self.MLD.nuc)), np.zeros((self.MLD.nsb,self.MLD.nub)), np.zeros((self.MLD.nsb,self.MLD.nsc)), np.eye(self.MLD.nsb)))
            ))

        # reorganize bounds
        w_lb = [np.zeros(self.MLD.nub+self.MLD.nsb) for t in range(self.N)]
        w_ub = [np.ones(self.MLD.nub+self.MLD.nsb) for t in range(self.N)]
        for key, val in identifier.items():
            if key[0] == 'u':
                w_lb[key[1]][key[2]] = val
                w_ub[key[1]][key[2]] = val
            elif key[0] == 's':
                w_lb[key[1]][self.MLD.nub+key[2]] = val
                w_ub[key[1]][self.MLD.nub+key[2]] = val

        # # test primal
        # assert np.linalg.norm(x[0] - x0) < tol
        # assert np.linalg.norm(y[self.N] - self.C.dot(x[self.N]) - self.c) < tol
        # for t in range(self.N):
        #     assert np.linalg.norm(x[t+1] - self.MLD.A.dot(x[t]) - B.dot(u[t]) - self.c) < tol
        #     assert np.linalg.norm(y[t] - self.C.dot(x[t]) - self.c) < tol
        #     assert np.linalg.norm(v[t] - D.dot(u[t]) - self.d) < tol
        #     assert np.max(self.MLD.F.dot(x[t]) + G.dot(u[t]) - self.MLD.g) < tol
        #     assert np.max(w_lb[t] - W.dot(u[t])) < tol
        #     assert np.max(W.dot(u[t]) - w_ub[t]) < tol

        # # test complementarity
        # print identifier
        # for t in range(self.N):
        #     assert np.abs(beta[t].dot(self.MLD.F.dot(x[t]) + G.dot(u[t]) - self.MLD.g)) < tol
        #     assert np.abs(pi_lb[t].dot(w_lb[t] - W.dot(u[t]))) < tol
        #     assert np.abs(pi_ub[t].dot(W.dot(u[t]) - w_ub[t])) < tol

        # check sign duals
        assert np.min(np.vstack(beta)) > -tol
        assert np.min(np.vstack(pi_lb)) > -tol
        assert np.min(np.vstack(pi_ub)) > -tol

        # test stationarity wrt x_t
        for t in range(self.N):
            assert np.linalg.norm(alpha[t] - self.MLD.A.T.dot(alpha[t+1]) + self.MLD.F.T.dot(beta[t]) - self.C.T.dot(gamma[t])) < tol

        # test stationarity wrt x_N
        assert np.linalg.norm(alpha[self.N] - self.C.T.dot(gamma[self.N])) < tol

        # test stationarity wrt y_t
        if sol['feasible']:
            for t in range(self.N+1):
                assert np.linalg.norm(2.*self.Q.dot(primal['y_%d'%t]) + gamma[t]) < tol

        # test stationarity wrt u_t
        for t in range(self.N):
            assert np.linalg.norm(- B.T.dot(alpha[t+1]) + G.T.dot(beta[t]) - D.T.dot(delta[t]) + W.T.dot(pi_ub[t] - pi_lb[t])) < tol

        # test stationarity wrt v_t
        if sol['feasible']:
            for t in range(self.N):
                assert np.linalg.norm(2.*self.R.dot(primal['v_%d'%t]) + delta[t]) < tol

        # quadratic terms
        Qinv = np.linalg.inv(self.Q)
        Rinv = np.linalg.inv(self.R)
        Pinv = np.linalg.inv(self.P)
        obj = - .25 * gamma[self.N].dot(Pinv).dot(gamma[self.N])
        for t in range(self.N):
            obj -= .25 * gamma[t].dot(Qinv).dot(gamma[t])
            obj -= .25 * delta[t].dot(Rinv).dot(delta[t])

        # linear terms
        obj -= x0.dot(alpha[0])
        obj -= self.c.dot(gamma[self.N])
        for t in range(self.N):
            obj -= self.MLD.b.dot(alpha[t+1])
            obj -= self.MLD.g.dot(beta[t])
            obj -= self.c.dot(gamma[t])
            obj -= self.d.dot(delta[t])
            obj += w_lb[t].dot(pi_lb[t])
            obj -= w_ub[t].dot(pi_ub[t])

        if sol['feasible']:
            assert np.abs(sol['objective'] - obj) < tol
        else:
            assert obj > -tol


    # def feedforward_gurobi(self, x0):

    #     # reset program
    #     self.prog.reset()

    #     # set up miqp
    #     self.reset_bounds_binaries()
    #     self.set_type_binaries('B')
    #     self.set_initial_condition(x0)

    #     # parameters
    #     self.prog.setParam('OutputFlag', 1)

    #     # run the optimization
    #     self.prog.optimize()
    #     sol = self.organize_result()

    #     return sol

    # def feedback_gurobi(self, x0):
    #     u_feedforward = self.feedforward_gurobi(x0)[0]
    #     if u_feedforward is None:
    #         return None
    #     return u_feedforward[0]

    # def organize_result(self):

    #     sol = {
    #     'x': None,
    #     'uc': None,
    #     'ub': None,
    #     'sc': None,
    #     'sb': None,
    #     'objective': None
    #     }

    #     # primal solution
    #     if self.prog.status in [2, 9, 11] and self.prog.SolCount > 0: # optimal, interrupted, time limit

    #         # sol['x'] = [np.array([self.prog.getVarByName('x_%d[%d]'%(t,k)).x for k in range(self.MLD.nx)]) for t in range(self.N+1)]
    #         # sol['uc'] = [np.array([self.prog.getVarByName('uc_%d[%d]'%(t,k)).x for k in range(self.MLD.nuc)]) for t in range(self.N)]
    #         # sol['ub'] = [np.array([self.prog.getVarByName('ub_%d[%d]'%(t,k)).x for k in range(self.MLD.nub)]) for t in range(self.N)]
    #         # sol['sc'] = [np.array([self.prog.getVarByName('sc_%d[%d]'%(t,k)).x for k in range(self.MLD.nsc)]) for t in range(self.N)]
    #         # sol['sb'] = [np.array([self.prog.getVarByName('sb_%d[%d]'%(t,k)).x for k in range(self.MLD.nsb)]) for t in range(self.N)]
    #         # sol['objective'] = self.prog.objVal

    #         # primal solution
    #         sol['primal'] = {}

    #         for k, v in self.variables.items():
    #             sol['primal'][k] = np.array([vk.x for vk in v])

    #         # dual solution
    #         sol['dual'] = {}
    #         for k, v in self.constraints.items():
    #             sol['dual'][k] = np.array([vk.Pi for vk in v])

    #     return sol

    # def propagate_bounds(self, feasible_leaves, x0, u0):
    #     delta = .5 * (x0.dot(self.Q.dot(x0)) + u0.dot(self.R.dot(u0)))
    #     warm_start = []
    #     for leaf in feasible_leaves:
    #         identifier = self.get_new_identifier(leaf.identifier)
    #         lower_bound = leaf.lower_bound - delta
    #         warm_start.append(Node(None, identifier, lower_bound))
    #     return warm_start

    # def get_new_identifier(self, old_id):
    #     new_id = copy(old_id)
    #     for t in range(self.N):
    #         for i in range(self.S.nm):
    #             if new_id.has_key((t, i)):
    #                 if t == 0:
    #                     new_id.pop((t, i))
    #                 else:
    #                     new_id[(t-1, i)] = new_id.pop((t, i))
    #     return new_id
