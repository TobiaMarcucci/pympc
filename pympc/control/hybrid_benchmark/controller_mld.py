# external imports
import numpy as np
import gurobipy as grb
from copy import copy, deepcopy
from operator import le, ge, eq

# internal inputs
from pympc.control.hybrid_benchmark.branch_and_bound_with_warm_start import Node, branch_and_bound, best_first

class GurobiModel(grb.Model):

    def __init__(self, **kwargs):
        super(GurobiModel, self).__init__(**kwargs)

    def add_variables(self, n, lb=None, **kwargs):
        if lb is None:
            lb = [-grb.GRB.INFINITY]*n
        x = self.addVars(n, lb=lb, **kwargs)
        self.update()
        return np.array([xi for xi in x.values()])

    def get_variables(self, name):
        v = []
        for i in range(self.NumVars):
            vi = self.getVarByName(name+'[%d]'%i)
            if vi:
                v.append(vi)
            else:
                break
        return np.array(v)

    def add_linear_constraints(self, x, operator, y, **kwargs):
        assert len(x) == len(y)
        c = self.addConstrs((operator(x[k],y[k]) for k in range(len(x))), **kwargs)
        self.update()
        return np.array([ci for ci in c.values()])

    def get_constraints(self, name):
        c = []
        for i in range(self.NumConstrs):
            ci = self.getConstrByName(name+'[%d]'%i)
            if ci:
                c.append(ci)
            else:
                break
        return np.array(c)

class HybridModelPredictiveController(object):

    def __init__(self, MLD, N, P, Q, R, C=None, c=None, D=None, d=None):

        # set default state and input selection
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
        [self.P, self.Q, self.R] = [P, Q, R]
        [self.C, self.c] = [C, c]
        [self.D, self.d] = [D, d]

        # build mixed integer program
        self.check_inputs()
        self.model = self.build_mip()

    def check_inputs(self, tol=1.e-5):

        # weight matrices
        assert self.P.shape[0] == self.P.shape[1]
        assert self.Q.shape[0] == self.Q.shape[1]
        assert self.R.shape[0] == self.R.shape[1]
        assert np.all(np.linalg.eigvals(self.P) > tol)
        assert np.all(np.linalg.eigvals(self.Q) > tol)
        assert np.all(np.linalg.eigvals(self.R) > tol)

        # state selection matrix
        assert self.Q.shape[0] == self.C.shape[0]
        assert self.C.shape[1] == self.MLD.nx
        assert self.c.size == self.Q.shape[0]

        # input selection matrix
        assert self.R.shape[0] == self.D.shape[0]
        assert self.D.shape[1] == self.MLD.nuc + self.MLD.nub
        assert self.d.size == self.R.shape[0]

    def build_mip(self):

        # initialize program
        model = GurobiModel()
        obj = 0.

        # initial state (initialized to zero)
        x_next = model.add_variables(self.MLD.nx, name='x_0')
        model.add_linear_constraints(x_next, eq, [0.]*self.MLD.nx, name='alpha_0')

        # loop over the time horizon
        for t in range(self.N):

            # stage variables
            x = x_next
            uc = model.add_variables(self.MLD.nuc, name='uc_%d'%t)
            ub = model.add_variables(self.MLD.nub, name='ub_%d'%t)
            sc = model.add_variables(self.MLD.nsc, name='sc_%d'%t)
            sb = model.add_variables(self.MLD.nsb, name='sb_%d'%t)
            x_next = model.add_variables(self.MLD.nx, name='x_%d'%(t+1))

            # bounds on the binaries
            # inequalities must be stated as expr <= num to get negative duals
            # note that num <= expr would be modified to expr => num
            # and would give positive duals
            model.add_linear_constraints(-ub, le, [0.]*self.MLD.nub, name='lbu_%d'%t)
            model.add_linear_constraints(-sb, le, [0.]*self.MLD.nsb, name='lbs_%d'%t)
            model.add_linear_constraints(ub, le, [1.]*self.MLD.nub, name='ubu_%d'%t)
            model.add_linear_constraints(sb, le, [1.]*self.MLD.nsb, name='ubs_%d'%t)

            # mld dynamics
            model.add_linear_constraints(
                x_next,
                eq,
                self.MLD.A.dot(x) +
                self.MLD.Buc.dot(uc) + self.MLD.Bub.dot(ub) +
                self.MLD.Bsc.dot(sc) + self.MLD.Bsb.dot(sb) +
                self.MLD.b,
                name='alpha_%d'%(t+1)
                )

            # mld constraints
            model.add_linear_constraints(
                self.MLD.F.dot(x) +
                self.MLD.Guc.dot(uc) + self.MLD.Gub.dot(ub) +
                self.MLD.Gsc.dot(sc) + self.MLD.Gsb.dot(sb),
                le,
                self.MLD.g,
                name='beta_%d'%t
                )

            # stage cost
            u = np.concatenate((uc, ub))
            y = model.add_variables(self.C.shape[0], name='y_%d'%t)
            v = model.add_variables(self.D.shape[0], name='v_%d'%t)
            model.add_linear_constraints(y, eq, self.C.dot(x) + self.c, name='gamma_%d'%t)
            model.add_linear_constraints(v, eq, self.D.dot(u) + self.d, name='delta_%d'%t)
            obj += y.dot(self.Q).dot(y) + v.dot(self.R).dot(v)

        # terminal cost
        y = model.add_variables(self.C.shape[0], name='y_%d'%self.N)
        model.add_linear_constraints(y, eq, self.C.dot(x_next) + self.c, name='gamma_%d'%self.N)
        obj += y.dot(self.P).dot(y)

        # set cost
        model.setObjective(obj)

        return model

    def set_initial_condition(self, x0):
        alpha_0 = self.model.get_constraints('alpha_0')
        for k, xk in enumerate(x0):
            alpha_0[k].RHS = xk
        self.model.update()

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
            # the minus is because constraints are stated as -u <= -lb
            self.model.get_constraints('lb%s_%d'%k[:2])[k[2]].RHS = - v
            self.model.get_constraints('ub%s_%d'%k[:2])[k[2]].RHS = v
        self.model.update()

    def reset_bounds_binaries(self):
        for t in range(self.N):
            lbu = self.model.get_constraints('lbu_%d'%t)
            ubu = self.model.get_constraints('ubu_%d'%t)
            lbs = self.model.get_constraints('lbs_%d'%t)
            ubs = self.model.get_constraints('ubs_%d'%t)
            for k in range(self.MLD.nub):
                lbu[k].RHS = 0.
                ubu[k].RHS = 1.
            for k in range(self.MLD.nsb):
                lbs[k].RHS = 0.
                ubs[k].RHS = 1.
        self.model.update()

    def set_type_binaries(self, var_type):
        for t in range(self.N):
            ub = self.model.get_variables('ub_%d'%t)
            sb = self.model.get_variables('sb_%d'%t)
            for k in range(self.MLD.nub):
                ub[k].VType = var_type
            for k in range(self.MLD.nsb):
                sb[k].VType = var_type
        self.model.update()

    def solve_relaxation(self, x0, identifier):

        # reset model (gurobi does not try to use the last solve to warm start)
        self.model.reset()

        # set up miqp
        self.set_type_binaries('C')
        self.set_initial_condition(x0)
        self.set_bounds_binaries(identifier)

        # parameters
        self.model.setParam('OutputFlag', 0)

        # run the optimization
        self.model.optimize()
        sol = self.organize_result()

        # integer feasibility
        all_binaries_fixed = len(identifier) == self.N*(self.MLD.nub+self.MLD.nsb)
        integer_feasible = sol['feasible'] and all_binaries_fixed

        # check that duals have correct signs
        self.test_solution(x0, identifier, sol)

        return sol['feasible'], sol['objective'], integer_feasible, sol

    def organize_result(self):

        # initialize solution
        sol = {}

        # if optimal
        if self.model.status == 2:
            sol['feasible'] = True
            sol['objective'] = self.model.objVal
            sol['primal'] = self.organize_primal()
            sol['dual'] = self.organize_dual(sol['feasible'])

        # if infeasible, infeasible_or_unbounded, numeric_errors, suboptimal
        elif self.model.status in [3, 4, 12, 13]:
            self.do_farkas_proof()
            sol['feasible'] = False
            sol['objective'] = None
            sol['primal'] = None
            sol['dual'] = self.organize_dual(sol['feasible'])
        else:
            raise ValueError('unknown model status %d.'%self.model.status)

        return sol

    def organize_primal(self):

        # initialize primal solution
        primal = {}

        # primal stage variables
        for l in ['x', 'uc', 'ub', 'sc', 'sb', 'y', 'v']:
            primal[l] = []
            for t in range(self.N):
                v = self.model.get_variables('%s_%d'%(l,t))
                primal[l].append(np.array([vi.x for vi in v]))

        # primal terminal variables
        for l in ['x', 'y']:
            v = self.model.get_variables('%s_%d'%(l,self.N))
            primal[l].append(np.array([vi.x for vi in v]))

        return primal

    def organize_dual(self, feasible):

        # initialize dual solution
        dual = {}

        # dual stage variables
        for l in ['alpha', 'beta', 'gamma', 'delta', 'lbu', 'ubu', 'lbs', 'ubs']:
            dual[l] = []
            for t in range(self.N):
                c = self.model.get_constraints('%s_%d'%(l,t))
                if feasible:
                    # weird! gurobi gives negative multipliers and positive farkas duals
                    dual[l].append(- np.array([ci.Pi for ci in c]))
                else:
                    dual[l].append(np.array([ci.FarkasDual for ci in c]))

        # dual terminal variables
        for l in ['alpha', 'gamma']:
            c = self.model.get_constraints('%s_%d'%(l,self.N))
            if feasible:
                dual[l].append(- np.array([ci.Pi for ci in c]))
            else:
                dual[l].append(np.array([ci.FarkasDual for ci in c]))

        return dual

    def do_farkas_proof(self):

        # copy objective
        obj = self.model.getObjective()

        # rerun the optimization with linear objective (only linear accepted for farkas proof)
        self.model.setParam('InfUnbdInfo', 1)
        self.model.setObjective(0.)
        self.model.optimize()

        # ensure new problem is actually infeasible
        assert self.model.status == 3

        # reset objective
        self.model.setObjective(obj)

    def get_bounds_on_binaries(self, identifier):
        w_lb = [np.zeros(self.MLD.nub+self.MLD.nsb) for t in range(self.N)]
        w_ub = [np.ones(self.MLD.nub+self.MLD.nsb) for t in range(self.N)]
        for key, val in identifier.items():
            if key[0] == 'u':
                w_lb[key[1]][key[2]] = val
                w_ub[key[1]][key[2]] = val
            elif key[0] == 's':
                w_lb[key[1]][self.MLD.nub+key[2]] = val
                w_ub[key[1]][self.MLD.nub+key[2]] = val
        return w_lb, w_ub


    def test_solution(self, x0, identifier, sol, tol=1.e-4):

        # rename
        MLD = self.MLD
        alpha = sol['dual']['alpha']
        beta = sol['dual']['beta']
        gamma = sol['dual']['gamma']
        delta = sol['dual']['delta']
        pi_lb = [np.concatenate((sol['dual']['lbu'][t], sol['dual']['lbs'][t])) for t in range(self.N)]
        pi_ub = [np.concatenate((sol['dual']['ubu'][t], sol['dual']['ubs'][t])) for t in range(self.N)]

        # reorganize matrices
        B = np.hstack((MLD.Buc, MLD.Bub, MLD.Bsc, MLD.Bsb))
        G = np.hstack((MLD.Guc, MLD.Gub, MLD.Gsc, MLD.Gsb))
        D = np.hstack((self.D, np.zeros((self.D.shape[0], MLD.nsc+MLD.nsb))))
        W = np.vstack((
            np.hstack((np.zeros((MLD.nub,MLD.nuc)), np.eye(MLD.nub), np.zeros((MLD.nub,MLD.nsc)), np.zeros((MLD.nub,MLD.nsb)))),
            np.hstack((np.zeros((MLD.nsb,MLD.nuc)), np.zeros((MLD.nsb,MLD.nub)), np.zeros((MLD.nsb,MLD.nsc)), np.eye(MLD.nsb)))
            ))

        # check sign duals
        assert np.min(np.vstack(beta)) > - tol
        assert np.min(np.vstack(pi_lb)) > - tol
        assert np.min(np.vstack(pi_ub)) > - tol

        # test stationarity wrt x_t
        for t in range(self.N):
            res = alpha[t] - MLD.A.T.dot(alpha[t+1]) + MLD.F.T.dot(beta[t]) - self.C.T.dot(gamma[t])
            assert np.linalg.norm(res) < tol

        # test stationarity wrt x_N
        res = alpha[self.N] - self.C.T.dot(gamma[self.N])
        assert np.linalg.norm(res) < tol

        # test stationarity wrt y_t
        if sol['feasible']:
            for t in range(self.N+1):
                res = 2.*self.Q.dot(sol['primal']['y'][t]) + gamma[t]
                assert np.linalg.norm(res) < tol

        # test stationarity wrt u_t
        for t in range(self.N):
            res = -B.T.dot(alpha[t+1]) + G.T.dot(beta[t]) - D.T.dot(delta[t]) + W.T.dot(pi_ub[t]-pi_lb[t])
            assert np.linalg.norm(res) < tol

        # test stationarity wrt v_t
        if sol['feasible']:
            for t in range(self.N):
                res = 2.*self.R.dot(sol['primal']['v'][t]) + delta[t]
                assert np.linalg.norm(res) < tol

        # test objective
        obj = self.evaluate_dual_solution(identifier, sol['dual'], x0)
        if sol['feasible']:
            assert np.abs(sol['objective'] - obj) < tol
        else:
            assert np.linalg.norm(np.concatenate(gamma)) < tol
            assert np.linalg.norm(np.concatenate(delta)) < tol
            assert obj > - tol

    def evaluate_dual_solution(self, identifier, dual, x0):

        # quadratic terms
        Qinv = np.linalg.inv(self.Q)
        Rinv = np.linalg.inv(self.R)
        Pinv = np.linalg.inv(self.P)
        obj = - .25 * dual['gamma'][self.N].dot(Pinv).dot(dual['gamma'][self.N])
        for t in range(self.N):
            obj -= .25 * dual['gamma'][t].dot(Qinv).dot(dual['gamma'][t])
            obj -= .25 * dual['delta'][t].dot(Rinv).dot(dual['delta'][t])

        # linear terms
        w_lb, w_ub = self.get_bounds_on_binaries(identifier)
        obj -= x0.dot(dual['alpha'][0])
        obj -= self.c.dot(dual['gamma'][self.N])
        for t in range(self.N):
            obj -= self.MLD.b.dot(dual['alpha'][t+1])
            obj -= self.MLD.g.dot(dual['beta'][t])
            obj -= self.c.dot(dual['gamma'][t])
            obj -= self.d.dot(dual['delta'][t])
            obj += w_lb[t].dot(np.concatenate((dual['lbu'][t], dual['lbs'][t])))
            obj -= w_ub[t].dot(np.concatenate((dual['ubu'][t], dual['ubs'][t])))

        return obj

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

    def generate_warm_start(self, leaves, x_0, uc_0, ub_0, sc_0, sb_0, e):

        warm_start = []
        u_0 = np.concatenate((uc_0, ub_0))
        s_0 = np.concatenate((sc_0, sb_0))

        for l in leaves:
            if self.is_leaf_propagable(l.identifier, ub_0, sb_0):
                new_dual = self.propagate_dual_solution(l.extra_data['dual'])
                new_extra_data = {'dual': new_dual,'feasible':True, 'objective':None, 'primal':None}
                new_identifier = self.get_new_identifier(l.identifier)
                lams = self.get_lambdas(l.identifier, l.extra_data['dual'], x_0, u_0, s_0, e)
                if l.feasible:
                    new_lower_bound = l.lower_bound + sum(lams)
                else:
                    theta_0_lb = self.evaluate_dual_solution(l.identifier, l.extra_data['dual'], x_0)
                    still_infeasible = e.dot(l.extra_data['dual']['alpha'][1]) < theta_0_lb + lams[2]
                    if still_infeasible:
                        new_lower_bound = np.inf
                    else:
                        new_lower_bound = - np.inf
                warm_start.append(Node(None, new_identifier, new_lower_bound, new_extra_data))

        return warm_start

    @staticmethod
    def is_leaf_propagable(identifier, ub_0, sb_0):
        propagable = True
        for key, value in identifier.items():
            if key[1] == 0:
                if key[0] == 'u' and not np.isclose(value, ub_0[key[2]]):
                    propagable = False
                    break
                if key[0] == 's' and not np.isclose(value, sb_0[key[2]]):
                    propagable = False
                    break
        return propagable

    @staticmethod
    def propagate_dual_solution(dual):
        new_dual = deepcopy(dual)
        for l in ['alpha', 'beta', 'gamma', 'delta', 'lbu', 'ubu', 'lbs', 'ubs']:
            new_dual[l].append(0.*new_dual[l][0])
            del new_dual[l][0]
        return new_dual

    def get_lambdas(self, identifier, dual, x_0, u_0, s_0, e, tol=1.e-5):

        # invers of the weights
        Pinv = np.linalg.inv(self.P)
        Qinv = np.linalg.inv(self.Q)
        Rinv = np.linalg.inv(self.R)

        # lambda 1
        y_0 = self.C.dot(x_0) + self.c
        v_0 = self.D.dot(u_0) + self.d
        lam1 = - y_0.dot(self.Q).dot(y_0) - v_0.dot(self.R).dot(v_0)

        # lambda 2
        gamma_N = dual['gamma'][self.N]
        lam2 = .25*gamma_N.dot(Pinv - Qinv).dot(gamma_N)

        # lambda 3
        w_lb, w_ub = self.get_bounds_on_binaries(identifier)
        us_0 = np.concatenate((u_0, s_0))
        lbus_0 = np.concatenate((dual['lbu'][0], dual['lbs'][0]))
        ubus_0 = np.concatenate((dual['ubu'][0], dual['ubs'][0]))
        lam3 = -(self.MLD.F.dot(x_0) + self.MLD.G.dot(us_0) - self.MLD.g).dot(dual['beta'][0])
        lam3 -= (w_lb[0] - self.MLD.W.dot(us_0)).dot(lbus_0)
        lam3 -= (self.MLD.W.dot(us_0) - w_ub[0]).dot(ubus_0)
        assert lam3 > -tol

        # lambda 4
        lam4_x = .5*dual['gamma'][0] + self.Q.dot(y_0)
        lam4_u = .5*dual['delta'][0] + self.R.dot(v_0)
        lam4 = lam4_x.dot(Qinv).dot(lam4_x) + lam4_u.dot(Rinv).dot(lam4_u)

        # lambda 5
        lam5 = - e.dot(dual['alpha'][1])

        return lam1, lam2, lam3, lam4, lam5

    @staticmethod
    def get_new_identifier(identifier):
        new_identifier = {}
        for k, v in identifier.items():
            if k[1] > 0:
                new_identifier[(k[0],k[1]-1,k[2])] = v
        return new_identifier

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


        