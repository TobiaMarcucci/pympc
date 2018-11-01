# external imports
import numpy as np
import gurobipy as grb
from scipy.linalg import block_diag
from copy import copy

# internal inputs
from pympc.control.hybrid_benchmark.utils import (add_vars,
                                                  add_linear_inequality,
                                                  add_linear_equality,
                                                  add_rotated_socc,
                                                  add_stage_cost,
                                                  add_terminal_cost
                                                  )
from pympc.optimization.programs import linear_program
from build_mip_mld import bild_mip_mld
from build_mip_bm import bild_mip_bm
from build_mip_ch import bild_mip_ch
from build_mip_pf import bild_mip_pf

class HybridModelPredictiveController(object):

    def __init__(self, S, N, Q, R, P, X_N, method='BM', norm='two'):

        # check that all the domains are in hrep (if not use two inequalities)
        assert max([D.d.size for D in S.domains]) == 0

        # store inputs
        self.S = S
        self.N = N

        # build mixed integer program
        if method == 'MLD':
            self.prog = bild_mip_mld(S, N, Q, R, P, X_N, norm)
        elif method == 'BM':
            self.prog = bild_mip_bm(S, N, Q, R, P, X_N, norm)
        elif method == 'CH':
            self.prog = bild_mip_ch(S, N, Q, R, P, X_N, norm)
        elif method == 'PF':
            self.prog = bild_mip_pf(S, N, Q, R, P, X_N, norm)
        else:
            raise ValueError('unknown method ' + method + '.')

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
        self._set_type_binaries('C')
        self._reset_bounds_binaries()
        for k, v in identifier.items():
            self.prog.getVarByName('d%d[%d]'%k).LB = v
            self.prog.getVarByName('d%d[%d]'%k).UB = v
        self.prog.update()

    def _reset_bounds_binaries(self):
        for t in range(self.N):
            for k in range(self.S.nm):
                self.prog.getVarByName('d%d[%d]'%(t, k)).LB = 0.
                self.prog.getVarByName('d%d[%d]'%(t, k)).UB = grb.GRB.INFINITY

    def _set_type_binaries(self, d_type):
        for t in range(self.N):
            for k in range(self.S.nm):
                self.prog.getVarByName('d%d[%d]'%(t,k)).VType = d_type

    def reset_program(self, d_type):
        self.prog.reset()
        self._set_type_binaries(d_type)
        self._reset_bounds_binaries()
        self._reset_initial_condition()
        self.prog.setParam('Cutoff', grb.GRB.INFINITY)
        self.prog.update()

    def solve_relaxation(self, x0, identifier, objective_cutoff=np.inf):

        # reset program
        self.prog.reset()

        # set up miqp
        self.set_initial_condition(x0)
        if type(identifier) in [list, tuple]:
            identifier = {(t,m): 1. for t, m in enumerate(identifier)}
        self.set_binaries(identifier)
        
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
        solution = self.organize_result()
        objective = solution['objective']
        feasible = objective is not None
        if feasible:
            integer_feasible = all(np.isclose(max(dt), 1., atol=1.e-6) for dt in solution['binaries'])
        else:
            integer_feasible = None

        return feasible, objective, integer_feasible, solution

    def explore_most_ambiguous_node(self, node):

        t = np.argmin([max(dt) - min(dt) for dt in node.solution['binaries']])
        free_bin_id = range(self.S.nm)
        [free_bin_id.remove(k[1]) for k in node.identifier.keys() if k[0] == t]
        free_bin_val = [node.solution['binaries'][t][i] for i in free_bin_id]
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

    def explore_in_chronological_order(self, node):
        t = len(node.identifier)
        branches = [{(t,mode): 1.} for mode in np.argsort(node.solution['binaries'][t])]
        return branches

    def feedforward(self, x0):

        # reset program
        self.prog.reset()

        # set up miqp
        self._reset_bounds_binaries()
        self._set_type_binaries('B')
        self.set_initial_condition(x0)

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