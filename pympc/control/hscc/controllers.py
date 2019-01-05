# external imports
import numpy as np
import gurobipy as grb

# internal inputs
from build_mip_mld import bild_mip_mld
from build_mip_bm import bild_mip_bm
from build_mip_ch import bild_mip_ch
from build_mip_pf import bild_mip_pf

class HybridModelPredictiveController(object):

    def __init__(self, S, N, Q, R, P, X_N, method='ch', norm='two'):

        # check that all the domains are in hrep (if not use two inequalities)
        assert max([D.d.size for D in S.domains]) == 0

        # store inputs
        self.S = S
        self.N = N

        # build mixed integer program

        # mld equivalent (Bemporad and Morari 1999)
        if method == 'mld':
            self.prog = bild_mip_mld(S, N, Q, R, P, X_N, norm)

        # big-m method (Vielma 2015)
        elif method == 'bm':
            self.prog = bild_mip_bm(S, N, Q, R, P, X_N, norm)

        # convex hull method
        elif method == 'ch':
            self.prog = bild_mip_ch(S, N, Q, R, P, X_N, norm)

        # perspective formulation
        elif method == 'pf':
            self.prog = bild_mip_pf(S, N, Q, R, P, X_N, norm)

        # else raise error
        else:
            raise ValueError('unknown method ' + method + '.')

    def set_initial_condition(self, x0):
        for k in range(self.S.nx):
            self.prog.getVarByName('x0[%d]'%k).LB = x0[k]
            self.prog.getVarByName('x0[%d]'%k).UB = x0[k]
        self.prog.update()

    def set_indicator_bounds(self, identifier):
        '''
        `identifier` is a dictionary of dictionaries.
        `identifier[time][mode]`, if the key exists, gives the value for d[time][mode].
        '''
        self._set_indicator_type('C')
        self._reset_indicator_bounds()
        for k, v in identifier.items():
            self.prog.getVarByName('d%d[%d]'%k).LB = v
            self.prog.getVarByName('d%d[%d]'%k).UB = v
        self.prog.update()

    def _reset_indicator_bounds(self):
        for t in range(self.N):
            for k in range(self.S.nm):
                self.prog.getVarByName('d%d[%d]'%(t, k)).LB = 0.
                self.prog.getVarByName('d%d[%d]'%(t, k)).UB = grb.GRB.INFINITY
        self.prog.update()

    def _set_indicator_type(self, d_type):
        for t in range(self.N):
            for k in range(self.S.nm):
                self.prog.getVarByName('d%d[%d]'%(t,k)).VType = d_type
        self.prog.update()

    def solve_relaxation(self, x0, identifier, gurobi_options={}):

        # set up program
        self.prog.reset()
        self.set_initial_condition(x0)
        if type(identifier) in [list, tuple]:
            identifier = {(t,m): 1. for t, m in enumerate(identifier)}
        self.set_indicator_bounds(identifier)

        # set parameters
        for parameter, value in gurobi_options.items():
            self.prog.setParam(parameter, value)

        # run the optimization
        self.prog.optimize()
        
        return self.organize_result()

    def feedforward(self, x0, gurobi_options={}):

        # set up miqp
        self.prog.reset()
        self._reset_indicator_bounds()
        self._set_indicator_type('B')
        self.set_initial_condition(x0)

        # set parameters
        for parameter, value in gurobi_options.items():
            self.prog.setParam(parameter, value)

        # run the optimization
        self.prog.optimize()
        # print self.prog.Runtime

        return self.organize_result()

    def feedback(self, x0, gurobi_options={}):
        u_feedforward = self.feedforward(x0, gurobi_options)[0]
        if u_feedforward is None:
            return None
        return u_feedforward[0]

    def organize_result(self):

        # initialize solution to None
        u = None
        x = None
        ms = None
        obj = None

        # store result if optimal, interrupted, or time limit
        if self.prog.status in [2, 9, 11] and self.prog.SolCount > 0:

            # input sequence
            u = [np.array([self.prog.getVarByName('u%d[%d]'%(t,k)).x for k in range(self.S.nu)]) for t in range(self.N)]

            # state trajectyory
            x = [np.array([self.prog.getVarByName('x%d[%d]'%(t,k)).x for k in range(self.S.nx)]) for t in range(self.N+1)]

            # mode sequence
            d = [[self.prog.getVarByName('d%d[%d]'%(t,k)).x for k in range(self.S.nm)] for t in range(self.N)]
            ms = [dt.index(max(dt)) for dt in d]

            # optimal value function
            obj = self.prog.objVal

        return u, x, ms, obj