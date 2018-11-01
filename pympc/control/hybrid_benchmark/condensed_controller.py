# external imports
import numpy as np
import gurobipy as grb
from scipy.linalg import block_diag
from copy import copy
from itertools import product

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.programs import linear_program
from pympc.control.hybrid_benchmark.utils import (add_vars,
                                                  add_linear_inequality,
                                                  add_linear_equality,
                                                  add_stage_cost,
                                                  add_terminal_cost
                                                  )

class HybridModelPredictiveController(object):

    def __init__(self, S, N, Q, R, P, X_N, T=1, norm='two'):

        assert N%T == 0

        # check that all the domains are in hrep (if not use two inequalities)
        assert max([D.d.size for D in S.domains]) == 0

        # store inputs
        self.S = S
        self.Pc = condense_dynamics(S, T)
        self.N = N
        self.T = T
        self.prog = bild_mip_condensed_ch(S, N, Q, R, P, X_N, norm, T, self.Pc)

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
        for t in range(self.N/self.T):
            for k in range(len(self.Pc)):
                self.prog.getVarByName('d%d[%d]'%(t, k)).LB = 0.
                self.prog.getVarByName('d%d[%d]'%(t, k)).UB = grb.GRB.INFINITY

    def _set_type_binaries(self, d_type):
        for t in range(self.N/self.T):
            for k in range(len(self.Pc)):
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
            sol['binaries'] = [[self.prog.getVarByName('d%d[%d]'%(t,k)).x for k in range(self.S.nm)] for t in range(self.N/self.T)]
            sol['mode_sequence'] = [dt.index(max(dt)) for dt in sol['binaries']]
            sol['objective'] = self.prog.objVal
        return sol


            # sol['input'] = [np.array([self.prog.getVarByName('u%d[%d]'%(t,k)).x for k in range(self.S.nu)]) for s,t in product(range(self.T), range(self.N/self.T))]
        #     sol['binaries'] = [[self.prog.getVarByName('d%d[%d]'%(t,k)).x for k in range(self.S.nm)] for t in range(self.N)]
        #     sol['mode_sequence'] = [dt.index(max(dt)) for dt in sol['binaries']]
        #     sol['objective'] = self.prog.objVal
        # return sol

# def bild_mip_condensed_ch(S, N, Q, R, P, X_N, norm, T):

#     # graph of the dynamics
#     Pc = condense_dynamics(S, T)
#     nm = len(Pc)

#     # initialize program
#     prog = grb.Model()
#     obj = 0.

#     # loop over the time horizon
#     for t in range(N/T):

#         # initial conditions
#         if t == 0:
#             x = [add_vars(prog, S.nx, name='x0_%d'%s) for s in range(T)]

#         # stage variables
#         else:
#             x = [x_next] + [add_vars(prog, S.nx, name='x%d_%d'%(t,s+1)) for s in range(T-1)]
#         x_next = add_vars(prog, S.nx, name='x%d_0'%(t+1))
#         u = [add_vars(prog, S.nu, name='u%d_%d'%(t,s)) for s in range(T)]
#         d = add_vars(prog, nm, lb=[0.]*nm, name='d%d'%t)

#         # auxiliary continuous variables for the convex-hull method
#         y = [[add_vars(prog, S.nx) for i in range(nm)] for s in range(T)]
#         z = [add_vars(prog, S.nx) for i in range(nm)]
#         v = [[add_vars(prog, S.nu) for i in range(nm)] for s in range(T)]

#         # constrained dynamics
#         for i in range(nm):
#             yvzi = np.concatenate([el for s in range(T) for el in [y[s][i], v[s][i]]] + [z[i]])
#             add_linear_inequality(prog, Pc[i].A.dot(yvzi), Pc[i].b * d[i])

#         # recompose the state and input (convex hull method)
#         [add_linear_equality(prog, x[s], sum(y[s])) for s in range(T)]
#         add_linear_equality(prog, x_next, sum(z))
#         [add_linear_equality(prog, u[s], sum(v[s])) for s in range(T)]

#         # constraints on the binaries
#         prog.addConstr(sum(d) == 1.)

#         # stage cost
#         obj += sum([add_stage_cost(prog, Q, R, x[s], u[s], norm) for s in range(T)])

#     # terminal constraint
#     # for i in range(nm):
#     #     add_linear_inequality(prog, X_N.A.dot(z[i]), X_N.b * d[i])
#     add_linear_inequality(prog, X_N.A.dot(x_next), X_N.b)

#     # terminal cost
#     obj += add_terminal_cost(prog, P, x_next, norm)
#     prog.setObjective(obj)

#     return prog

def bild_mip_condensed_ch(S, N, Q, R, P, X_N, norm, T, Pc):

    # graph of the dynamics
    nm = len(Pc)

    # initialize program
    prog = grb.Model()
    obj = 0.

    # loop over the time horizon
    for t in range(N/T):

        # initial conditions
        if t == 0:
            x = [add_vars(prog, S.nx, name='x%d'%s) for s in range(T)]

        # stage variables
        else:
            x = [x_next] + [add_vars(prog, S.nx, name='x%d'%(t*T+s+1)) for s in range(T-1)]
        x_next = add_vars(prog, S.nx, name='x%d'%((t+1)*T))
        u = [add_vars(prog, S.nu, name='u%d'%(t*T+s)) for s in range(T)]
        d = add_vars(prog, nm, lb=[0.]*nm, name='d%d'%t)

        # auxiliary continuous variables for the convex-hull method
        y = [[add_vars(prog, S.nx) for i in range(nm)] for s in range(T)]
        z = [add_vars(prog, S.nx) for i in range(nm)]
        v = [[add_vars(prog, S.nu) for i in range(nm)] for s in range(T)]

        # constrained dynamics
        for i in range(nm):
            # yvzi = np.zeros(0)
            # for s in range(T):
            #     yvzi = np.concatenate((yvzi, y[s][i], v[s][i]))
            # yvzi = np.concatenate((yvzi, z[i]))
            yvzi = np.concatenate([el for s in range(T) for el in [y[s][i], v[s][i]]] + [z[i]])
            add_linear_inequality(prog, Pc[i].A.dot(yvzi), Pc[i].b * d[i])

        # recompose the state and input
        [add_linear_equality(prog, x[s], sum(y[s])) for s in range(T)]
        add_linear_equality(prog, x_next, sum(z))
        [add_linear_equality(prog, u[s], sum(v[s])) for s in range(T)]

        # constraints on the binaries
        prog.addConstr(sum(d) == 1.)

        # stage cost
        obj += sum([add_stage_cost(prog, Q, R, x[s], u[s], norm) for s in range(T)])

    # terminal constraint
    # for i in range(nm):
    #     add_linear_inequality(prog, X_N.A.dot(z[i]), X_N.b * d[i])
    add_linear_inequality(prog, X_N.A.dot(x_next), X_N.b)

    # terminal cost
    obj += add_terminal_cost(prog, P, x_next, norm)
    prog.setObjective(obj)

    return prog


def graph_representation(S):
    P = []
    for i in range(S.nm):
        Di = S.domains[i]
        Si = S.affine_systems[i]
        Ai = np.vstack((
            np.hstack((Si.A, Si.B, -np.eye(S.nx))),
            np.hstack((-Si.A, -Si.B, np.eye(S.nx))),
            np.hstack((Di.A, np.zeros((Di.A.shape[0], S.nx))))
            ))
        bi = np.concatenate((-Si.c, Si.c, Di.b))
        P.append(Polyhedron(Ai, bi))
    return P

def concatenate_affine_dynamics(P1, P2, nx):
    n1, m1 = P1.A.shape
    n2, m2 = P2.A.shape
    A = np.vstack((
        np.hstack((P1.A, np.zeros((n1, m2-nx)))),
        np.hstack((np.zeros((n2, m1-nx)), P2.A))
        ))
    b = np.concatenate((P1.b, P2.b))
    return Polyhedron(A, b)

def condense_dynamics(S, T):
    P = graph_representation(S)
    Pc = copy(P)
    for t in range(1, T):
        Pc_next = []
        for i, Pci in enumerate(Pc):
            for j, Pj in enumerate(P):
                Pc_next_ij = concatenate_affine_dynamics(Pci, Pj, S.nx)
                if not Pc_next_ij.empty:
                    Pc_next.append(Pc_next_ij)
        Pc = Pc_next
    return Pc