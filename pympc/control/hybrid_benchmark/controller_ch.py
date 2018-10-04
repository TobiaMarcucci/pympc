# external imports
import numpy as np
import gurobipy as grb

# internal inputs
from pympc.control.hybrid_benchmark.utils import (add_vars,
                                                  add_linear_inequality,
                                                  add_linear_equality,
                                                  add_stage_cost,
                                                  add_terminal_cost
                                                  )

class HybridModelPredictiveController(object):

    def __init__(self, S, N, Q, R, P, X_N, norm='two'):

        # check that all the domains are in hrep (if not use two inequalities)
        assert max([D.d.size for D in S.domains]) == 0

        # store inputs
        self.S = S
        self.N = N
        self.Q = Q
        self.R = R
        self.P = P
        self.X_N = X_N
        self.norm = norm

    def feedforward(self, x0):

        # build micp
        prog = bild_mip_ch(x0, self.S, self.N, self.Q, self.R, self.P, self.X_N, self.norm)
        prog.setParam('OutputFlag', 0)

        # run the optimization
        prog.optimize()
        print 'runtime:', prog.Runtime
        sol = self.organize_result(prog, x0)

        return sol['input'], sol['state'], sol['mode_sequence'], sol['objective']

    def organize_result(self, prog, x0):
        sol = {'state': None, 'input': None, 'mode_sequence': None, 'objective': None, 'binaries': None}
        if prog.status in [2, 9, 11] and prog.SolCount > 0: # optimal, interrupted, time limit
            sol['state'] = [x0] + [np.array([prog.getVarByName('x%d[%d]'%(t,k)).x for k in range(self.S.nx)]) for t in range(1,self.N+1)]
            sol['input'] = [np.array([prog.getVarByName('u%d[%d]'%(t,k)).x for k in range(self.S.nu)]) for t in range(self.N)]
            sol['binaries'] = [[prog.getVarByName('d%d[%d]'%(t,k)).x for k in range(self.S.nm)] for t in range(self.N)]
            sol['mode_sequence'] = [dt.index(max(dt)) for dt in sol['binaries']]
            sol['objective'] = prog.objVal
        return sol

def bild_mip_ch(x0, S, N, Q, R, P, X_N, norm):

    # shortcuts
    [nx, nu, nm] = [S.nx, S.nu, S.nm]

    # initialize program
    prog = grb.Model()
    obj = 0.

    # loop over the time horizon
    for t in range(N):

        # optimization variables
        if t == 0:
            x = x0
        else:
            x = x_next
        x_next = add_vars(prog, nx, name='x%d'%(t+1))
        u = add_vars(prog, nu, name='u%d'%t)
        d = add_vars(prog, nm, lb=[0.]*nm, name='d%d'%t, vtype=grb.GRB.BINARY)

        # auxiliary continuous variables for the convex-hull method
        v = [add_vars(prog, nu) for i in range(nm)]
        z = [add_vars(prog, nx) for i in range(nm)]
        if t > 0:
            y = [add_vars(prog, nx) for i in range(nm)]

        # initial stage
        if t == 0:

            # constrained dynamics
            for i in range(nm):

                # enforce dynamics
                Si = S.affine_systems[i]
                add_linear_equality(prog, z[i] - Si.B.dot(v[i]), (Si.c + Si.A.dot(x)) * d[i])

                # enforce state and input constraints
                Di = S.domains[i]
                add_linear_inequality(prog, Di.A[:,nx:].dot(v[i]), (Di.b - Di.A[:,:nx].dot(x)) * d[i])

        # other stages
        else:
            
            # constrained dynamics
            for i in range(nm):

                # enforce dynamics
                Si = S.affine_systems[i]
                add_linear_equality(prog, z[i], Si.A.dot(y[i]) + Si.B.dot(v[i]) + Si.c * d[i])

                # enforce state and input constraints
                Di = S.domains[i]
                yvi = np.concatenate((y[i], v[i]))
                add_linear_inequality(prog, Di.A.dot(yvi), Di.b * d[i])

        # recompose variables
        add_linear_equality(prog, u, sum(v))
        add_linear_equality(prog, x_next, sum(z))
        if t > 0:
            add_linear_equality(prog, x, sum(y))

        # constraints on the binaries
        prog.addConstr(sum(d) == 1.)
        # prog.addSOS(grb.GRB.SOS_TYPE1, d, [1.]*d.size)

        # stage cost
        obj += add_stage_cost(prog, Q, R, x, u, norm)

    # terminal constraint
    for i in range(nm):
        add_linear_inequality(prog, X_N.A.dot(z[i]), X_N.b * d[i])

    # terminal cost
    obj += add_terminal_cost(prog, P, x_next, norm)
    prog.setObjective(obj)

    return prog