# external imports
import numpy as np
import gurobipy as grb
from scipy.linalg import block_diag

# internal inputs
from pympc.control.hybrid_benchmark.utils import (add_vars,
                                                  add_linear_inequality,
                                                  add_linear_equality,
                                                  add_rotated_socc,
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
        prog = bild_mip_pf(x0, self.S, self.N, self.Q, self.R, self.P, self.X_N, self.norm)
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

def bild_mip_pf(x0, S, N, Q, R, P, X_N, norm):

    # shortcuts
    [nx, nu, nm] = [S.nx, S.nu, S.nm]

    # initialize program
    prog = grb.Model()
    obj = 0.
    
    # loop over time
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

        # infinity norm cost
        if norm == 'inf':

            # stage cost
            su = [add_vars(prog, 1, lb=[0.])[0] for i in range(nm)]
            obj += sum(su)
            for i in range(nm):
                add_linear_inequality(prog,  R.dot(v[i]), np.ones(R.shape[0]) * su[i])
                add_linear_inequality(prog, -R.dot(v[i]), np.ones(R.shape[0]) * su[i])
            if t == 0:
                obj += np.max(np.abs(Q.dot(x)))
            else:
                sx = [add_vars(prog, 1, lb=[0.])[0] for i in range(nm)]
                obj += sum(sx)
                for i in range(nm):
                    add_linear_inequality(prog,  Q.dot(y[i]), np.ones(Q.shape[0]) * sx[i])
                    add_linear_inequality(prog, -Q.dot(y[i]), np.ones(Q.shape[0]) * sx[i])

            # terminal cost
            if t == N - 1:
                sxN = [add_vars(prog, 1, lb=[0.])[0] for i in range(nm)]
                obj += sum(sxN)
                for i in range(nm):
                    add_linear_inequality(prog,  P.dot(z[i]), np.ones(P.shape[0]) * sxN[i])
                    add_linear_inequality(prog, -P.dot(z[i]), np.ones(P.shape[0]) * sxN[i])

        # one norm cost
        elif norm == 'one':

            # stage cost
            su = [add_vars(prog, R.shape[0], lb=[0.]*R.shape[0]) for i in range(nm)]
            obj += sum(sum(su))
            for i in range(nm):
                add_linear_inequality(prog,  R.dot(v[i]), su[i])
                add_linear_inequality(prog, -R.dot(v[i]), su[i])
            if t == 0:
                obj += sum(np.abs(Q.dot(x)))
            else:
                sx = [add_vars(prog, Q.shape[0], lb=[0.]*Q.shape[0]) for i in range(nm)]
                obj += sum(sum(sx))
                for i in range(nm):
                    add_linear_inequality(prog,  Q.dot(y[i]), sx[i])
                    add_linear_inequality(prog, -Q.dot(y[i]), sx[i])

            # terminal cost
            if t == N - 1:
                sxN = [add_vars(prog, P.shape[0], lb=[0.]*P.shape[0]) for i in range(nm)]
                obj += sum(sum(sxN))
                for i in range(nm):
                    add_linear_inequality(prog,  P.dot(z[i]), sxN[i])
                    add_linear_inequality(prog, -P.dot(z[i]), sxN[i])

        # two norm
        elif norm == 'two':

            # stage cost
            s = [add_vars(prog, 1, lb=[0.])[0] for i in range(nm)]
            obj += sum(s)
            if t == 0:
                for i in range(nm):
                    add_rotated_socc(prog, R, v[i], s[i], d[i])
                obj += .5*x.dot(Q.dot(x))
            elif t < N - 1:
                for i in range(nm):
                    QR = block_diag(Q, R)
                    yvi = np.concatenate((y[i], v[i]))
                    add_rotated_socc(prog, QR, yvi, s[i], d[i])

            # terminal cost
            else:
                for i in range(nm):
                    QRP = block_diag(Q, R, P)
                    yvzi = np.concatenate((y[i], v[i], z[i]))
                    add_rotated_socc(prog, QRP, yvzi, s[i], d[i])

        # recompose variables
        add_linear_equality(prog, u, sum(v))
        add_linear_equality(prog, x_next, sum(z))
        if t > 0:
            add_linear_equality(prog, x, sum(y))

        # constraints on the binaries
        prog.addConstr(sum(d) == 1.)

    # terminal constraint
    for i in range(nm):
        add_linear_inequality(prog, X_N.A.dot(z[i]), X_N.b * d[i])

    # set cost
    prog.setObjective(obj)

    return prog