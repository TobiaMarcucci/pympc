# external imports
import numpy as np
import gurobipy as grb

# internal inputs
from pympc.control.hybrid_benchmark.utils import graph_representation, big_m, add_vars, add_linear_inequality, add_linear_equality

class HybridModelPredictiveController(object):

    def __init__(self, S, N, Q, R, P, X_N, method='big_m'):

        # store inputs
        self.S = S
        self.N = N
        self.Q = Q
        self.R = R
        self.P = P
        self.X_N = X_N

        # mpMIQP
        self.prog = self.build_mpmiqp(method)
        self.partial_mode_sequence = []

    def build_mpmiqp(self, method):
        if method == 'big_m':
            return self.bild_miqp_bigm()
        if method == 'improved_big_m':
            return self.bild_miqp_improved_bigm()
        if method == 'convex_hull':
            return self.bild_miqp_convex_hull()
        else:
            raise ValueError('unknown method ' + method + '.')

    def bild_miqp_bigm(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # graph of the dynamics and big-Ms
        P = graph_representation(self.S)
        _, mi = big_m(P)

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
            for i in range(nm):
                add_linear_inequality(prog, P[i].A.dot(xux), P[i].b + mi[i]*(1.-d[i]))

            # constraints on the binaries
            cons = prog.addConstr(sum(d) == 1.)

            # stage cost to the objective
            obj += .5 * (x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))

        # terminal constraint
        add_linear_inequality(prog, self.X_N.A.dot(x_next), self.X_N.b)

        # terminal cost
        obj += .5 * x_next.dot(self.P).dot(x_next)
        prog.setObjective(obj)

        return prog

    def bild_miqp_improved_bigm(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # graph of the dynamics and big-Ms
        P = graph_representation(self.S)
        m, _ = big_m(P)

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
            for i in range(nm):
                sum_mi = sum(m[i][j] * d[j] for j in range(self.S.nm) if j != i)
                add_linear_inequality(prog, P[i].A.dot(xux), P[i].b + sum_mi)

            # constraints on the binaries
            cons = prog.addConstr(sum(d) == 1.)

            # stage cost to the objective
            obj += .5 * (x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))

        # terminal constraint
        add_linear_inequality(prog, self.X_N.A.dot(x_next), self.X_N.b)

        # terminal cost
        obj += .5 * x_next.dot(self.P).dot(x_next)
        prog.setObjective(obj)

        return prog

    def bild_miqp_convex_hull(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # graph of the dynamics and big-Ms
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
            cons = prog.addConstr(sum(d) == 1.)

            # stage cost to the objective
            obj += .5 * (x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))

        # terminal constraint
        add_linear_inequality(prog, self.X_N.A.dot(x_next), self.X_N.b)

        # terminal cost
        obj += .5 * x_next.dot(self.P).dot(x_next)
        prog.setObjective(obj)

        return prog

    def set_initial_condition(self, x0):
        for k in range(self.S.nx):
            self.prog.getVarByName('x0[%d]'%k).LB = x0[k,0]
            self.prog.getVarByName('x0[%d]'%k).UB = x0[k,0]
        self.prog.update()

    def reset_program(self):
        self.set_type_auxliaries('C')
        self.update_mode_sequence([])
        for k in range(self.S.nx):
            self.prog.getVarByName('x0[%d]'%k).LB = -grb.GRB.INFINITY
            self.prog.getVarByName('x0[%d]'%k).UB = grb.GRB.INFINITY
        self.prog.update()

    def update_mode_sequence(self, partial_mode_sequence):

        # loop over the time horizon
        for t in range(self.N):

            # earse and write
            if t < len(partial_mode_sequence) and t < len(self.partial_mode_sequence):
                if partial_mode_sequence[t] != self.partial_mode_sequence[t]:
                    self.prog.getVarByName('d%d[%d]'%(t,self.partial_mode_sequence[t])).LB = 0.
                    self.prog.getVarByName('d%d[%d]'%(t,partial_mode_sequence[t])).LB = 1.

            # erase only
            elif t >= len(partial_mode_sequence) and t < len(self.partial_mode_sequence):
                self.prog.getVarByName('d%d[%d]'%(t,self.partial_mode_sequence[t])).LB = 0.

            # write only
            elif t < len(partial_mode_sequence) and t >= len(self.partial_mode_sequence):
                self.prog.getVarByName('d%d[%d]'%(t,partial_mode_sequence[t])).LB = 1.

        # update partial mode sequence
        self.partial_mode_sequence = partial_mode_sequence

    def feedforward_relaxation(self, x0, partial_mode_sequence):

        # reset program
        self.prog.reset()

        # set up miqp
        self.set_type_auxliaries('C')
        self.update_mode_sequence(partial_mode_sequence)
        self.set_initial_condition(x0)

        # parameters
        self.prog.setParam('OutputFlag', 0)
        self.prog.setParam('Method', 0)

        # fix part of the mode sequence
        self.update_mode_sequence(partial_mode_sequence)

        # run the optimization
        self.prog.optimize()

        return self.organize_result()

    def feedforward(self, x0):

        # reset program
        self.prog.reset()

        # set up miqp
        self.set_type_auxliaries('B')
        self.update_mode_sequence([])
        self.set_initial_condition(x0)

        # parameters
        self.prog.setParam('OutputFlag', 1)
        self.prog.setParam('Threads', 0)

        # run the optimization
        self.prog.optimize()
        # print self.prog.Runtime

        return self.organize_result()

    def set_type_auxliaries(self, d_type):
        for t in range(self.N):
            for i in range(self.S.nm):
                self.prog.getVarByName('d%d[%d]'%(t,i)).VType = d_type

    def organize_result(self):
        if self.prog.status == 2:
            x = [np.vstack([self.prog.getVarByName('x%d[%d]'%(t,k)).x for k in range(self.S.nx)]) for t in range(self.N+1)]
            u = [np.vstack([self.prog.getVarByName('u%d[%d]'%(t,k)).x for k in range(self.S.nu)]) for t in range(self.N)]
            d = [[self.prog.getVarByName('d%d[%d]'%(t,k)).x for k in range(self.S.nm)] for t in range(self.N)]
            ms = [dt.index(max(dt)) for dt in d]
            cost = self.prog.objVal
            return u, x, ms, cost
        else:
            return [None]*4