# external imports
import numpy as np
import gurobipy as grb
from scipy.linalg import block_diag
from copy import copy

# internal inputs
from pympc.control.hybrid_benchmark.utils import graph_representation, big_m, add_vars, add_linear_inequality, add_linear_equality, add_rotated_socc
from pympc.optimization.programs import linear_program


class HybridModelPredictiveController(object):

    def __init__(self, S, N, Q, R, P, X_N, method='Big-M'):

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
        if method == 'Traditional formulation':
            return self.bild_miqp_bemporad_morari()
        if method == 'Big-M':
            return self.bild_miqp_bigm()
        if method == 'Stronger big-M':
            return self.bild_miqp_improved_bigm()
        if method == 'Convex hull':
            return self.bild_miqp_convex_hull()
        if method == 'Convex hull, lifted constraints':
            return self.build_miqp_improved_convex_hull()
        else:
            raise ValueError('unknown method ' + method + '.')

    def bild_miqp_bemporad_morari(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # big-M dynamics
        alpha = []
        beta = []
        for i, S_i in enumerate(self.S.affine_systems):
            alpha_i = []
            beta_i = []
            A_i = np.hstack((S_i.A, S_i.B))
            for j, S_j in enumerate(self.S.affine_systems):
                alpha_ij = []
                beta_ij = []
                D_j = self.S.domains[j]
                for k in range(S_i.nx):
                    f = A_i[k:k+1,:].T
                    sol = linear_program(f, D_j.A, D_j.b, D_j.C, D_j.d)
                    alpha_ij.append(sol['min'] + S_i.c[k,0])
                    sol = linear_program(-f, D_j.A, D_j.b, D_j.C, D_j.d)
                    beta_ij.append(- sol['min'] + S_i.c[k,0])
                alpha_i.append(np.vstack(alpha_ij))
                beta_i.append(np.vstack(beta_ij))
            alpha.append(alpha_i)
            beta.append(beta_i)
        alpha = np.minimum.reduce([np.minimum.reduce([alpha_ij for alpha_ij in alpha_i]) for alpha_i in alpha])
        beta = np.maximum.reduce([np.maximum.reduce([beta_ij for beta_ij in beta_i]) for beta_i in beta])

        # big-M domains
        gamma = []
        for i, D_i in enumerate(self.S.domains):
            gamma_i = []
            for j, D_j in enumerate(self.S.domains):
                gamma_ij = []
                for k in range(D_i.A.shape[0]):
                    f = -D_i.A[k:k+1,:].T
                    sol = linear_program(f, D_j.A, D_j.b, D_j.C, D_j.d)
                    gamma_ij.append(- sol['min'] - D_i.b[k,0])
                gamma_i.append(np.vstack(gamma_ij))
            gamma.append(gamma_i)
        gamma = [np.maximum.reduce([gamma_ij for gamma_ij in gamma_i]) for gamma_i in gamma]

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
            z = [add_vars(prog, nx) for i in range(nm)]
            u = add_vars(prog, nu, name='u%d'%t)
            d = add_vars(prog, nm, lb=[0.]*nm, name='d%d'%t)
            prog.update()

            # constrained dynamics
            add_linear_equality(prog, x_next, sum(z))
            xu = np.concatenate((x, u))
            for i in range(nm):
                Ai = self.S.affine_systems[i].A
                Bi = self.S.affine_systems[i].B
                ci = self.S.affine_systems[i].c
                dyn_i = Ai.dot(x) + Bi.dot(u) + ci.flatten()
                add_linear_inequality(prog, alpha*d[i], z[i])
                add_linear_inequality(prog, z[i], beta*d[i])
                add_linear_inequality(prog, alpha*(1.-d[i]), dyn_i - z[i])
                add_linear_inequality(prog, dyn_i - z[i], beta*(1.-d[i]))
                add_linear_inequality(prog, self.S.domains[i].A.dot(xu), self.S.domains[i].b + gamma[i]*(1.-d[i]))

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

    def bild_miqp_bigm(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # graph of the dynamics and big-Ms
        P = graph_representation(self.S)
        _, mi = big_m(P)

        # graph of the dynamics and big-Ms, final stage
        P_N = []
        P_N_index = []
        for i, Pi in enumerate(P):
            Pi = copy(Pi)
            Pi.add_inequality(self.X_N.A, self.X_N.b, range(nx+nu, 2*nx+nu))
            if not Pi.empty:
                P_N.append(Pi)
                P_N_index.append(i)
        _, mi_N = big_m(P_N)

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
            if t < self.N-1:
                d = add_vars(prog, nm, lb=[0.]*nm, name='d%d'%t)
            else:
                d = np.array([prog.addVar(lb=0., name='d%d[%d]'%(t,k)) for k in P_N_index])
            prog.update()

            # constrained dynamics
            xux = np.concatenate((x, u, x_next))
            if t < self.N-1:
                for i in range(nm):
                    add_linear_inequality(prog, P[i].A.dot(xux), P[i].b + mi[i]*(1.-d[i]))
            else:
                for i in range(len(P_N)):
                    add_linear_inequality(prog, P_N[i].A.dot(xux), P_N[i].b + mi_N[i]*(1.-d[i]))

            # constraints on the binaries
            cons = prog.addConstr(sum(d) == 1.)

            # stage cost to the objective
            obj += .5 * (x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))

        # # terminal constraint
        # add_linear_inequality(prog, self.X_N.A.dot(x_next), self.X_N.b)

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

        # graph of the dynamics and big-Ms, final stage
        P_N = []
        P_N_index = []
        for i, Pi in enumerate(P):
            Pi = copy(Pi)
            Pi.add_inequality(self.X_N.A, self.X_N.b, range(nx+nu, 2*nx+nu))
            if not Pi.empty:
                P_N.append(Pi)
                P_N_index.append(i)
        m_N, _ = big_m(P_N)

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
            if t < self.N-1:
                d = add_vars(prog, nm, lb=[0.]*nm, name='d%d'%t)
            else:
                d = np.array([prog.addVar(lb=0., name='d%d[%d]'%(t,k)) for k in P_N_index])
            prog.update()

            # constrained dynamics
            xux = np.concatenate((x, u, x_next))
            if t < self.N-1:
                for i in range(nm):
                    sum_mi = sum(m[i][j] * d[j] for j in range(nm) if j != i)
                    add_linear_inequality(prog, P[i].A.dot(xux), P[i].b + sum_mi)
            else:
                for i in range(len(P_N)):
                    sum_mi = sum(m_N[i][j] * d[j] for j in range(len(P_N)) if j != i)
                    add_linear_inequality(prog, P_N[i].A.dot(xux), P_N[i].b + sum_mi)

            # constraints on the binaries
            cons = prog.addConstr(sum(d) == 1.)

            # stage cost to the objective
            obj += .5 * (x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))

        # # terminal constraint
        # add_linear_inequality(prog, self.X_N.A.dot(x_next), self.X_N.b)

        # terminal cost
        obj += .5 * x_next.dot(self.P).dot(x_next)
        prog.setObjective(obj)

        return prog

    def bild_miqp_convex_hull(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # graph of the dynamics
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

            # # constrained dynamics (with explicit equalities)
            # for i in range(nm):
            #     Di = self.S.domains[i]
            #     Si = self.S.affine_systems[i]
            #     add_linear_inequality(
            #         prog,
            #         Di.A.dot(np.concatenate((y[i], v[i]))),
            #         Di.b * d[i]
            #         )
            #     add_linear_equality(
            #         prog,
            #         Si.A.dot(y[i]) + Si.B.dot(v[i]) - z[i],
            #         - Si.c * d[i]
            #         )

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

    def build_miqp_improved_convex_hull(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # graph of the dynamics
        P = graph_representation(self.S)

        # initialize program
        prog = grb.Model()
        obj = 0.
        
        # loop over time
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

            # auxiliary continuous variables for the improved convex-hull method
            y = [add_vars(prog, nx) for i in range(nm)]
            z = [add_vars(prog, nx) for i in range(nm)]
            v = [add_vars(prog, nu) for i in range(nm)]
            s = add_vars(prog, nm, lb=[0.]*nm)
            prog.update()

            # constrained dynamics
            for i in range(nm):
                yvzi = np.concatenate((y[i], v[i], z[i]))
                add_linear_inequality(prog, P[i].A.dot(yvzi), P[i].b * d[i])

            # # constrained dynamics (with explicit equalities)
            # for i in range(nm):
            #     Di = self.S.domains[i]
            #     Si = self.S.affine_systems[i]
            #     add_linear_inequality(
            #         prog,
            #         Di.A.dot(np.concatenate((y[i], v[i]))),
            #         Di.b * d[i]
            #         )
            #     add_linear_equality(
            #         prog,
            #         Si.A.dot(y[i]) + Si.B.dot(v[i]) - z[i],
            #         - Si.c * d[i]
            #         )

                # stage cost
                if t < self.N - 1:
                    add_rotated_socc(
                        prog,
                        block_diag(self.Q, self.R),
                        np.concatenate((y[i], v[i])),
                        s[i],
                        d[i]
                        )

                # final time step
                else:

                    # terminal cost
                    add_rotated_socc(
                        prog,
                        block_diag(self.Q, self.R, self.P),
                        np.concatenate((y[i], v[i], z[i])),
                        s[i],
                        d[i]
                        )

                    # terminal constraint
                    add_linear_inequality(prog, self.X_N.A.dot(z[i]), self.X_N.b * d[i])

            # update cost
            obj += sum(s)

            # recompose the state and input (convex hull method)
            add_linear_equality(prog, x, sum(y))
            add_linear_equality(prog, x_next, sum(z))
            add_linear_equality(prog, u, sum(v))

            # constraints on the binaries
            cons = prog.addConstr(sum(d) == 1.)

        # set cost
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
        self.prog.setParam('BarConvTol', 1.e-8)

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
        # self.prog.setParam('OutputFlag', 1)
        self.prog.setParam('Threads', 0)

        # run the optimization
        self.prog.optimize()
        # print self.prog.Runtime

        return self.organize_result()

    def set_type_auxliaries(self, d_type):
        for t in range(self.N):
            for i in range(self.S.nm):
                v = self.prog.getVarByName('d%d[%d]'%(t,i))
                if v is not None:
                    v.VType = d_type

    def organize_result(self):
        if self.prog.status in [2, 9, 11] and self.prog.SolCount > 0: # optimal or interrupted or time limit
            x = [np.vstack([self.prog.getVarByName('x%d[%d]'%(t,k)).x for k in range(self.S.nx)]) for t in range(self.N+1)]
            u = [np.vstack([self.prog.getVarByName('u%d[%d]'%(t,k)).x for k in range(self.S.nu)]) for t in range(self.N)]
            d = []
            for t in range(self.N):
                dt = []
                for k in range(self.S.nm):
                    dtk = self.prog.getVarByName('d%d[%d]'%(t,k))
                    if dtk is not None:
                        dt.append(dtk.x)
                    else:
                        dt.append(0.)
                d.append(dt)
            # d = [[self.prog.getVarByName('d%d[%d]'%(t,k)).x for k in range(self.S.nm)] for t in range(self.N)]
            ms = [dt.index(max(dt)) for dt in d]
            cost = self.prog.objVal
            return u, x, ms, cost
        else:
            return [None]*4