# external imports
import numpy as np
import gurobipy as grb

# internal inputs
from pympc.optimization.programs import linear_program
from pympc.control.hscc.utils import (add_vars,
                                      add_linear_inequality,
                                      add_linear_equality,
                                      add_stage_cost,
                                      add_terminal_cost
                                      )

def bild_mip_mld(S, N, Q, R, P, X_N, norm):

    # shortcuts
    [nx, nu, nm] = [S.nx, S.nu, S.nm]

    # big-Ms
    M1, M2 = bigm_dynamics(S)
    M3 = bigm_domains(S)

    # initialize program
    prog = grb.Model()
    obj = 0.

    # loop over the time horizon
    for t in range(N):

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
        xu = np.concatenate((x, u))

        # loop over system modes
        for i in range(nm):

            # shortcuts
            Ai = S.affine_systems[i].A
            Bi = S.affine_systems[i].B
            ci = S.affine_systems[i].c
            FGi = S.domains[i].A
            hi = S.domains[i].b

            # enforce dynamics
            dyn_i = Ai.dot(x) + Bi.dot(u) + ci
            add_linear_inequality(prog, z[i], M1*d[i])
            add_linear_inequality(prog, M2*d[i], z[i])
            add_linear_inequality(prog, dyn_i - z[i], M1*(1. - d[i]))
            add_linear_inequality(prog, M2*(1. - d[i]), dyn_i - z[i])

            # enforce state and input constraints            
            add_linear_inequality(prog, FGi.dot(xu), hi + M3[i]*(1. - d[i]))

        # reconstruct state and binaries
        add_linear_equality(prog, x_next, sum(z))
        prog.addConstr(sum(d) == 1.)
        # prog.addSOS(grb.GRB.SOS_TYPE1, d, [1.]*d.size)

        # stage cost
        obj += add_stage_cost(prog, Q, R, x, u, norm)

    # terminal constraint
    add_linear_inequality(prog, X_N.A.dot(x_next), X_N.b)

    # terminal cost
    obj += add_terminal_cost(prog, P, x_next, norm)
    prog.setObjective(obj)

    return prog

def bigm_dynamics(S):

    # max/min over i (max/min over j (max/min over domain j of dynamics i))
    M1 = np.array([-np.inf]*S.nx)
    M2 = np.array([np.inf]*S.nx)

    # loop over the ith dynamics
    for S_i in S.affine_systems:
        M1_i = np.array([-np.inf]*S.nx)
        M2_i = np.array([np.inf]*S.nx)
        f_i = np.hstack((S_i.A, S_i.B))

        # loop over the jth domain
        for D_j in S.domains:
            M1_ij = []
            M2_ij = []

            # loop over the states
            for k in range(S_i.nx):

                # maximize
                sol = linear_program(-f_i[k], D_j.A, D_j.b)
                M1_ij.append(- sol['min'] + S_i.c[k])

                # minimize
                sol = linear_program(f_i[k], D_j.A, D_j.b)
                M2_ij.append(sol['min'] + S_i.c[k])

            # max/min over j
            M1_i = np.maximum.reduce([M1_i, np.array(M1_ij)])
            M2_i = np.minimum.reduce([M2_i, np.array(M2_ij)])

        # max/min over i
        M1 = np.maximum.reduce([M1, np.array(M1_i)])
        M2 = np.minimum.reduce([M2, np.array(M2_i)])

    return M1, M2

def bigm_domains(S):

    # max over j (max over domain j of domain i)
    M = []

    # loop over the ith domain
    for D_i in S.domains:
        M_i = np.array([-np.inf]*D_i.A.shape[0])

        # loop over the jth domain
        for D_j in S.domains:
            M_ij = []

            # loop over the rows of the ith constraint
            for k in range(D_i.A.shape[0]):
                sol = linear_program(-D_i.A[k], D_j.A, D_j.b)
                M_ij.append(- sol['min'] - D_i.b[k])

            # max over j
            M_i = np.maximum.reduce([M_i, np.array(M_ij)])

        # append the ith big-M
        M.append(M_i)

    return M