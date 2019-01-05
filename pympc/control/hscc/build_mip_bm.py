# external imports
import numpy as np
import gurobipy as grb

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.programs import linear_program
from pympc.control.hscc.utils import (add_vars,
                                      add_linear_inequality,
                                      add_stage_cost,
                                      add_terminal_cost
                                      )

def bild_mip_bm(S, N, Q, R, P, X_N, norm):

    # shortcuts
    [nx, nu, nm] = [S.nx, S.nu, S.nm]

    # big-Ms
    M1, M2, M3 = get_bigm_stages(S)
    modes = range(nm)

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
        u = add_vars(prog, nu, name='u%d'%t)
        d = add_vars(prog, nm, lb=[0.]*nm, name='d%d'%t)
        xu = np.concatenate((x, u))

        # use different big-Ms for the terminal stage (some of the domains can be empty!)
        if t == N-1:
            M1, M2, M3, M4 = get_bigm_terminal(S, X_N)
            modes = [i for i in range(nm) if M1[0][i][0] is not None]

        # loop over system modes
        for i in modes:

            # shortcuts
            Si = S.affine_systems[i]
            Di = S.domains[i]

            # enforce dynamics
            sum_M1_i = sum(M1[i][j] * d[j] for j in modes if j != i)
            sum_M2_i = sum(M2[i][j] * d[j] for j in modes if j != i)
            dyn_i = Si.A.dot(x) + Si.B.dot(u) + Si.c
            add_linear_inequality(prog, dyn_i, x_next + sum_M1_i)
            add_linear_inequality(prog, x_next - sum_M2_i, dyn_i)

            # enforce state and input constraints
            sum_M3_i = sum(M3[i][j] * d[j] for j in modes if j != i)          
            add_linear_inequality(prog, Di.A.dot(xu), Di.b + sum_M3_i)

            # terminal constraint
            if t == N-1:
                sum_M4_i = sum(M4[i][j] * d[j] for j in modes if j != i)
                add_linear_inequality(prog, X_N.A.dot(x_next), X_N.b + sum_M4_i)

        # constraints on the binaries
        prog.addConstr(sum(d) == 1.)

        # stage cost
        obj += add_stage_cost(prog, Q, R, x, u, norm)

    # terminal cost
    obj += add_terminal_cost(prog, P, x_next, norm)
    prog.setObjective(obj)

    return prog

def get_bigm_stages(S):

    # express the graph of the PWA dyanmics as a list of polyhedra
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

    # get and split big-Ms
    M = bigm(P)
    M1 = [[M[i][j][:S.nx] for j in range(S.nm)] for i in range(S.nm)]
    M2 = [[M[i][j][S.nx:2*S.nx] for j in range(S.nm)] for i in range(S.nm)]
    M3 = [[M[i][j][2*S.nx:] for j in range(S.nm)] for i in range(S.nm)]

    return M1, M2, M3

def get_bigm_terminal(S, X_N):

    # express the graph of the PWA dyanmics as a list of polyhedra
    P = []
    for i in range(S.nm):
        Di = S.domains[i]
        Si = S.affine_systems[i]
        Ai = np.vstack((
            np.hstack((Si.A, Si.B, -np.eye(S.nx))),
            np.hstack((-Si.A, -Si.B, np.eye(S.nx))),
            np.hstack((Di.A, np.zeros((Di.A.shape[0], S.nx)))),
            np.hstack((np.zeros((X_N.A.shape[0], Di.A.shape[1])), X_N.A))
            ))
        bi = np.concatenate((-Si.c, Si.c, Di.b, X_N.b))
        P.append(Polyhedron(Ai, bi))

    # get and split big-Ms
    M = bigm(P)
    M1 = [[M[i][j][:S.nx] for j in range(S.nm)] for i in range(S.nm)]
    M2 = [[M[i][j][S.nx:2*S.nx] for j in range(S.nm)] for i in range(S.nm)]
    M3 = [[M[i][j][2*S.nx:-X_N.A.shape[0]] for j in range(S.nm)] for i in range(S.nm)]
    M4 = [[M[i][j][-X_N.A.shape[0]:] for j in range(S.nm)] for i in range(S.nm)]

    return M1, M2, M3, M4

def bigm(P_list):
    '''
    For the list of Polyhedron P_list in the from Pi = {x | Ai x <= bi} returns a
    list of lists of numpy arrays with m[i][j] := max_{x in Pj} Ai x - bi.
    '''

    # m[i][j] := max_{x in Pj} Ai x - bi
    M = []

    # loop over the objectives
    for Pi in P_list:
        Mi = []

        # loop over the domains
        for Pj in P_list:
            Mij = []

            # loop over the facets of the ith polyhedron
            for k in range(Pi.A.shape[0]):
                sol = linear_program(-Pi.A[k], Pj.A, Pj.b)

                # the polyhedron can be empty
                if sol['min'] is not None:
                    Mijk = - sol['min'] - Pi.b[k]
                else:
                    Mijk = None
                Mij.append(Mijk)

            # append i
            Mi.append(np.array(Mij))

        # append j
        M.append(Mi)

    return M