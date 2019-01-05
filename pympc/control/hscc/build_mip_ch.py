# external imports
import numpy as np
import gurobipy as grb

# internal inputs
from pympc.control.hscc.utils import (add_vars,
                                      add_linear_inequality,
                                      add_linear_equality,
                                      add_stage_cost,
                                      add_terminal_cost
                                      )

def bild_mip_ch(S, N, Q, R, P, X_N, norm):

    # shortcuts
    [nx, nu, nm] = [S.nx, S.nu, S.nm]

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

        # auxiliary continuous variables for the convex-hull method
        y = [add_vars(prog, nx) for i in range(nm)]
        v = [add_vars(prog, nu) for i in range(nm)]
        z = [add_vars(prog, nx) for i in range(nm)]

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
        add_linear_equality(prog, x, sum(y))
        add_linear_equality(prog, u, sum(v))
        add_linear_equality(prog, x_next, sum(z))

        # constraints on the binaries
        prog.addConstr(sum(d) == 1.)

        # stage cost
        obj += add_stage_cost(prog, Q, R, x, u, norm)

    # terminal constraint
    for i in range(nm):
        add_linear_inequality(prog, X_N.A.dot(z[i]), X_N.b * d[i])

    # terminal cost
    obj += add_terminal_cost(prog, P, x_next, norm)
    prog.setObjective(obj)

    return prog