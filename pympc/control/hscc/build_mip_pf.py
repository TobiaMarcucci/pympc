# external imports
import numpy as np
import gurobipy as grb
from scipy.linalg import block_diag

# internal inputs
from pympc.control.hscc.utils import (add_vars,
                                      add_linear_inequality,
                                      add_linear_equality,
                                      add_rotated_socc
                                      )

def bild_mip_pf(S, N, Q, R, P, X_N, norm):

    # shortcuts
    [nx, nu, nm] = [S.nx, S.nu, S.nm]

    # initialize program
    prog = grb.Model()
    obj = 0.
    
    # loop over time
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

        # auxiliary continuous variables for the improved convex-hull method
        y = [add_vars(prog, nx) for i in range(nm)]
        v = [add_vars(prog, nu) for i in range(nm)]
        z = [add_vars(prog, nx) for i in range(nm)]

        # slacks for the stage cost infinity norm
        if norm == 'inf':
            sx = [add_vars(prog, 1, lb=[0.])[0] for i in range(nm)]
            su = [add_vars(prog, 1, lb=[0.])[0] for i in range(nm)]
            obj += sum(sx) + sum(su)

        # slacks for the stage cost one norm
        elif norm == 'one':
            sx = [add_vars(prog, Q.shape[0], lb=[0.]*Q.shape[0]) for i in range(nm)]
            su = [add_vars(prog, R.shape[0], lb=[0.]*R.shape[0]) for i in range(nm)]
            obj += sum(sum(sx)) + sum(sum(su))

        # slacks for the stage cost two norm
        elif norm == 'two':
            s = [add_vars(prog, 1, lb=[0.])[0] for i in range(nm)]
            obj += sum(s)

        # constrained dynamics
        for i in range(nm):

            # enforce dynamics
            Si = S.affine_systems[i]
            add_linear_equality(prog, z[i], Si.A.dot(y[i]) + Si.B.dot(v[i]) + Si.c * d[i])

            # enforce state and input constraints
            Di = S.domains[i]
            yvi = np.concatenate((y[i], v[i]))
            add_linear_inequality(prog, Di.A.dot(yvi), Di.b * d[i])

        # stage cost infinity norm
        if norm == 'inf':
            for i in range(nm):
                add_linear_inequality(prog,  Q.dot(y[i]), np.ones(Q.shape[0]) * sx[i])
                add_linear_inequality(prog, -Q.dot(y[i]), np.ones(Q.shape[0]) * sx[i])
                add_linear_inequality(prog,  R.dot(v[i]), np.ones(R.shape[0]) * su[i])
                add_linear_inequality(prog, -R.dot(v[i]), np.ones(R.shape[0]) * su[i])

            # terminal cost infinity norm
            if t == N - 1:
                sxN = [add_vars(prog, 1, lb=[0.])[0] for i in range(nm)]
                obj += sum(sxN)
                for i in range(nm):
                    add_linear_inequality(prog,  P.dot(z[i]), np.ones(P.shape[0]) * sxN[i])
                    add_linear_inequality(prog, -P.dot(z[i]), np.ones(P.shape[0]) * sxN[i])
                    
        # stage cost one norm
        elif norm == 'one':
            for i in range(nm):
                add_linear_inequality(prog,  Q.dot(y[i]), sx[i])
                add_linear_inequality(prog, -Q.dot(y[i]), sx[i])
                add_linear_inequality(prog,  R.dot(v[i]), su[i])
                add_linear_inequality(prog, -R.dot(v[i]), su[i])

            # terminal cost one norm
            if t == N - 1:
                sxN = [add_vars(prog, P.shape[0], lb=[0.]*P.shape[0]) for i in range(nm)]
                obj += sum(sum(sxN))
                for i in range(nm):
                    add_linear_inequality(prog,  P.dot(z[i]), sxN[i])
                    add_linear_inequality(prog, -P.dot(z[i]), sxN[i])

        # stage cost two norm
        elif norm == 'two':
            for i in range(nm):
                if t < N - 1:
                    QR = block_diag(Q, R)
                    yvi = np.concatenate((y[i], v[i]))
                    add_rotated_socc(prog, QR, yvi, s[i], d[i])

                # terminal cost two norm
                else:
                    QRP = block_diag(Q, R, P)
                    yvzi = np.concatenate((y[i], v[i], z[i]))
                    add_rotated_socc(prog, QRP, yvzi, s[i], d[i])

        # recompose variables
        add_linear_equality(prog, x, sum(y))
        add_linear_equality(prog, u, sum(v))
        add_linear_equality(prog, x_next, sum(z))

        # constraints on the binaries
        prog.addConstr(sum(d) == 1.)

    # terminal constraint
    for i in range(nm):
        add_linear_inequality(prog, X_N.A.dot(z[i]), X_N.b * d[i])

    # set cost
    prog.setObjective(obj)

    return prog