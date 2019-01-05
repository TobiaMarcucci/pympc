# external imports
import numpy as np
import gurobipy as grb
from itertools import product
from scipy.linalg import block_diag

# internal inputs
from pympc.geometry.polyhedron import Polyhedron

def add_vars(prog, n, lb=None, **kwargs):
    if lb is None:
        lb = [-grb.GRB.INFINITY] * n
    x = prog.addVars(n, lb=lb, **kwargs)
    prog.update()
    return np.array([xi for xi in x.values()])

def add_linear_inequality(prog, x, y):
    assert x.size == y.size
    return [prog.addConstr(x[k] - y[k] <= 0.) for k in range(x.size)] # sometimes x is a vector of floats: gurobi raises errors if variables are not in the lhs

def add_linear_equality(prog, x, y):
    assert x.size == y.size
    return [prog.addConstr(x[k] - y[k] == 0.) for k in range(x.size)] # sometimes x is a vector of floats: gurobi raises errors if variables are not in the lhs

def add_stage_cost(prog, Q, R, x, u, norm):

    # stage cost infinity norm
    if norm == 'inf':

        # add slacks
        sx = add_vars(prog, 1, lb=[0.])[0]
        su = add_vars(prog, 1, lb=[0.])[0]
        obj = sx + su

        # enforce infinity norm
        add_linear_inequality(prog,  Q.dot(x), np.ones(x.size)*sx)
        add_linear_inequality(prog, -Q.dot(x), np.ones(x.size)*sx)
        add_linear_inequality(prog,  R.dot(u), np.ones(u.size)*su)
        add_linear_inequality(prog, -R.dot(u), np.ones(u.size)*su)

    # stage cost one norm
    elif norm == 'one':

        # add slacks
        sx = add_vars(prog, x.size, lb=[0.]*x.size)
        su = add_vars(prog, u.size, lb=[0.]*u.size)
        obj = sum(sx) + sum(su)

        # enforce one norm
        add_linear_inequality(prog,  Q.dot(x), sx)
        add_linear_inequality(prog, -Q.dot(x), sx)
        add_linear_inequality(prog,  R.dot(u), su)
        add_linear_inequality(prog, -R.dot(u), su)

    # stage cost one norm
    elif norm == 'two':
        obj = .5 * (x.dot(Q).dot(x) + u.dot(R).dot(u))

    return obj

def add_terminal_cost(prog, P, x, norm):

    # stage cost infinity norm
    if norm == 'inf':

        # add slacks
        s = add_vars(prog, 1, lb=[0.])[0]
        obj = s

        # enforce infinity norm
        add_linear_inequality(prog,  P.dot(x), np.ones(x.size)*s)
        add_linear_inequality(prog, -P.dot(x), np.ones(x.size)*s)

    # stage cost one norm
    elif norm == 'one':

        # add slacks
        s = add_vars(prog, x.size, lb=[0.]*x.size)
        obj = sum(s)

        # enforce one norm
        add_linear_inequality(prog,  P.dot(x), s)
        add_linear_inequality(prog, -P.dot(x), s)

    # stage cost one norm
    elif norm == 'two':
        obj = .5 * x.dot(P).dot(x)

    return obj

def add_rotated_socc(prog, H, x, y, z, tol=1.e-8):
    '''
    Adds the constraint 1/2 x' H x <= y z, with H symmetric positive semidefinite, y and z nonegative.
    '''
    assert np.allclose(H - H.T, 0.) # ensures H sym
    eigvals, eigvecs = np.linalg.eig(H)
    assert np.all(eigvals >= 0.) # ensures H psd
    x_aux = add_vars(prog, x.size)
    R = np.diag(np.sqrt(eigvals)).dot(eigvecs.T)
    assert np.allclose(R.T.dot(R) - H, 0.) # ensures R' R is a decomposition of H
    add_linear_equality(prog, x_aux, R.dot(x))
    cons = prog.addConstr(.5 * x_aux.dot(x_aux) <= y * z),
    return cons, x_aux

def feasible_mode_sequences(S, T, tol=1.e-7):
    '''
    For the piecewise affine system S returns the feasible and
    infeasible modesequences for a time window of T time steps.
    '''
    fmss = []
    imss = []
    for t in range(T):
        for ms in product(*[range(S.nm)]*(t+1)):
            if not any(is_included(ims, ms) for ims in imss):
                F_c = block_diag(*[S.domains[m].A[:,:S.nx] for m in ms])
                G_c = block_diag(*[S.domains[m].A[:,S.nx:] for m in ms])
                h_c = np.concatenate([S.domains[m].b for m in ms])
                A_c, B_c, c_c = [M[:-S.nx] for M in S.condense(ms)]
                lhs = np.hstack((G_c + F_c.dot(B_c), F_c.dot(A_c)))
                rhs = h_c - F_c.dot(c_c)
                fs = Polyhedron(lhs, rhs)
                if fs.empty:
                    imss.append(ms)
                elif fs.radius < tol:
                    imss.append((ms, fs.radius))
                elif t == T-1:
                    fmss.append((ms, fs.radius))
    return fmss, imss

def is_included(l1, l2):
    for i in range(len(l2)-len(l1)+1):
        if l2[i:i+len(l1)] == l1:
            return True
    return False