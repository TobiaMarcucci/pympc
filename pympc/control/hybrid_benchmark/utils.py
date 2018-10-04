# external imports
import numpy as np
import gurobipy as grb
from scipy.linalg import ldl
from itertools import product
from scipy.linalg import block_diag

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.programs import linear_program
from pympc.geometry.utils import plane_through_points

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






##########################

def get_constraint_set(prog):
    '''
    Returns the linear constraints of prog in the form A x < = b.
    '''
    prog.update()
    vs = prog.getVars()
    v_id = {v: i for i, v in enumerate(vs)}
    nv = len(vs)
    P = Polyhedron(np.zeros((0, nv)), np.zeros(0))
    for i, v in enumerate(vs):
        if v.getAttr(grb.GRB.Attr.LB) != -grb.GRB.INFINITY:
            P.add_lower_bound(v.getAttr(grb.GRB.Attr.LB), [i])
        if v.getAttr(grb.GRB.Attr.UB) != grb.GRB.INFINITY:
            P.add_upper_bound(v.getAttr(grb.GRB.Attr.UB), [i])
    cs = prog.getConstrs()
    for i, c in enumerate(cs):
        expr = prog.getRow(c)
        Ai = np.zeros((1, nv))
        for j in range(expr.size()):
            Ai[0, v_id[expr.getVar(j)]] = expr.getCoeff(j)
        bi = np.array([c.getAttr(grb.GRB.Attr.RHS)])
        if c.getAttr(grb.GRB.Attr.Sense) in ['<', '=']:
            P.add_inequality(Ai, bi)
        if c.getAttr(grb.GRB.Attr.Sense) in ['>', '=']:
            P.add_inequality(-Ai, -bi)
    return P

def remove_redundant_inequalities_fast(P, tol=1.e-7):

    assert P.C.shape[0] == 0
    assert P.d.shape[0] == 0
    
    # intialize program
    prog = grb.Model()
    x = add_vars(prog, P.A.shape[1])
    prog.update()
    cons = add_linear_inequality(prog, P.A.dot(x), P.b)
    prog.update()
    prog.setParam('OutputFlag', 0)
    
    # initialize list of non-redundant facets
    minimal_facets = list(range(len(P.A)))
    
    # check each facet
    for i in range(P.A.shape[0]):

        # solve linear program
        cons[i].RHS += 1.
        prog.setObjective(P.A[i].dot(x), sense=grb.GRB.MAXIMIZE) # DO NOT PUT AN OFFSET TERM IN THE OBJECTIVE !!!
        prog.optimize()

        # remove redundant facets from the list
        if  prog.objVal - P.b[i] < tol:
            prog.remove(cons[i])
            minimal_facets.remove(i)
        else:
            cons[i].RHS -= 1.

    return Polyhedron(P.A[minimal_facets,:], P.b[minimal_facets,:])

def is_included(l1, l2):
    for i in range(len(l2)-len(l1)+1):
        if l2[i:i+len(l1)] == l1:
            return True
    return False

def infeasible_mode_sequences(PWA, t_max):
    imss = []
    for t in range(1, t_max):
        for ms in product(*[range(PWA.nm)]*(t+1)):
            if not any(is_included(ims, ms) for ims in imss):
                F_c = block_diag(*[PWA.domains[m].A[:,:PWA.nx] for m in ms])
                G_c = block_diag(*[PWA.domains[m].A[:,PWA.nx:] for m in ms])
                h_c = np.concatenate([PWA.domains[m].b for m in ms])
                A_c, B_c, c_c = [M[:-PWA.nx] for M in PWA.condense(ms)]
                lhs = np.hstack((G_c + F_c.dot(B_c), F_c.dot(A_c)))
                rhs = h_c - F_c.dot(c_c)
                fs = Polyhedron(lhs, rhs)
                if fs.empty:
                    imss.append(ms)
    return imss


def read_gurobi_status(status):
    return {
        1: 'loaded', # Model is loaded, but no solution information is available.'
        2: 'optimal',   # Model was solved to optimality (subject to tolerances), and an optimal solution is available.
        3: 'infeasible', # Model was proven to be infeasible.
        4: 'inf_or_unbd', # Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.
        5: 'unbounded', # Model was proven to be unbounded. Important note: an unbounded status indicates the presence of an unbounded ray that allows the objective to improve without limit. It says nothing about whether the model has a feasible solution. If you require information on feasibility, you should set the objective to zero and reoptimize.
        6: 'cutoff', # Optimal objective for model was proven to be worse than the value specified in the Cutoff parameter. No solution information is available. (Note: problem might also be infeasible.)
        7: 'iteration_limit', # Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterationLimit parameter, or because the total number of barrier iterations exceeded the value specified in the BarIterLimit parameter.
        8: 'node_limit', # Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter.
        9: 'time_limit', # Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter.
        10: 'solution_limit', # Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter.
        11: 'interrupted', # Optimization was terminated by the user.
        12: 'numeric', # Optimization was terminated due to unrecoverable numerical difficulties.
        13: 'suboptimal', # Unable to satisfy optimality tolerances; a sub-optimal solution is available.
        14: 'in_progress', # An asynchronous optimization call was made, but the associated optimization run is not yet complete.
        15: 'user_obj_limit' # User specified an objective limit (a bound on either the best objective or the best bound), and that limit has been reached.
        }[status]