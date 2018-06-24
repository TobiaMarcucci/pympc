# external imports
import numpy as np
import gurobipy as grb

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.programs import linear_program

def graph_representation(S):
    '''
    For the PWA system S
    x+ = Ai x + Bi u + ci if Fi x + Gi u <= hi,
    returns the graphs of the dynamics (list of Polyhedron)
    [ Fi  Gi  0] [ x]    [ hi]
    [ Ai  Bi -I] [ u] <= [-ci]
    [-Ai -Bi  I] [x+]    [ ci]
    '''
    P = []
    for i in range(S.nm):
        Di = S.domains[i]
        Si = S.affine_systems[i]
        Ai = np.vstack((
            np.hstack((Di.A, np.zeros((Di.A.shape[0], S.nx)))),
            np.hstack((Si.A, Si.B, -np.eye(S.nx))),
            np.hstack((-Si.A, -Si.B, np.eye(S.nx))),
            ))
        bi = np.vstack((Di.b, -Si.c, Si.c))
        P.append(Polyhedron(Ai, bi))
    return P

def big_m(P_list, tol=1.e-6):
    '''
    For the list of Polyhedron P_list in the from Pi = {x | Ai x <= bi}
    - m, list of lists of numpy arrays with m[i][j] := max_{x in Pj} Ai x - bi
    - mi, list of numpy arrays with mi[i] := max_{x in P} Ai x - bi
    where P is the union of Pi for Pi in P_list.

    '''
    m = []
    for i, Pi in enumerate(P_list):
        mi = []
        for j, Pj in enumerate(P_list):
            mij = []
            for k in range(Pi.A.shape[0]):
                f = -Pi.A[k:k+1,:].T
                sol = linear_program(f, Pj.A, Pj.b)
                mijk = - sol['min'] - Pi.b[k,0]
                if np.abs(mijk) < tol:
                    mijk = 0.
                mij.append(mijk)
            mi.append(np.vstack(mij))
        m.append(mi)
    mi = [np.maximum.reduce([mij for mij in mi]) for mi in m]
    return m, mi

def add_vars(prog, n, lb=None, **kwargs):
    if lb is None:
        lb=[-grb.GRB.INFINITY] * n
    x = prog.addVars(n, lb=lb, **kwargs)
    return np.array([xi for xi in x.values()])

def add_linear_inequality(prog, x, y):
    return [prog.addConstr(x.flatten()[k] <= y.flatten()[k]) for k in range(x.size)]

def add_linear_equality(prog, x, y):
    return [prog.addConstr(x.flatten()[k] == y.flatten()[k]) for k in range(x.size)]

def get_constraint_set(prog):
    cs = prog.getConstrs()
    vs = prog.getVars()
    v_id = {v: i for i, v in enumerate(vs)}
    nv = len(vs)
    A = np.zeros((0, nv))
    b = np.zeros((0, 1))
    for i, v in enumerate(vs):
        if v.getAttr(grb.GRB.Attr.LB) != -grb.GRB.INFINITY:
            Ai = np.zeros((1, nv))
            Ai[0,i] = -1.
            bi = -v.getAttr(grb.GRB.Attr.LB)
            A = np.vstack((A, Ai))
            b = np.vstack((b, bi))
        if v.getAttr(grb.GRB.Attr.UB) != grb.GRB.INFINITY:
            Ai = np.zeros((1, nv))
            Ai[0,i] = 1.
            bi = v.getAttr(grb.GRB.Attr.UB)
            A = np.vstack((A, Ai))
            b = np.vstack((b, bi))
    for i, c in enumerate(cs):
        expr = prog.getRow(c)
        Ai = np.zeros((1, nv))
        for j in range(expr.size()):
            Ai[0, v_id[expr.getVar(j)]] = expr.getCoeff(j)
        bi = c.getAttr(grb.GRB.Attr.RHS)
        if c.getAttr(grb.GRB.Attr.Sense) == '>':
            Ai = -Ai
            bi = -bi
        if c.getAttr(grb.GRB.Attr.Sense) == '=':
            Ai = np.vstack((Ai, -Ai))
            bi = np.vstack((bi, -bi))
        A = np.vstack((A, Ai))
        b = np.vstack((b, bi))
    return Polyhedron(A, b)


# def get_constraint_set(prog):
#     cs = prog.getConstrs()
#     vs = prog.getVars()
#     v_id = {v: i for i, v in enumerate(vs)}
#     nv = len(vs)
#     A = np.zeros((0, nv))
#     b = np.zeros((0, 1))
#     C = np.zeros((0, nv))
#     d = np.zeros((0, 1))
#     for i, v in enumerate(vs):
#         if v.getAttr(grb.GRB.Attr.LB) != -grb.GRB.INFINITY:
#             Ai = np.zeros((1, nv))
#             Ai[0,i] = -1.
#             bi = -v.getAttr(grb.GRB.Attr.LB)
#             A = np.vstack((A, Ai))
#             b = np.vstack((b, bi))
#         if v.getAttr(grb.GRB.Attr.UB) != grb.GRB.INFINITY:
#             Ai = np.zeros((1, nv))
#             Ai[0,i] = 1.
#             bi = v.getAttr(grb.GRB.Attr.UB)
#             A = np.vstack((A, Ai))
#             b = np.vstack((b, bi))
#     for i, c in enumerate(cs):
#         expr = prog.getRow(c)
#         ACi = np.zeros((1, nv))
#         for j in range(expr.size()):
#             ACi[0, v_id[expr.getVar(j)]] = expr.getCoeff(j)
#         bdi = c.getAttr(grb.GRB.Attr.RHS)
#         if c.getAttr(grb.GRB.Attr.Sense) == '<':
#             A = np.vstack((A, ACi))
#             b = np.vstack((b, bdi))
#         if c.getAttr(grb.GRB.Attr.Sense) == '>':
#             A = np.vstack((A, -ACi))
#             b = np.vstack((b, -bdi))
#         if c.getAttr(grb.GRB.Attr.Sense) == '=':
#             C = np.vstack((C, ACi))
#             d = np.vstack((d, bdi))
#     return Polyhedron(A, b, C, d)


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
