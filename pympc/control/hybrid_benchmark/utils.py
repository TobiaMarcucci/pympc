# external imports
import numpy as np
import gurobipy as grb
from scipy.linalg import ldl
from scipy.spatial import ConvexHull

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.programs import linear_program
from pympc.geometry.utils import plane_through_points


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

def add_rotated_socc(prog, H, x, y, z, tol=1.e-8):
    '''
    Adds the constraint 1/2 x' H x <= y z, with H symmetric positive semidefinite, y and z nonegative.
    '''
    assert np.allclose(H - H.T, 0.) # ensures H sym
    eigvals, eigvecs = np.linalg.eig(H)
    assert np.all(eigvals >= 0.) # ensures H psd
    x_aux = add_vars(prog, x.size)
    prog.update()
    R = np.diag(np.sqrt(eigvals)).dot(eigvecs.T)
    assert np.allclose(R.T.dot(R) - H, 0.) # ensures R' R is a decomposition of H
    add_linear_equality(prog, x_aux, R.dot(x))
    cons = prog.addConstr(.5 * x_aux.dot(x_aux) <= y * z),
    return cons, x_aux
    
def get_constraint_set(prog):
    '''
    Returns the linear constraints of prog in the form A x < = b.
    '''
    cs = prog.getConstrs()
    vs = prog.getVars()
    v_id = {v: i for i, v in enumerate(vs)}
    nv = len(vs)
    CS = Polyhedron(np.zeros((0, nv)), np.zeros((0, 1)))
    for i, v in enumerate(vs):
        if v.getAttr(grb.GRB.Attr.LB) != -grb.GRB.INFINITY:
            CS.add_lower_bound(v.getAttr(grb.GRB.Attr.LB), [i])
        if v.getAttr(grb.GRB.Attr.UB) != grb.GRB.INFINITY:
            CS.add_upper_bound(v.getAttr(grb.GRB.Attr.UB), [i])
    for i, c in enumerate(cs):
        expr = prog.getRow(c)
        Ai = np.zeros((1, nv))
        for j in range(expr.size()):
            Ai[0, v_id[expr.getVar(j)]] = expr.getCoeff(j)
        bi = np.vstack([c.getAttr(grb.GRB.Attr.RHS)])
        if c.getAttr(grb.GRB.Attr.Sense) in ['<', '=']:
            CS.add_inequality(Ai, bi)
        if c.getAttr(grb.GRB.Attr.Sense) in ['>', '=']:
            CS.add_inequality(-Ai, -bi)
    return CS

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
        if  prog.objVal - P.b[i,0] < tol:
            prog.remove(cons[i])
            minimal_facets.remove(i)
        else:
            cons[i].RHS -= 1.

    return Polyhedron(P.A[minimal_facets,:], P.b[minimal_facets,:])

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

def convex_hull_method_fast(P, resiudal_dimensions):

    # reorder coordinates
    n = len(resiudal_dimensions)
    dropped_dimensions = [i for i in range(P.A.shape[1]) if i not in resiudal_dimensions]
    A = np.hstack((
        P.A[:, resiudal_dimensions],
        P.A[:, dropped_dimensions]
        ))
    b = P.b

    # intialize program
    prog = grb.Model()
    x = add_vars(prog, A.shape[1])
    prog.update()
    cons = add_linear_inequality(prog, A.dot(x), b)
    prog.update()
    prog.setParam('OutputFlag', 0)

    # initialize projection
    vertices = _get_two_vertices(prog, x, n)
    if n == 1:
        E = np.array([[1.],[-1.]])
        f = np.array([
            [max(v[0,0] for v in vertices)],
            [- min(v[0,0] for v in vertices)]
            ])
        return E, f, vertices
    vertices = _get_inner_simplex(prog, x, vertices)

    # expand facets
    hull = ConvexHull(
        np.hstack(vertices).T,
        incremental=True
        )
    hull = _expand_simplex(prog, x, hull)
    hull.close()

    # get outputs
    E = hull.equations[:, :-1]
    f = - hull.equations[:, -1:]
    vertices = [np.reshape(v, (v.shape[0], 1)) for v in hull.points]

    proj = Polyhedron(E, f)
    proj._vertices = vertices

    return proj

def _get_two_vertices(prog, x, n):

    # select any direction to explore (it has to belong to the projected space, i.e. a_i = 0 for all i > n)
    a = np.zeros(len(x))
    a[0] = 1.

    # minimize and maximize in the given direction
    vertices = []
    for f in [a, -a]:
        prog.setObjective(f.dot(x))
        prog.optimize()
        v = np.vstack([xi.x for xi in x[:n]])
        vertices.append(v)

    return vertices

def _get_inner_simplex(prog, x, vertices, tol=1.e-7):

    # initialize LPs
    n = vertices[0].shape[0]

    # expand increasing at every iteration the dimension of the space
    for i in range(2, n+1):
        a, d = plane_through_points([v[:i,:] for v in vertices])
        f = np.vstack((a, np.zeros((len(x)-i, 1))))
        prog.setObjective(f.flatten().dot(x))
        prog.optimize()
        argmin = np.vstack([xi.x for xi in x])

        # check the length of the expansion wrt to the plane, if zero expand in the opposite direction
        expansion = np.abs(a.T.dot(argmin[:i, :]) - d) # >= 0
        if expansion < tol:
            prog.setObjective(-f.flatten().dot(x))
            prog.optimize()
            argmin = np.vstack([xi.x for xi in x])
        vertices.append(argmin[:n,:])

    return vertices

def _expand_simplex(prog, x, hull, tol=1.e-7):

    # initialize algorithm's variables
    n = hull.points[0].shape[0]
    a_explored = []

    # start convex-hull method
    convergence = False
    while not convergence:
        convergence = True

        # check if every facet of the inner approximation belongs to the projection
        for i in range(hull.equations.shape[0]):

            # get normalized halfplane {x | a' x <= d} of the ith facet
            a = hull.equations[i:i+1, :-1].T
            d = - hull.equations[i, -1]
            a_norm = np.linalg.norm(a)
            a /= a_norm
            d /= a_norm

            # check it the direction a has been explored so far
            is_explored = any((np.allclose(a, a2) for a2 in a_explored))
            if not is_explored:
                a_explored.append(a)

                # maximize in the direction a
                f = np.vstack((
                    - a,
                    np.zeros((len(x)-n, 1))
                    ))
                prog.setObjective(f.flatten().dot(x))
                prog.optimize()
                argmin = np.vstack([xi.x for xi in x])

                # check if expansion wrt to the halfplane is greater than zero
                expansion = - prog.objVal - d # >= 0
                if expansion > tol:
                    convergence = False
                    hull.add_points(argmin[:n,:].T)
                    break

    return hull