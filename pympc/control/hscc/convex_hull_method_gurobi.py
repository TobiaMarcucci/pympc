# external imports
import numpy as np
import gurobipy as grb
from scipy.spatial import ConvexHull

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.geometry.utils import plane_through_points
from pympc.control.hybrid_benchmark.utils import (add_vars,
                                                  add_linear_inequality
                                                  )

def convex_hull_method(P, resiudal_dimensions):

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
            max(v[0] for v in vertices),
            - min(v[0] for v in vertices)
            ])
        return E, f, vertices
    vertices = _get_inner_simplex(prog, x, vertices)

    # expand facets
    hull = ConvexHull(
        np.vstack(vertices),
        incremental=True
        )
    hull = _expand_simplex(prog, x, hull)
    hull.close()

    # get outputs
    E = hull.equations[:, :-1]
    f = - hull.equations[:, -1]
    vertices = [v for v in hull.points]

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
        v = np.array([xi.x for xi in x[:n]])
        vertices.append(v)

    return vertices

def _get_inner_simplex(prog, x, vertices, tol=1.e-7):

    # initialize LPs
    n = vertices[0].shape[0]

    # expand increasing at every iteration the dimension of the space
    for i in range(2, n+1):
        a, d = plane_through_points([v[:i] for v in vertices])
        f = np.concatenate((a, np.zeros(len(x)-i)))
        prog.setObjective(f.dot(x))
        prog.optimize()
        argmin = np.array([xi.x for xi in x])

        # check the length of the expansion wrt to the plane, if zero expand in the opposite direction
        expansion = np.abs(a.dot(argmin[:i]) - d) # >= 0
        if expansion < tol:
            prog.setObjective(-f.dot(x))
            prog.optimize()
            argmin = np.array([xi.x for xi in x])
        vertices.append(argmin[:n])

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
            a = hull.equations[i, :-1]
            d = - hull.equations[i, -1]
            a_norm = np.linalg.norm(a)
            a /= a_norm
            d /= a_norm

            # check it the direction a has been explored so far
            is_explored = any((np.allclose(a, a2) for a2 in a_explored))
            if not is_explored:
                a_explored.append(a)

                # maximize in the direction a
                f = np.concatenate((
                    - a,
                    np.zeros(len(x)-n)
                    ))
                prog.setObjective(f.dot(x))
                prog.optimize()
                argmin = np.array([xi.x for xi in x])

                # check if expansion wrt to the halfplane is greater than zero
                expansion = - prog.objVal - d # >= 0
                if expansion > tol:
                    convergence = False
                    hull.add_points(argmin[:n].reshape((1,n)))
                    break

    return hull