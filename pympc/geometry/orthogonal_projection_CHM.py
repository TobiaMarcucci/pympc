import time
import numpy as np
from itertools import combinations
from pympc.geometry.chebyshev_center import chebyshev_center
from pympc.optimization.pnnls import linear_program

def orthogonal_projection_CHM(A, b, resiudal_dimensions):
    """
    This is an implementation of the Convex Hull Method for orthogonal projections of polytopes (see, e.g., http://www.ece.drexel.edu/walsh/JayantCHM.pdf).

    Inputs:
        - A, b: H-representation of the polytope P := {x | A x <= b}
        - resiudal_dimensions: list of integers defining the dimensions of the projected polytope.

    Outputs:
        - G, g: non-redundant H-representation of the projection of P
        - v_proj: list of the vertices of the projection of P
    """

    # change of coordinates
    tic = time.time()
    center = chebyshev_center(A, b)[0]
    b = b - A.dot(center) # don't do -= otherwise b is modifies also outside the function!
    dropped_dimensions = sorted(set(range(A.shape[1])) - set(resiudal_dimensions))
    A = np.hstack((A[:, resiudal_dimensions], A[:, dropped_dimensions]))
    n_proj = len(resiudal_dimensions)

    # projection
    v_proj = first_two_points(A, b, n_proj)
    v_proj = inner_simplex(A, b, v_proj)
    halfspaces = halfspaces_simplex(v_proj)
    G, g, v_proj = expand_simplex(A, b, v_proj, halfspaces)

    # back to original coordinates
    g += G.dot(center[resiudal_dimensions,:])
    v_proj = [v + center[resiudal_dimensions,:] for v in v_proj]

    print('Projection successfully computed in ' + str(time.time()-tic) + ' seconds: number of facets is ' + str(G.shape[0]) + ', number of vertices is ' + str(len(v_proj)) + '.')
    return G, g, v_proj

def first_two_points(A, b, n_proj):

    v_proj = []
    a = np.zeros((A.shape[1], 1))
    a[0,0] = 1.
    for a in [a, -a]:
        sol = linear_program(a, A, b)
        v_proj.append(sol.argmin[:n_proj,:])

    print('Projection algorithm initiliazed with two vertices.')
    return v_proj

def inner_simplex(A, b, v_proj, tol=1.e-7):

    n_proj = v_proj[0].shape[0]
    for i in range(2, n_proj+1):
        a, d = plane_through_points([v[:i,:] for v in v_proj])
        a = np.vstack((a, np.zeros((A.shape[1]-i, 1))))
        sol = linear_program(-a, A, b)
        if -sol.min < d[0,0] + tol:
            a = -a
            sol = linear_program(-a, A, b)
        v_proj.append(sol.argmin[:n_proj,:])

    print('Inner simplicial approximation with ' + str(len(v_proj)) + ' vertices found.')
    return v_proj

def halfspaces_simplex(v_proj):

    halfspaces = []
    for i, v in enumerate(v_proj):
        vertices = [u for j,u in enumerate(v_proj) if j!=i]
        a, d = plane_through_points([u for j, u in enumerate(v_proj) if j!=i])
        if a.T.dot(v)[0,0] < d[0,0]:
            halfspaces.append([a, d, vertices])
        else:
            halfspaces.append([-a, -d, vertices])
 
    print('H-representation of the inner simplex derived, number of halfspaces is ' + str(len(halfspaces)) + '.')
    return halfspaces

def expand_simplex(A, b, v_proj, halfspaces, tol=1.e-7):

    n_proj = v_proj[0].shape[0]
    G = np.zeros((0, n_proj))
    g = np.zeros((0, 1))
    residual = np.nan
    while halfspaces:
        hs = halfspaces[0]
        a = np.zeros((A.shape[1], 1))
        a[:n_proj,:] = hs[0]
        sol = linear_program(-a, A, b)
        if - sol.min - hs[1][0,0] < tol:
            G, g = add_facet(hs[0], hs[1], G, g)
            del halfspaces[0]
        else:
            residual = '%e' % (- sol.min - hs[1][0,0])
            new_vertex = sol.argmin[:n_proj,:]
            violated_halfspaces = []
            satisfied_halfspaces = []
            for plane in halfspaces:
                if (plane[0].T.dot(new_vertex) - plane[1])[0,0] > tol:
                    violated_halfspaces.append(plane)
                else:
                    satisfied_halfspaces.append(plane)
            halfspaces = satisfied_halfspaces
            for plane in violated_halfspaces:
                for vertices in combinations(plane[2], n_proj-1):
                    vertices = list(vertices) + [new_vertex]
                    a, d = plane_through_points(vertices)
                    residuals = [(a.T.dot(v) - d)[0,0] for v in v_proj]
                    if max(residuals) < tol:
                        halfspaces.append([a, d, vertices])
                    elif min(residuals) > -tol:
                        halfspaces.append([-a, -d, vertices])
            v_proj.append(new_vertex)
        print('Facets found so far ' + str(G.shape[0]) + ', vertices found so far ' + str(len(v_proj)) + ', length of the last inflation ' + str(residual) + '.\r'),
    print('\n'),

    return G, g, v_proj

def add_facet(a, d, G, g, tol=1.e-5):

    new_row = np.hstack((a.T, d))
    new_row = new_row.flatten()/np.linalg.norm(new_row)
    Gg = np.hstack((G, g))
    duplicate = False
    for row in Gg:
        row = row/np.linalg.norm(row)
        if np.allclose(new_row, row, atol=tol):
            duplicate = True
            break
    if not duplicate:
        G = np.vstack((G, a.T))
        g = np.vstack((g, d))

    return G, g

def plane_through_points(points):
    """
    Returns the plane a^T x = b passing through the input points. It first adds a random offset to be shure that the matrix of the points is invertible (it wouldn't be the case if the plane we are looking for passes through the origin). The a vector has norm equal to one.
    """

    offset = np.random.rand(points[0].shape[0],1)
    points = [p + offset for p in points]
    P = np.hstack(points).T
    a = np.linalg.solve(P, np.ones(offset.shape))
    b = 1. - a.T.dot(offset)
    a_norm = np.linalg.norm(a)
    a /= a_norm
    b /= a_norm

    return a, b






# import time
# import numpy as np
# from itertools import combinations
# from pympc.geometry.chebyshev_center import chebyshev_center
# from pympc.geometry.nullspace_basis import nullspace_basis
# from pympc.optimization.pnnls import linear_program



# def orthogonal_projection_CHM(A, b, resiudal_dimensions):
#     """
#     This is an implementation of the Convex Hull Method for orthogonal projections of polytopes (see, e.g., http://www.ece.drexel.edu/walsh/JayantCHM.pdf).

#     Inputs:
#         - A, b: H-representation of the polytope P := {x | A x <= b}
#         - resiudal_dimensions: list of integers defining the dimensions of the projected polytope.

#     Outputs:
#         - G, g: non-redundant H-representation of the projection of P
#         - v_proj: list of the vertices of the projection of P
#     """

#     # change of coordinates
#     tic = time.time()
#     center = chebyshev_center(A, b)[0]
#     b -= A.dot(center)
#     dropped_dimensions = sorted(set(range(A.shape[1])) - set(resiudal_dimensions))
#     A = np.hstack((A[:, resiudal_dimensions], A[:, dropped_dimensions]))
#     n_proj = len(resiudal_dimensions)

#     # projection
#     v_proj = first_two_points(A, b, n_proj)
#     v_proj = inner_simplex(A, b, v_proj)
#     halfspaces = halfspaces_simplex(v_proj)
#     G, g, v_proj = expand_simplex(A, b, v_proj, halfspaces)

#     # back to original coordinates
#     g += G.dot(center[resiudal_dimensions,:])
#     v_proj = [v + center[resiudal_dimensions,:] for v in v_proj]

#     print('Projection successfully computed in ' + str(time.time()-tic) + ' seconds: number of facets is ' + str(G.shape[0]) + ', number of vertices is ' + str(len(v_proj)) + '.')
#     return G, g, v_proj

# def first_two_points(A, b, n_proj):

#     v_proj = []
#     a = np.zeros((A.shape[1], 1))
#     a[0,0] = 1.
#     for a in [a, -a]:
#         sol = linear_program(a, A, b)
#         v_proj.append(sol.argmin[:n_proj,:])

#     print('Projection algorithm initiliazed with two vertices.')
#     return v_proj

# def inner_simplex(A, b, v_proj, tol=1.e-7):

#     n_proj = v_proj[0].shape[0]
#     for i in range(2, n_proj+1):
#         a = normal_vector([v[:i,:] for v in v_proj])
#         a = np.vstack((a, np.zeros((A.shape[1]-i, 1))))
#         d = a[:n_proj,:].T.dot(v_proj[0])
#         sol = linear_program(-a, A, b)
#         if -sol.min < d[0,0] + tol:
#             a = -a
#             sol = linear_program(-a, A, b)
#         v_proj.append(sol.argmin[:n_proj,:])

#     print('Inner simplicial approximation with ' + str(len(v_proj)) + ' vertices found.')
#     return v_proj

# def halfspaces_simplex(v_proj):

#     halfspaces = []
#     for i, v in enumerate(v_proj):
#         vertices = [u for j,u in enumerate(v_proj) if j!=i]
#         a = normal_vector([u for j, u in enumerate(v_proj) if j!=i])
#         d = a.T.dot(v_proj[i-1])
#         if a.T.dot(v)[0,0] < d[0,0]:
#             halfspaces.append([a, d, vertices])
#         else:
#             halfspaces.append([-a, -d, vertices])
 
#     print('H-representation of the inner simplex derived, number of halfspaces is ' + str(len(halfspaces)) + '.')
#     return halfspaces

# def expand_simplex(A, b, v_proj, halfspaces, tol=1.e-7):

#     n_proj = v_proj[0].shape[0]
#     G = np.zeros((0, n_proj))
#     g = np.zeros((0, 1))
#     while halfspaces:
#         hs = halfspaces[0]
#         a = np.zeros((A.shape[1], 1))
#         a[:n_proj,:] = hs[0]
#         sol = linear_program(-a, A, b)
#         if -sol.min < hs[1][0,0] + tol:
#             G, g = add_facet(hs[0], hs[1], G, g)
#             del halfspaces[0]
#         else:
#             new_vertex = sol.argmin[:n_proj,:]
#             violated_halfspaces = []
#             satisfied_halfspaces = []
#             for plane in halfspaces:
#                 if (plane[0].T.dot(new_vertex) - plane[1])[0,0] > tol:
#                     violated_halfspaces.append(plane)
#                 else:
#                     satisfied_halfspaces.append(plane)
#             halfspaces = satisfied_halfspaces
#             for plane in violated_halfspaces:
#                 for vertices in combinations(plane[2], n_proj-1):
#                     vertices = list(vertices) + [new_vertex]
#                     a = normal_vector(vertices)
#                     d = a.T.dot(new_vertex)
#                     residuals = [(a.T.dot(v) - d)[0,0] for v in v_proj]
#                     if max(residuals) < tol:
#                         halfspaces.append([a, d, vertices])
#                     elif min(residuals) > -tol:
#                         halfspaces.append([-a, -d, vertices])
#             v_proj.append(new_vertex)
#         print('Facets found so far ' + str(G.shape[0]) + ', vertices found so far ' + str(len(v_proj)) + '.\r'),
#     print('\n'),

#     return G, g, v_proj

# def add_facet(a, d, G, g, tol=1.e-5):

#     new_row = np.hstack((a.T, d))
#     new_row = new_row.flatten()/np.linalg.norm(new_row)
#     Gg = np.hstack((G, g))
#     duplicate = False
#     for row in Gg:
#         row = row/np.linalg.norm(row)
#         if np.allclose(new_row, row, atol=tol):
#             duplicate = True
#             break
#     if not duplicate:
#         G = np.vstack((G, a.T))
#         g = np.vstack((g, d))

#     return G, g

# def normal_vector(points):

#     points_relative = [p - points[0] for p in points[1:]]
#     P = np.hstack(points_relative).T

#     return nullspace_basis(P)