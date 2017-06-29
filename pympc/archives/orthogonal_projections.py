def orthogonal_projection_from_vertices(self, dim_proj):
    """
    Projects the polytope in the given directions: from H-rep to V-rep, keeps the component of the vertices in the projected dimensions, from V-rep to H-rep.
    """
    
    vertices_proj = np.hstack(self.vertices).T[:,dim_proj]
    if len(dim_proj) > 1:
        hull = spatial.ConvexHull(vertices_proj)
        lhs = np.array(hull.equations)[:, :-1]
        rhs = - (np.array(hull.equations)[:, -1]).reshape((lhs.shape[0], 1))
    else:
        lhs = np.array([[1.],[-1.]])
        rhs = np.array([[max(vertices_proj)[0]],[-min(vertices_proj)[0]]])
    projected_polytope = Polytope(lhs, rhs)
    projected_polytope.assemble()
    return projected_polytope



def fourier_motzkin_elimination(self, variable_index, tol=1.e-12):
    [n_facets, n_variables] = self.lhs_min.shape
    lhs_leq = np.zeros((0, n_variables-1))
    rhs_leq = np.zeros((0, 1))
    lhs_geq = np.zeros((0, n_variables-1))
    rhs_geq = np.zeros((0, 1))
    lhs = np.zeros((0, n_variables-1))
    rhs = np.zeros((0, 1))
    for i in range(n_facets):
        pivot = self.lhs_min[i, variable_index]
        lhs_row = np.hstack((self.lhs_min[i,:variable_index], self.lhs_min[i,variable_index+1:]))
        rhs_row = self.rhs_min[i, 0]
        if pivot > tol:
            lhs_leq = np.vstack((lhs_leq, lhs_row/pivot))
            rhs_leq = np.vstack((rhs_leq, rhs_row/pivot))
        elif pivot < -tol:
            lhs_geq = np.vstack((lhs_geq, lhs_row/pivot))
            rhs_geq = np.vstack((rhs_geq, rhs_row/pivot))
        else:
            lhs = np.vstack((lhs, lhs_row))
            rhs = np.vstack((rhs, rhs_row))
    for i in range(lhs_leq.shape[0]):
        for j in range(lhs_geq.shape[0]):
            lhs_row = lhs_leq[i,:] - lhs_geq[j,:]
            rhs_row = rhs_leq[i,0] - rhs_geq[j,0]
            lhs = np.vstack((lhs, lhs_row))
            rhs = np.vstack((rhs, rhs_row))
    p = Polytope(lhs,rhs)
    p.assemble()
    return p


def orthogonal_projection_fourier_motzkin(self, projection_dimensions):
    """
    Projects the polytope in the given directions: from H-rep to V-rep, keeps the component of the vertices in the projected dimensions, from V-rep to H-rep.
    """
    remove_dimensions = sorted(list(set(range(self.lhs_min.shape[1])) - set(projection_dimensions)))
    p = self
    for dimension in reversed(remove_dimensions):
        p = p.fourier_motzkin_elimination(dimension)
    return p



######## ESP method



def orthogonal_projection_esp(A, b, residual_dimensions):

    # translate polytope to contain the origin
    center = chebyshev_center(A,b)[0]
    b -= A.dot(center)

    # split variables
    d = len(residual_dimensions)
    k = A.shape[1] - d
    removed_dimensions = [i for i in range(d+k) if i not in residual_dimensions]
    C = A[:,residual_dimensions]
    D = A[:,removed_dimensions]

    # shoot to find the starting equality set
    E_0, a_f, b_f = shoot(C, D, b)
    G = a_f.T
    g = b_f

    # initialize list of equality sets that project into a facet
    E_list = [E_0]

    # ridges of E_0
    E_r_list = rdg(E_0, a_f, b_f, C, D, b)

    # initialize search list
    L = [(i, (E_0, a_f, b_f)) for i in E_r_list]

    while len(L) > 0:

        # adjacent equality set to the first in the list
        E_adj, a_adj, b_adj = adj(L[0], C, D, b)

        # ridges of the adjacent equality set
        E_r_list = rdg(E_adj, a_adj, b_adj, C, D, b)

        # add(remove) couples to(from) the list
        for i in E_r_list:
            A_r_list = [j[0][0] for j in L]
            if i[0] in A_r_list:
                del L[A_r_list.index(i[0])]
            else:
                L.append((i, (E_adj, a_adj, b_adj)))

        # update H-rep of the projection
        G = np.vstack((G, a_adj.T))
        g = np.vstack((g, b_adj))
        E_list.append(E_adj)

    # translate back
    g += G.dot(center[residual_dimensions,:])

    return G, g, E_list

def shoot(C, D, b, max_iter=100):

    # loop to get a facet of the projection
    is_facet = False
    iter = 0
    while not is_facet:
        iter += 1
        if iter > max_iter:
            raise ValueError('Too many iteration in the shooting phase!')
    
        # random search direction
        gamma = np.random.randn(C.shape[1], 1)

        # maximize in direction of gamma until a face is found
        cost = np.vstack((np.zeros((D.shape[1], 1)), np.array([[-1.]])))
        lhs = np.hstack((D, C.dot(gamma)))
        sol = linear_program(cost, lhs, b)

        # check if it is a lower dimensional face
        is_facet = not(sol.primal_degenerate)

    # get result
    y_star = sol.argmin[:-1,:]
    r_star = sol.argmin[-1,0]

    # equality set and half space
    E_0 = sol.active_set
    a_f, b_f = affine_hull_from_equality_set(E_0, C, D, b)

    # check dual degeneracy
    if sol.dual_degenerate:
        print 'Dual-degenerate shoot, recovering equality set ...'
        lhs_eq = np.hstack((a_f.T, np.zeros((1, D.shape[1]))))
        E_0 = equality_set_from_polytope(C, D, b, lhs_eq, b_f)

    print 'Facet', E_0, 'shot'
    return E_0, a_f, b_f

def rdg(E, a_f, b_f, C, D, b):

    # partition of the matrices
    E_c = [i for i in range(C.shape[0]) if i not in E]
    D_E, C_E, b_E = D[E, :], C[E, :], b[E, :]
    D_E_c, C_E_c, b_E_c = D[E_c, :], C[E_c, :], b[E_c, :]

    #  check dimension of P_E
    if np.linalg.matrix_rank(D_E) < D.shape[1]:
        raise ValueError('n > 0')

    # matrices from lemma 31
    D_E_pinv = np.linalg.pinv(D_E)
    S = C - D.dot(D_E_pinv).dot(C_E)
    L = D.dot(nullspace_basis(D_E))
    t = b - D.dot(D_E_pinv).dot(b_E)

    # test elements of E_c to find ridges
    E_r_list = []
    for i in E_c:

        # check rank (first condition in proposition 35)
        mat = np.vstack((np.hstack((a_f.T, b_f)), np.hstack((S[i:i+1,:], t[i:i+1,:]))))
        if np.linalg.matrix_rank(mat) == 2:

            # derive set Q(i) as in equation (16)
            Q_i = []
            for j in E_c:
                mat = np.vstack((
                    np.hstack((a_f.T, b_f)),
                    np.hstack((S[[i,j], :], t[[i,j], :]))
                    ))
                if np.linalg.matrix_rank(mat) == 2:
                    Q_i.append(j)

            # definition of other index sets
            E_c_diff_Q_i = [k for k in E_c if k not in Q_i]
            Q_i_diff_i = [k for k in Q_i if k != i]

            # check existence of a x_0 (second condition in proposition 35)
            cost = np.vstack((np.zeros((C.shape[1], 1)), np.array([[1.]])))
            lhs_ineq = np.vstack((
                np.hstack((S[E_c_diff_Q_i, :], -np.ones((len(E_c_diff_Q_i), 1)))),
                np.hstack((S[Q_i_diff_i, :], -np.zeros((len(Q_i_diff_i), 1)))),
                np.hstack((np.zeros((1, C.shape[1])), np.array([[-1.]])))
                ))
            rhs_ineq = np.vstack((t[E_c_diff_Q_i, :], t[Q_i_diff_i, :], np.array([[1.]])))
            lhs_eq = np.vstack((
                np.hstack((a_f.T, np.array([[0.]]))),
                np.hstack((S[i:i+1,:], np.array([[0.]])))
                ))
            rhs_eq = np.vstack((b_f, t[i:i+1, :]))
            sol = linear_program(cost, lhs_ineq, rhs_ineq, lhs_eq, rhs_eq)
            tau_star = sol.argmin[-1,0]

            # check if all the conditions of proposition 35 are satisfied
            if tau_star < 0:
                a_r = S[i:i+1,:].T
                b_r = t[i:i+1, :]

                # normalize as in remark 36
                norm_factor = a_f.T.dot(a_r)[0,0]
                a_r -= norm_factor*a_f
                b_r -= norm_factor*b_f
                norm_factor = 1./np.linalg.norm(a_r) # the paper is wrong, don't change the sign here!
                a_r *= norm_factor
                b_r *= norm_factor
                E_r_list.append((sorted(Q_i + E), a_r, b_r))

    # eliminate duplicates
    E_r_list_min = []
    for i in E_r_list:
        E_r_list_min_0 = [j[0] for j in E_r_list_min]
        if i[0] not in E_r_list_min_0:
            E_r_list_min.append(i)

    print 'Ridges of facet', E, 'are', [i[0] for i in E_r_list_min]
    return E_r_list_min

def adj(L_0, C, D, b, eps=1.e-6):

    # unpack inputs
    [[E_r, a_r, b_r], [E_0, a_f, b_f]] = L_0
    C_E_r, D_E_r, b_E_r = C[E_r, :], D[E_r, :], b[E_r, :]
    print 'Crossing the ridge', E_r, 'from facet', E_0, '...',

    # set up lp
    cost = - np.vstack((a_r, np.zeros((D.shape[1], 1))))
    lhs_ineq = np.hstack((C_E_r, D_E_r))
    lhs_eq = np.hstack((a_f.T, np.zeros((1, D.shape[1]))))
    rhs_eq = b_f*(1. - eps)
    sol = linear_program(cost, lhs_ineq, b_E_r, lhs_eq, rhs_eq)
    if np.isnan(sol.min):
        raise ValueError('Unable to find adjacent facet for facet ' + str(E_0) + ' with ridge ' + str(E_r))
    x_star = sol.argmin[:C.shape[1],:]
    y_star = sol.argmin[C.shape[1]:,:]

    # equality set of adjacent facet
    if sol.dual_degenerate:
        gamma = - (a_r.T.dot(x_star) - b_r)[0,0]/(a_f.T.dot(x_star) - b_f)[0,0]
        lhs_eq = np.hstack((a_r.T + gamma*a_f.T, np.zeros((1, D.shape[1]))))
        rhs_eq = b_r + gamma*b_f
        E_adj = equality_set_from_polytope(C, D, b, lhs_eq, rhs_eq)
    else:
        E_adj = [E_r[i] for i in sol.active_set]

    # affine hull of adjacent facet
    a_adj, b_adj = affine_hull_from_equality_set(E_adj, C, D, b)

    print 'facet', E_adj, 'found'
    return E_adj, a_adj, b_adj

def equality_set_from_polytope(C, D, b, lhs_eq, rhs_eq, tol=1.e-5):
    E = []
    A = np.hstack((C, D))
    for i in range(A.shape[0]):
        sol = linear_program(A[i:i+1,:].T, A, b, lhs_eq, rhs_eq)
        if sol.min - b[i,0] > - tol:
            E.append(i)
    return E

def affine_hull_from_equality_set(E, C, D, b_in):

    # partition of the matrices
    D_E, C_E, b_E = D[E, :], C[E, :], b_in[E, :]

    # halfspace of the face
    N_D_E_T = nullspace_basis(D_E.T)
    a = N_D_E_T.T.dot(C_E).T
    b = N_D_E_T.T.dot(b_E)

    # check the dimensions
    if a.shape[1] > 1:
        if np.linalg.matrix_rank(a) == 1:
            a, b = a[:,0:1], b[0:1,:]
        else:
            raise ValueError('The projection of the affine hull associated to the equality set ' + str(E) + ' generated more than one facet!')
    elif a.shape[1] == 0:
        raise ValueError('Numerica error, try increasing tol in equality_set_from_polytope() or the tollerance in the active set detection in the LP solver...')

    # normalize
    norm_factor = np.sign(b)/np.linalg.norm(a)
    a *= norm_factor
    b *= norm_factor

    return a, b