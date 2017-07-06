def backwards_reachability_analysis(self, switching_sequence):
    """
    Returns the feasible set (Polytope) for the given switching sequence.
    It consists in N orthogonal projections:
    X_N := { x | G x <= g }
    x_N = A_{N-1} x_{N-1} + B_{N-1} u_{N-1} + c_{N-1}
    X_{N-1} = { x | exists u: G A_{N-1} x <= g - G B_{N-1} u - G c_{N-1} and (x,u) \in D }
    and so on...
    Note that in this case mixed constraints in x and u are allowed.
    """

    # check that all the data are present
    if self.X_N is None:
        raise ValueError('A terminal constraint is needed for the backward reachability analysis!')
    if len(switching_sequence) != self.N:
        raise ValueError('Switching sequence not coherent with the controller horizon.')

    # start
    print('Backwards reachability analysis for the switching sequence ' + str(switching_sequence))
    tic = time.time()
    feasible_set = self.X_N

    # fix the switching sequence
    A_sequence = [self.sys.affine_systems[switch].A for switch in switching_sequence]
    B_sequence = [self.sys.affine_systems[switch].B for switch in switching_sequence]
    c_sequence = [self.sys.affine_systems[switch].c for switch in switching_sequence]
    D_sequence = [self.sys.domains[switch] for switch in switching_sequence]

    # iterations over the horizon
    for i in range(self.N-1,-1,-1):
        print i
        lhs_x = feasible_set.lhs_min.dot(A_sequence[i])
        lhs_u = feasible_set.lhs_min.dot(B_sequence[i])
        lhs = np.hstack((lhs_x, lhs_u))
        rhs = feasible_set.rhs_min - feasible_set.lhs_min.dot(c_sequence[i])
        feasible_set = Polytope(lhs, rhs)
        feasible_set.add_facets(D_sequence[i].lhs_min, D_sequence[i].rhs_min)
        feasible_set.assemble_light()
        feasible_set = feasible_set.orthogonal_projection(range(self.sys.n_x))

    print('Feasible set computed in ' + str(time.time()-tic) + ' s')
    return feasible_set



def backwards_reachability_analysis_from_orthogonal_domains(self, switching_sequence, X_list, U_list):
    """
    Returns the feasible set (Polytope) for the given switching sequence.
    It uses the set-relation approach presented in Scibilia et al. "On feasible sets for MPC and their approximations"
    It consists in:
    X_N := { x | G x <= g }
    X_{N-1} = A^(-1) ( X_N \oplus (- B U - c) ) \cap X
    and so on...
    where X and U represent the domain of the state and the input respectively, and \oplus the Minkowsky sum.
    """

    # check that all the data are present
    if self.X_N is None:
        raise ValueError('A terminal constraint is needed for the backward reachability analysis!')
    if len(switching_sequence) != self.N:
        raise ValueError('Switching sequence not coherent with the controller horizon.')

    # start
    print('Backwards reachability analysis for the switching sequence ' + str(switching_sequence))
    tic = time.time()
    feasible_set = self.X_N

    # fix the switching sequence
    A_sequence = [self.sys.affine_systems[switch].A for switch in switching_sequence]
    B_sequence = [self.sys.affine_systems[switch].B for switch in switching_sequence]
    c_sequence = [self.sys.affine_systems[switch].c for switch in switching_sequence]
    X_sequence = [X_list[switch] for switch in switching_sequence]
    U_sequence = [U_list[switch] for switch in switching_sequence]

    # iterations over the horizon
    for i in range(self.N-1,8,-1):
        print i

        # vertices of the linear transformation - B U - c
        trans_U_vertices = [- B_sequence[i].dot(v) - c_sequence[i] for v in U_sequence[i].vertices]

        # vertices of X_N \oplus (- B U - c)
        mink_sum = [v1 + v2 for v1 in feasible_set.vertices for v2 in trans_U_vertices]

        # vertices of the linear transformation A^(-1) ( X_N \oplus (- B U - c) )
        A_inv = np.linalg.inv(A_sequence[i])
        trans_mink_sum = [A_inv.dot(v) for v in mink_sum]


        hull = ConvexHull(np.hstack(trans_mink_sum).T)
        A = np.array(hull.equations)[:, :-1]
        b = - np.array(hull.equations)[:, -1:]

        # # remove "almost zeros" to avoid numerical errors in cddlib
        # trans_mink_sum = [clean_matrix(v) for v in trans_mink_sum]

        # try:
        #     M = np.hstack(trans_mink_sum).T
        #     M = cdd.Matrix([[1.] + list(m) for m in M])
        #     M.rep_type = cdd.RepType.GENERATOR
        #     print M.row_size
        #     M.canonicalize()
        #     print M.row_size
        #     p = cdd.Polyhedron(M)
        #     M_ineq = p.get_inequalities()
        #     # raise RuntimeError('Just to switch to qhull...')
        # except RuntimeError:

        #     # print len(trans_mink_sum)
        #     # M = np.hstack(trans_mink_sum).T
        #     # M = cdd.Matrix([[1.] + list(m) for m in M])
        #     # M.rep_type = cdd.RepType.GENERATOR
        #     # M.canonicalize()
        #     # trans_mink_sum = [np.array([[M[j][k]] for k in range(1,M.col_size)]) for j in range(M.row_size)]
        #     # print len(trans_mink_sum)

        #     print 'RuntimeError with cddlib: switched to qhull'
        #     hull = ConvexHull(np.hstack(trans_mink_sum).T)
        #     A = np.array(hull.equations)[:, :-1]
        #     b = - np.array(hull.equations)[:, -1:]
        #     print A.shape
        #     M_ineq = np.hstack((b, -A))
        #     M_ineq = cdd.Matrix([list(m) for m in M_ineq])
        #     M_ineq.rep_type = cdd.RepType.INEQUALITY

        # M_ineq.canonicalize()
        # A = - np.array([list(M_ineq[j])[1:] for j in range(M_ineq.row_size)])
        # b = np.array([list(M_ineq[j])[:1] for j in range(M_ineq.row_size)])
        # print A.shape

        # intersect with X
        feasible_set = Polytope(A, b)
        feasible_set.add_facets(X_sequence[i].lhs_min, X_sequence[i].rhs_min)
        feasible_set.assemble_light()

    print('Feasible set computed in ' + str(time.time()-tic) + ' s')
    return feasible_set

def plot_feasible_set(self, switching_sequence, **kwargs):
    feasible_set = self.backwards_reachability_analysis(switching_sequence)
    feasible_set.plot(**kwargs)
    plt.text(feasible_set.center[0], feasible_set.center[1], str(switching_sequence))
    return



class HybridPolicyLibrary:
    """
    library (dict)
        ss 0,...,n (dicts)
            'program' (parametric_qp instance)
            'feasible_set'
            'active_sets'
                active_set 0,...,m (dicts)
                    'x' (list of states)
                    'V' (list of optimal values)
    """

    def __init__(self, controller):
        self.controller = controller
        self.library = dict()
        return

    def sample_policy(self, n_samples, terminal_domain, X_list=None, U_list=None):
        n_unfeasible = 0
        for i in range(n_samples):
            x = self.random_sample()
            if not self.sampling_rejection(x):
                with suppress_stdout():
                    ss_star = self.controller.feedforward(x) [2]
                if not any(np.isnan(ss_star)):
                    ss_list = [ss_star]
                    ss_list += self.shift_switching_sequence(ss_list[0], terminal_domain)
                    for ss in ss_list:
                        if not self.library.has_key(ss):

                            # if X_list is None and U_list is None:
                            #     feasible_set = self.controller.backwards_reachability_analysis(ss)
                            # elif X_list is not None and U_list is not None:
                            #     feasible_set = self.controller.backwards_reachability_analysis_from_orthogonal_domains(ss, X_list, U_list)

                            prog = self.controller.condense_program(ss)
                            tic = time.time()
                            print('\nComputing feasible set for the switching sequence ' + str(ss) + ' ...')
                            fs = prog.feasible_set
                            print('... feasible set computed in ' + str(time.time()-tic) + ' seconds.\n')


                            self.library[ss] = {
                            'feasible_set': fs,
                            'program': prog,
                            'lower_bound': {'A': np.zeros((1, self.controller.sys.n_x)), 'b': np.zeros((1,1))},
                            'upper_bound': {'convex_hull': None, 'A': None, 'b': None},
                            'generating_point': (x, ss_star) ###### useless
                            }
                else:
                    print 'Sample', i, 'unfeasible'
                    n_unfeasible += 1
            else:
                print 'Sample', i, 'rejected'


        print('Detected ' + str(len(self.library)) + ' switching sequences.')
        print(str(n_unfeasible) + ' unfeasible samples out of ' + str(n_samples) + '.')
        return

    def random_sample(self):
        x = np.random.rand(self.controller.sys.n_x,1)
        x = np.multiply(x, (self.controller.sys.x_max - self.controller.sys.x_min)) + self.controller.sys.x_min
        return x

    def sampling_rejection(self, x):
        for ss_value in self.library.values():
            if ss_value['feasible_set'].applies_to(x):
                return True
        return False

    def add_vertices_of_feasible_regions(self, eps=1.e-4):

        print('Adding vertices of the feasible sets ...')
        for ss, ss_values in self.library.items():
            x_list = ss_values['feasible_set'].vertices
            x_list = [x + (ss_values['feasible_set'].center - x)*eps for x in x_list]
            xV_list = [np.vstack((x, ss_values['program'].solve(x)[1])) for x in x_list]

            ### add an auxiliary vertex when the number of samples in not enough to generate a simplex
            if len(xV_list) == xV_list[0].shape[0]:
                auxiliary_vertex = sum([x[:-1,:] for x in xV_list])/len(xV_list)
                max_cost = max([x[-1,0] for x in xV_list])
                auxiliary_vertex = np.vstack((auxiliary_vertex, np.array([[2.*max_cost]])))
                xV_list.append(auxiliary_vertex)

            # ss_values['upper_bound']['convex_hull'] = ConvexHull(np.hstack(xV_list).T, incremental=True)
            ss_values['upper_bound']['convex_hull'] = ConvexHull(xV_list)
            A, b = self.upper_bound_from_hull(ss_values['upper_bound']['convex_hull'])
            ss_values['upper_bound']['A'] = A
            ss_values['upper_bound']['b'] = b

        print('... vertices of the feasible sets added.')
            
        return

    def bound_optimal_value_functions(self, n_samples):

        for i in range(n_samples):
            x = self.random_sample()
            css = self.get_candidate_switching_sequences(x)

            # if there is more than 1 candidate
            if len(css) > 1:
                u_list = []
                V_list = []
                for ss in css:
                    u, V = self.library[ss]['program'].solve(x)
                    u_list.append(u)
                    V_list.append(V)
                V_star = min(V_list)
                ss_star = css[V_list.index(V_star)]

                # refine minimum upper bound
                # self.library[ss_star]['upper_bound']['convex_hull'].add_points(np.vstack((x, V_star)).T)
                self.library[ss_star]['upper_bound']['convex_hull'].add_points([np.vstack((x, V_star))])
                A, b = self.upper_bound_from_hull(self.library[ss_star]['upper_bound']['convex_hull'])
                self.library[ss_star]['upper_bound']['A'] = A
                self.library[ss_star]['upper_bound']['b'] = b

                # refine lower bounds
                for j, ss in enumerate(css):
                    lb = self.get_lower_bound(ss, x)
                    if ss != ss_star and lb < V_star:
                        active_set = self.library[ss]['program'].get_active_set(x, u_list[j])
                        [A, b] = self.library[ss]['program'].get_cost_sensitivity([x], active_set)[0]
                        self.library[ss]['lower_bound']['A'] = np.vstack((self.library[ss]['lower_bound']['A'], A))
                        self.library[ss]['lower_bound']['b'] = np.vstack((self.library[ss]['lower_bound']['b'], b))

        return

    def get_lower_bound(self, ss, x):
        if not self.library[ss]['feasible_set'].applies_to(x):
            raise ValueError('Switching sequence ' + str(ss) + ' is not feasible at x=' + str(x) + '.')
        return max((self.library[ss]['lower_bound']['A'].dot(x) + self.library[ss]['lower_bound']['b']).flatten())

    def get_upper_bound(self, ss, x):
        if not self.library[ss]['feasible_set'].applies_to(x):
            raise ValueError('Switching sequence ' + str(ss) + ' is not feasible at x = ' + str(x) + '.')
        return max((self.library[ss]['upper_bound']['A'].dot(x) + self.library[ss]['upper_bound']['b']).flatten())

    def get_feasible_switching_sequences(self, x):
        return [ss for ss, ss_values in self.library.items() if ss_values['feasible_set'].applies_to(x)]

    def get_candidate_switching_sequences(self, x, tol= 0.):
        fss = self.get_feasible_switching_sequences(x)
        if not fss:
            return fss
        lb = [self.get_lower_bound(ss, x) for ss in fss]
        ub = [self.get_upper_bound(ss, x) for ss in fss]
        if any([lb[i] > ub[i] for i in range(len(fss))]):
            real_cost = [self.library[ss]['program'].solve(x)[1] for ss in fss]
            raise ValueError('Wrong lower bound ' + str(lb) + ' or upper bound ' + str(ub) + ' for optimal values ' + str(real_cost) + '.')
        return [ss for i, ss in enumerate(fss) if lb[i] < min(ub) - tol]

    def feedforward(self, x):
        css = self.get_candidate_switching_sequences(x)
        # print len(css), 'candidate switching sequences'
        if not css:
            V_star = np.nan
            u_star = [np.full((self.controller.sys.n_u, 1), np.nan) for i in range(self.controller.N)]
            return u_star, V_star
        V_star = np.inf
        for ss in css:
            u, V = self.library[ss]['program'].solve(x)
            if V < V_star:
                V_star = V
                u_star = [u[i*self.controller.sys.n_u:(i+1)*self.controller.sys.n_u,:] for i in range(self.controller.N)]
        return u_star, V_star

    def feedback(self, x):
        return self.feedforward(x)[0][0]

    @staticmethod
    def shift_switching_sequence(ss, terminal_domain):
        return [ss[i:] + (terminal_domain,)*i for i in range(1,len(ss))]

    @staticmethod
    def upper_bound_from_hull(hull):
        """
        For all the facets that don't have vertical gradient, we start from the form a1 x1 + a2 x2 + an xn + ... <= b and we derive xn = - a1/an x1 - a2/an x2 - ... + b.
        The output is in the form xn = max(A x + b).
        """
        # A = np.array(hull.equations)[:, :-1]
        # b = (np.array(hull.equations)[:, -1]).reshape((A.shape[0], 1))
        A, b = hull.get_minimal_H_rep()
        facets_to_keep = [i for i in range(A.shape[0]) if A[i,-1] < 0.]
        A_new = np.zeros((0, A.shape[1]-1))
        b_new = np.zeros((0, 1))
        for i in facets_to_keep:
            A_new = np.vstack((A_new, - A[i:i+1,:-1]/A[i,-1]))
            # b_new = np.vstack((b_new, - b[i:i+1,:]/A[i,-1]))
            b_new = np.vstack((b_new, b[i:i+1,:]/A[i,-1]))
        return A_new, b_new

def clean_matrix(A, threshold=1.e-6):
    A_abs = np.abs(A)
    elements_to_remove = np.where(np.abs(A) < threshold)
    for i in range(elements_to_remove[0].shape[0]):
        A[elements_to_remove[0][i], elements_to_remove[1][i]] = 0.
    return A