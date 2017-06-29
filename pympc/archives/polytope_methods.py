def assemble_light(self):
    if self.assembled:
        raise ValueError('Polytope already assembled, cannot assemble again!')
    self.assembled = True
    [self.n_facets, self.n_variables] = self.A.shape
    self.normalize()
    self.check_emptiness()
    if self.empty:
        return
    # self.find_minimal_facets_cddlib()
    # print 'shapes ccd',self.lhs_min.shape
    # self.find_minimal_facets()
    # print 'shapes me',self.lhs_min.shape
    self.lhs_min = self.A
    self.rhs_min = self.b
    self._facet_centers = [None] * self.n_facets
    self._facet_radii = [None] * self.n_facets
    self._vertices = None
    self._x_min = None
    self._x_max = None
    return

def find_minimal_facets_cddlib(self):
    M = np.hstack((self.A, -self.b))
    M = cdd.Matrix([list(m) for m in M])
    M.rep_type = cdd.RepType.INEQUALITY
    p = cdd.Polyhedron(M)
    print 'before canonicalize', p.get_generators()
    M.canonicalize()
    p = cdd.Polyhedron(M)
    print 'after canonicalize', p.get_generators()
    self.lhs_min = - np.array([list(M[i])[1:] for i in range(M.row_size)])
    self.rhs_min = np.array([list(M[i])[:1] for i in range(M.row_size)])
    return

def included_in_union_of(self, p_list):
    """
    Checks if the polytope is a subset of the union of the polytopes in p_list (returns True or False).
    """

    # check if it is included in one
    for p in p_list:
        if self.included_in(p):
            return True

    # list of polytope with which it has non-empty intersection
    intersection_list = []
    for i, p in enumerate(p_list):
        if self.intersect_with(p):
            intersection_list.append(i)

    # milp
    model = grb.Model()
    x, model = real_variable(model, [self.n_variables])

    # domain
    model = point_inside_polyhedron(model, self.lhs_min, self.rhs_min, x)

    # point not in polyhedra
    d = []
    for i in intersection_list:
        p_i = p_list[i]
        d_i = []
        for j in range(p_i.lhs_min.shape[0]):
            model, d_ij = iff_point_in_halfspace(model, p_i.lhs_min[j:j+1,:], p_i.rhs_min[j:j+1,:], x, self)
            d_i.append(d_ij)
        model.addConstr(sum(d_i) <= len(d_i) - 1.)
        d += d_i

    # objective
    model.setObjective(0.)

    # run optimization
    model.setParam('OutputFlag', False)
    model.optimize()

    # return solution
    inclusion = True
    if model.status == grb.GRB.Status.OPTIMAL:
        inclusion = False

    return inclusion