@staticmethod
def minimum_cardinality_active_set_completion(qp, active_set_fragment, inactive_set_fragment):

    # gurobi model
    model = grb.Model()

    # program dimensions
    n_z = qp.H.shape[0]
    n_x = qp.S.shape[1]
    n_l = qp.G.shape[0]
    n_d = n_l - len(active_set_fragment) - len(inactive_set_fragment)

    # parameters
    bigM = 1.e3
    delta_x = np.random.rand(n_x, 1)*1.e-5

    # list of non determined complementarities
    free_constraints = list(set(range(n_l)) - set(active_set_fragment + inactive_set_fragment))
    free_constraints.sort()

    # variables
    z = model.addVars(n_z, 2, lb=[[- grb.GRB.INFINITY]*n_z]*2, name='z')
    l = model.addVars(n_l, 2, lb=[[- grb.GRB.INFINITY]*n_l]*2, name='l')
    x = model.addVars(n_x, lb=[- grb.GRB.INFINITY]*n_x, name='x')
    d = model.addVars(n_d, vtype=grb.GRB.BINARY, name='d')

    # numpy variables
    z_np = np.array([[z[i,j] for j in range(2)] for i in range(n_z)])
    l_np = np.array([[l[i,j] for j in range(2)] for i in range(n_l)])
    x_np = np.array([[x[i]] for i in range(n_x)])
    x_np = np.hstack((x_np, x_np+delta_x))

    # set objective
    V = sum([d[i] for i in range(n_d)])
    model.setObjective(V)

    # stationarity
    expr = qp.H.dot(z_np) + qp.G.T.dot(l_np)
    model.addConstrs((expr[i,j] == 0. for i in range(n_z) for j in range(2)))

    # primal feasibility determined complementarities
    expr = qp.G.dot(z_np) - np.hstack((qp.W, qp.W)) - qp.S.dot(x_np)
    model.addConstrs((expr[i,j] == 0. for i in active_set_fragment for j in range(2)))
    model.addConstrs((expr[i,j] <= 0. for i in inactive_set_fragment for j in range(2)))

    # primal feasibility undetermined complementarities
    model.addConstrs((expr[i,j] <= 0. for i in free_constraints for j in range(2)))
    model.addConstrs((expr[i,j] >= bigM*(d[k]-1) for k, i in enumerate(free_constraints) for j in range(2)))
    
    # dual feasibility determined complementarities
    model.addConstrs((l[i,j] >= 0. for i in active_set_fragment for j in range(2)))
    model.addConstrs((l[i,j] == 0. for i in inactive_set_fragment for j in range(2)))

    # dual feasibility undetermined complementarities
    model.addConstrs((l[i,j] >= 0. for i in free_constraints for j in range(2)))
    model.addConstrs((l[i,j] <= bigM*d[k] for k, i in enumerate(free_constraints) for j in range(2)))

    # run optimization
    model.setParam('OutputFlag', False)
    model.optimize()

    # return active set
    if model.status != grb.GRB.Status.OPTIMAL:
        active_set = None
        x_star = np.full((n_x,1), np.nan)
    else:
        z_star = np.array([[model.getAttr('x', z)[i,0]] for i in range(n_z)])
        l_star = np.array([[model.getAttr('x', l)[i,0]] for i in range(n_l)])
        x_star = np.array([[model.getAttr('x', x)[i]] for i in range(n_x)])
        d_star = [int(round(model.getAttr('x', d)[i])) for i in range(n_d)]
        active_set_completion = [free_constraints[i] for i, d in enumerate(d_star) if d == 1]
        active_set = (active_set_fragment + active_set_completion)
        active_set.sort()

        # check big-Ms
        residuals = qp.G.dot(z_star) - qp.W - qp.S.dot(x_star)
        residuals_bigM = residuals[free_constraints]
        multipliers_bigM = l_star[free_constraints]
        if any(residuals_bigM < -.9*bigM) or any(multipliers_bigM > .9*bigM):
            raise ValueError('BigM too small!')

    return active_set, x_star