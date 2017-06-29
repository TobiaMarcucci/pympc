import time
import sys, os
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import gurobipy as grb
from contextlib import contextmanager
from optimization.pnnls import linear_program
from optimization.gurobi import quadratic_program, real_variable
from geometry.polytope import Polytope
from dynamical_systems import DTAffineSystem, DTPWASystem
from optimization.mpqpsolver import MPQPSolver, CriticalRegion
from scipy.spatial import ConvexHull
import cdd



class MPCController:

    def __init__(self, sys, N, objective_norm, Q, R, P=None, X=None, U=None, X_N=None):
        self.sys = sys
        self.N = N
        self.objective_norm = objective_norm
        self.Q = Q
        self.R = R
        if P is None:
            self.P = Q
        else:
            self.P = P
        self.X = X
        self.U = U
        if X_N is None and X is not None:
            self.X_N = X
        else:
            self.X_N = X_N
        self.condense_program()
        self.critical_regions = None
        return

    def condense_program(self):
        c = np.zeros((self.sys.n_x, 1))
        a_sys = DTAffineSystem(self.sys.A, self.sys.B, c)
        sys_list = [a_sys]*self.N
        X_list = [self.X]*self.N
        U_list = [self.U]*self.N
        switching_sequence = [0]*self.N
        pwa_sys = DTPWASystem.from_orthogonal_domains(sys_list, X_list, U_list)
        self.condensed_program = OCP_condenser(pwa_sys, self.objective_norm, self.Q, self.R, self.P, self.X_N, switching_sequence)
        self.remove_intial_state_contraints()
        return

    def remove_intial_state_contraints(self, tol=1e-10):
        C_u_rows_norm = list(np.linalg.norm(self.condensed_program.C_u, axis=1))
        intial_state_contraints = [i for i, row_norm in enumerate(C_u_rows_norm) if row_norm < tol]
        if len(intial_state_contraints) > self.X.lhs_min.shape[0]:
            raise ValueError('Wrong number of zero rows in the constrinats')
        self.condensed_program.C_u = np.delete(self.condensed_program.C_u,intial_state_contraints, 0)
        self.condensed_program.C_x = np.delete(self.condensed_program.C_x,intial_state_contraints, 0)
        self.condensed_program.C = np.delete(self.condensed_program.C,intial_state_contraints, 0)
        return

    def feedforward(self, x0):
        u_feedforward, cost = self.condensed_program.solve(x0)
        u_feedforward = [u_feedforward[self.sys.n_u*i:self.sys.n_u*(i+1),:] for i in range(self.N)]
        if any(np.isnan(u_feedforward).flatten()):
            print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
        return u_feedforward, cost

    def feedback(self, x0):
        return self.feedforward(x0)[0][0]

    def get_explicit_solution(self):
        """
        Attention: since the method remove_intial_state_contraints() modifies the variables of condensed_program, I have to call remove_linear_terms() again!
        """
        self.condensed_program.remove_linear_terms()
        mpqp_solution = MPQPSolver(self.condensed_program)
        self.critical_regions = mpqp_solution.critical_regions
        return

    def feedforward_explicit(self, x0):
        if self.critical_regions is None:
            raise ValueError('Explicit solution not available, call .get_explicit_solution()')
        cr_x0 = self.critical_regions.lookup(x0)
        if cr_x0 is not None:
            u_feedforward = cr_x0.u_offset + cr_x0.u_linear.dot(x0)
            u_feedforward = [u_feedforward[self.sys.n_u*i:self.sys.n_u*(i+1),:] for i in range(self.N)]
            cost = .5*x0.T.dot(cr_x0.V_quadratic).dot(x0) + cr_x0.V_linear.dot(x0) + cr_x0.V_offset
            cost = cost[0,0]
        else:
            print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            u_feedforward = [np.full((self.sys.n_u, 1), np.nan) for i in range(self.N)]
            cost = np.nan
        return u_feedforward, cost

    def feedback_explicit(self, x0):
            return self.feedforward(x0)[0][0]

    def optimal_value_function(self, x0):
        if self.critical_regions is None:
            raise ValueError('Explicit solution not available, call .get_explicit_solution()')
        cr_x0 = self.critical_regions.lookup(x0)
        if cr_x0 is not None:
            cost = .5*x0.T.dot(cr_x0.V_quadratic).dot(x0) + cr_x0.V_linear.dot(x0) + cr_x0.V_offset
        else:
            #print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            cost = np.nan
        return cost

    # def group_critical_regions(self):
    #     self.u_offset_list = []
    #     self.u_linear_list = []
    #     self.cr_families = []
    #     for cr in self.critical_regions:
    #         cr_family = np.where(np.isclose(cr.u_offset[0], self.u_offset_list))[0]
    #         if cr_family and all(np.isclose(cr.u_linear[0,:], self.u_linear_list[cr_family[0]])):
    #             self.cr_families[cr_family[0]].append(cr)
    #         else:
    #             self.cr_families.append([cr])
    #             self.u_offset_list.append(cr.u_offset[0])
    #             self.u_linear_list.append(cr.u_linear[0,:])
    #     print 'Critical regions grouped in ', str(len(self.cr_families)), ' sets.'
    #     return



class MPCHybridController:

    def __init__(self, sys, N, objective_norm, Q, R, P, X_N):
        self.sys = sys
        self.N = N
        self.objective_norm = objective_norm
        self.Q = Q
        self.R = R
        self.P = P
        self.X_N = X_N
        self.compute_M_domains()
        self.compute_M_dynamics()
        return

    def compute_M_domains(self):
        """
        Denoting with s the number of affine systems, M_domains is a list with s elements, each element has other s elements, each one of these is a bigM vector.
        """
        self.M_domains = []
        for i, domain_i in enumerate(self.sys.domains):
            M_i = []
            for j, domain_j in enumerate(self.sys.domains):
                M_ij = []
                if i != j:
                    for k in range(domain_i.lhs_min.shape[0]):
                        sol = linear_program(-domain_i.lhs_min[k,:], domain_j.lhs_min, domain_j.rhs_min)
                        M_ijk = (- sol.min - domain_i.rhs_min[k])[0]
                        M_ij.append(M_ijk)
                M_ij = np.reshape(M_ij, (len(M_ij), 1))
                M_i.append(M_ij)
            self.M_domains.append(M_i)
        return

    def compute_M_dynamics(self):
        self.M_dynamics = []
        self.m_dynamics = []
        for i in range(self.sys.n_sys):
            M_i = []
            m_i = []
            lhs_i = np.hstack((self.sys.affine_systems[i].A, self.sys.affine_systems[i].B))
            rhs_i = self.sys.affine_systems[i].c
            for domain_j in self.sys.domains:
                M_ij = []
                m_ij = []
                for k in range(lhs_i.shape[0]):
                    sol = linear_program(-lhs_i[k,:], domain_j.lhs_min, domain_j.rhs_min)
                    M_ijk = (- sol.min + rhs_i[k])[0]
                    M_ij.append(M_ijk)
                    sol = linear_program(lhs_i[k,:], domain_j.lhs_min, domain_j.rhs_min)
                    m_ijk = (sol.min + rhs_i[k])[0]
                    m_ij.append(m_ijk)
                M_ij = np.reshape(M_ij, (len(M_ij), 1))
                m_ij = np.reshape(m_ij, (len(m_ij), 1))
                M_i.append(M_ij)
                m_i.append(np.array(m_ij))
            self.M_dynamics.append(M_i)
            self.m_dynamics.append(m_i)
        return

    def feedforward(self, x0, u_ws=None, x_ws=None, ss_ws=None):

        # gurobi model
        model = grb.Model()

        # variables
        x, model = real_variable(model, [self.N+1, self.sys.n_x])
        u, model = real_variable(model, [self.N, self.sys.n_u])
        z, model = real_variable(model, [self.N, self.sys.n_sys, self.sys.n_x])
        d = model.addVars(self.N, self.sys.n_sys, vtype=grb.GRB.BINARY, name='d')
        model.update()

        # warm start
        if x_ws is not None:
            for i in range(self.sys.n_x):
                for k in range(self.N+1):
                    x[k,i].setAttr('Start', x_ws[k][i,0])
            if ss_ws is not None:
                for i in range(self.sys.n_x):
                    for j in range(self.sys.n_sys):
                        for k in range(self.N):
                            if j == ss_ws[k]:
                                z[k,j,i].setAttr('Start', x_ws[k+1][i,0])
                            else:
                                z[k,j,i].setAttr('Start', 0.)
        if u_ws is not None:
            for i in range(self.sys.n_u):
                for k in range(self.N):
                    u[k,i].setAttr('Start', u_ws[k][i,0])
        if ss_ws is not None:
            for j in range(self.sys.n_sys):
                for k in range(self.N):
                    if j == ss_ws[k]:
                        d[k,j].setAttr('Start', 1)
                    else:
                        d[k,j].setAttr('Start', 0)

        # numpy variables (list of numpy matrices ordered in time)
        x_np = [np.array([[x[k,i]] for i in range(self.sys.n_x)]) for k in range(self.N+1)]
        u_np = [np.array([[u[k,i]] for i in range(self.sys.n_u)]) for k in range(self.N)]

        # set objective
        model = self.mip_objective(model, x_np, u_np)

        # initial condition
        model.addConstrs((x[0,i] == x0[i,0] for i in range(self.sys.n_x)))

        # set constraints
        model = self.mip_constraints(model, x_np, u_np, z, d)

        # set parameters
        model.setParam('OutputFlag', False)
        model.setParam(grb.GRB.Param.OptimalityTol, 1.e-9)
        model.setParam(grb.GRB.Param.FeasibilityTol, 1.e-9)
        model.setParam(grb.GRB.Param.IntFeasTol, 1.e-9)
        model.setParam(grb.GRB.Param.MIPGap, 1.e-9)

        # run optimization
        model.optimize()

        # return solution
        if model.status != grb.GRB.Status.OPTIMAL:
            print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            u_feedforward = [np.full((self.sys.n_u,1), np.nan) for k in range(self.N)]
            x_trajectory = [np.full((self.sys.n_x,1), np.nan) for k in range(self.N+1)]
            cost = np.nan
            switching_sequence = [np.nan]*self.N
        else:
            cost = model.objVal
            u_feedforward = [np.array([[model.getAttr('x', u)[k,i]] for i in range(self.sys.n_u)]) for k in range(self.N)]
            x_trajectory = [np.array([[model.getAttr('x', x)[k,i]] for i in range(self.sys.n_x)]) for k in range(self.N+1)]
            d_star = [np.array([[model.getAttr('x', d)[k,i]] for i in range(self.sys.n_sys)]) for k in range(self.N)]
            switching_sequence = [np.where(np.isclose(d, 1.))[0][0] for d in d_star]
        return u_feedforward, x_trajectory, tuple(switching_sequence), cost

    def mip_objective(self, model, x_np, u_np, x_ws=None, u_ws=None):

        # linear objective
        if self.objective_norm == 'one':
            phi = model.addVars(self.N+1, self.sys.n_x, name='phi')
            if x_ws is not None:
                for i in range(self.sys.n_x):
                    for k in range(self.N):
                        phi[k,i].setAttr('Start', self.Q[i,:].dot(x_ws[k]))
                    phi[self.N,i].setAttr('Start', self.P[i,:].dot(x_ws[self.N]))
            psi = model.addVars(self.N, self.sys.n_u, name='psi')
            if u_ws is not None:
                for i in range(self.sys.n_u):
                    for k in range(self.N):
                        psi[k,i].setAttr('Start', self.R[i,:].dot(u_ws[k]))
            model.update()
            V = 0.
            for k in range(self.N+1):
                for i in range(self.sys.n_x):
                    V += phi[k,i]
            for k in range(self.N):
                for i in range(self.sys.n_u):
                    V += psi[k,i]
            model.setObjective(V)
            for k in range(self.N):
                for i in range(self.sys.n_x):
                    model.addConstr(phi[k,i] >= self.Q[i,:].dot(x_np[k])[0])
                    model.addConstr(phi[k,i] >= - self.Q[i,:].dot(x_np[k])[0])
                for i in range(self.sys.n_u):
                    model.addConstr(psi[k,i] >= self.R[i,:].dot(u_np[k])[0])
                    model.addConstr(psi[k,i] >= - self.R[i,:].dot(u_np[k])[0])
            for i in range(self.sys.n_x):
                model.addConstr(phi[self.N,i] >= self.P[i,:].dot(x_np[self.N])[0])
                model.addConstr(phi[self.N,i] >= - self.P[i,:].dot(x_np[self.N])[0])

       # quadratic objective 
        elif self.objective_norm == 'two':
            V = 0.
            for k in range(self.N):
                V += x_np[k].T.dot(self.Q).dot(x_np[k]) + u_np[k].T.dot(self.R).dot(u_np[k])
            V += x_np[self.N].T.dot(self.P).dot(x_np[self.N])
            model.setObjective(V[0,0])

        return model

    def mip_constraints(self, model, x_np, u_np, z, d):

        with suppress_stdout():

            # disjuction
            for k in range(self.N):
                model.addConstr(np.sum([d[k,i] for i in range(self.sys.n_sys)]) == 1.)

            # relaxation of the domains
            for k in range(self.N):
                for i in range(self.sys.n_sys):
                    expr_xu = self.sys.domains[i].lhs_min.dot(np.vstack((x_np[k], u_np[k]))) - self.sys.domains[i].rhs_min
                    expr_M = np.sum([self.M_domains[i][j]*d[k,j] for j in range(self.sys.n_sys) if j != i], axis=0)
                    expr = expr_xu - expr_M
                    model.addConstrs((expr[j][0] <= 0. for j in range(len(expr))))

            # state transition
            for k in range(self.N):
                for j in range(self.sys.n_x):
                    expr = 0.
                    for i in range(self.sys.n_sys):
                        expr += z[k,i,j]
                    model.addConstr(x_np[k+1][j,0] == expr)

            # relaxation of the dynamics, part 1
            for k in range(self.N):
                for i in range(self.sys.n_sys):
                    expr_M = self.M_dynamics[i][i]*d[k,i]
                    expr_m = self.m_dynamics[i][i]*d[k,i]
                    for j in range(self.sys.n_x):
                        model.addConstr(z[k,i,j] <= expr_M[j,0])
                        model.addConstr(z[k,i,j] >= expr_m[j,0])
            
            # relaxation of the dynamics, part 2
            for k in range(self.N):
                for i in range(self.sys.n_sys):
                    expr = self.sys.affine_systems[i].A.dot(x_np[k]) + self.sys.affine_systems[i].B.dot(u_np[k]) + self.sys.affine_systems[i].c
                    expr_M = expr - np.sum([self.M_dynamics[i][j]*d[k,j] for j in range(self.sys.n_sys) if j != i], axis=0)
                    expr_m = expr - np.sum([self.m_dynamics[i][j]*d[k,j] for j in range(self.sys.n_sys) if j != i], axis=0)
                    for j in range(self.sys.n_x):
                        model.addConstr(z[k,i,j] >= expr_M[j,0])
                        model.addConstr(z[k,i,j] <= expr_m[j,0])

            # terminal constraint
            expr = self.X_N.lhs_min.dot(x_np[self.N]) - self.X_N.rhs_min
            model.addConstrs((expr[i,0] <= 0. for i in range(len(self.X_N.minimal_facets))))

        return model

    def feedback(self, x0):
        return self.feedforward(x0)[0][0]

    def condense_program(self, switching_sequence):
        tic = time.time()
        print('- Condensing the OCP for the switching sequence ' + str(switching_sequence) + ' ...')
        if len(switching_sequence) != self.N:
            raise ValueError('Switching sequence not coherent with the controller horizon.')
        prog = OCP_condenser(self.sys, self.objective_norm, self.Q, self.R, self.P, self.X_N, switching_sequence)
        print('- OCP condensed in ' + str(time.time() -tic ) + ' seconds.')
        return prog

    # def backwards_reachability_analysis(self, switching_sequence):
    #     """
    #     Returns the feasible set (Polytope) for the given switching sequence.
    #     It consists in N orthogonal projections:
    #     X_N := { x | G x <= g }
    #     x_N = A_{N-1} x_{N-1} + B_{N-1} u_{N-1} + c_{N-1}
    #     X_{N-1} = { x | exists u: G A_{N-1} x <= g - G B_{N-1} u - G c_{N-1} and (x,u) \in D }
    #     and so on...
    #     Note that in this case mixed constraints in x and u are allowed.
    #     """

    #     # check that all the data are present
    #     if self.X_N is None:
    #         raise ValueError('A terminal constraint is needed for the backward reachability analysis!')
    #     if len(switching_sequence) != self.N:
    #         raise ValueError('Switching sequence not coherent with the controller horizon.')

    #     # start
    #     print('Backwards reachability analysis for the switching sequence ' + str(switching_sequence))
    #     tic = time.time()
    #     feasible_set = self.X_N

    #     # fix the switching sequence
    #     A_sequence = [self.sys.affine_systems[switch].A for switch in switching_sequence]
    #     B_sequence = [self.sys.affine_systems[switch].B for switch in switching_sequence]
    #     c_sequence = [self.sys.affine_systems[switch].c for switch in switching_sequence]
    #     D_sequence = [self.sys.domains[switch] for switch in switching_sequence]

    #     # iterations over the horizon
    #     for i in range(self.N-1,-1,-1):
    #         print i
    #         lhs_x = feasible_set.lhs_min.dot(A_sequence[i])
    #         lhs_u = feasible_set.lhs_min.dot(B_sequence[i])
    #         lhs = np.hstack((lhs_x, lhs_u))
    #         rhs = feasible_set.rhs_min - feasible_set.lhs_min.dot(c_sequence[i])
    #         feasible_set = Polytope(lhs, rhs)
    #         feasible_set.add_facets(D_sequence[i].lhs_min, D_sequence[i].rhs_min)
    #         feasible_set.assemble_light()
    #         feasible_set = feasible_set.orthogonal_projection(range(self.sys.n_x))

    #     print('Feasible set computed in ' + str(time.time()-tic) + ' s')
    #     return feasible_set

    # def backwards_reachability_analysis_from_orthogonal_domains(self, switching_sequence, X_list, U_list):
    #     """
    #     Returns the feasible set (Polytope) for the given switching sequence.
    #     It uses the set-relation approach presented in Scibilia et al. "On feasible sets for MPC and their approximations"
    #     It consists in:
    #     X_N := { x | G x <= g }
    #     X_{N-1} = A^(-1) ( X_N \oplus (- B U - c) ) \cap X
    #     and so on...
    #     where X and U represent the domain of the state and the input respectively, and \oplus the Minkowsky sum.
    #     """

    #     # check that all the data are present
    #     if self.X_N is None:
    #         raise ValueError('A terminal constraint is needed for the backward reachability analysis!')
    #     if len(switching_sequence) != self.N:
    #         raise ValueError('Switching sequence not coherent with the controller horizon.')

    #     # start
    #     print('Backwards reachability analysis for the switching sequence ' + str(switching_sequence))
    #     tic = time.time()
    #     feasible_set = self.X_N

    #     # fix the switching sequence
    #     A_sequence = [self.sys.affine_systems[switch].A for switch in switching_sequence]
    #     B_sequence = [self.sys.affine_systems[switch].B for switch in switching_sequence]
    #     c_sequence = [self.sys.affine_systems[switch].c for switch in switching_sequence]
    #     X_sequence = [X_list[switch] for switch in switching_sequence]
    #     U_sequence = [U_list[switch] for switch in switching_sequence]

    #     # iterations over the horizon
    #     for i in range(self.N-1,8,-1):
    #         print i

    #         # vertices of the linear transformation - B U - c
    #         trans_U_vertices = [- B_sequence[i].dot(v) - c_sequence[i] for v in U_sequence[i].vertices]

    #         # vertices of X_N \oplus (- B U - c)
    #         mink_sum = [v1 + v2 for v1 in feasible_set.vertices for v2 in trans_U_vertices]

    #         # vertices of the linear transformation A^(-1) ( X_N \oplus (- B U - c) )
    #         A_inv = np.linalg.inv(A_sequence[i])
    #         trans_mink_sum = [A_inv.dot(v) for v in mink_sum]


    #         hull = ConvexHull(np.hstack(trans_mink_sum).T)
    #         A = np.array(hull.equations)[:, :-1]
    #         b = - np.array(hull.equations)[:, -1:]

    #         # # remove "almost zeros" to avoid numerical errors in cddlib
    #         # trans_mink_sum = [clean_matrix(v) for v in trans_mink_sum]

    #         # try:
    #         #     M = np.hstack(trans_mink_sum).T
    #         #     M = cdd.Matrix([[1.] + list(m) for m in M])
    #         #     M.rep_type = cdd.RepType.GENERATOR
    #         #     print M.row_size
    #         #     M.canonicalize()
    #         #     print M.row_size
    #         #     p = cdd.Polyhedron(M)
    #         #     M_ineq = p.get_inequalities()
    #         #     # raise RuntimeError('Just to switch to qhull...')
    #         # except RuntimeError:

    #         #     # print len(trans_mink_sum)
    #         #     # M = np.hstack(trans_mink_sum).T
    #         #     # M = cdd.Matrix([[1.] + list(m) for m in M])
    #         #     # M.rep_type = cdd.RepType.GENERATOR
    #         #     # M.canonicalize()
    #         #     # trans_mink_sum = [np.array([[M[j][k]] for k in range(1,M.col_size)]) for j in range(M.row_size)]
    #         #     # print len(trans_mink_sum)

    #         #     print 'RuntimeError with cddlib: switched to qhull'
    #         #     hull = ConvexHull(np.hstack(trans_mink_sum).T)
    #         #     A = np.array(hull.equations)[:, :-1]
    #         #     b = - np.array(hull.equations)[:, -1:]
    #         #     print A.shape
    #         #     M_ineq = np.hstack((b, -A))
    #         #     M_ineq = cdd.Matrix([list(m) for m in M_ineq])
    #         #     M_ineq.rep_type = cdd.RepType.INEQUALITY

    #         # M_ineq.canonicalize()
    #         # A = - np.array([list(M_ineq[j])[1:] for j in range(M_ineq.row_size)])
    #         # b = np.array([list(M_ineq[j])[:1] for j in range(M_ineq.row_size)])
    #         # print A.shape

    #         # intersect with X
    #         feasible_set = Polytope(A, b)
    #         feasible_set.add_facets(X_sequence[i].lhs_min, X_sequence[i].rhs_min)
    #         feasible_set.assemble_light()

    #     print('Feasible set computed in ' + str(time.time()-tic) + ' s')
    #     return feasible_set

    # def plot_feasible_set(self, switching_sequence, **kwargs):
    #     feasible_set = self.backwards_reachability_analysis(switching_sequence)
    #     feasible_set.plot(**kwargs)
    #     plt.text(feasible_set.center[0], feasible_set.center[1], str(switching_sequence))
    #     return

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
                            print('- Computing feasible set for the switching sequence ' + str(ss) + ' ...')
                            fs = prog.feasible_set
                            print('- Feasible set computed in ' + str(time.time()-tic) + ' seconds.')


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

            ss_values['upper_bound']['convex_hull'] = ConvexHull(np.hstack(xV_list).T, incremental=True)
            A, b = self.upper_bound_from_hull(ss_values['upper_bound']['convex_hull'])
            ss_values['upper_bound']['A'] = A
            ss_values['upper_bound']['b'] = b
            
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
                self.library[ss_star]['upper_bound']['convex_hull'].add_points(np.vstack((x, V_star)).T)
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
            raise ValueError('Switching sequence ' + str(ss) + ' is not feasible at x=' + str(x) + '.')
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
        A = np.array(hull.equations)[:, :-1]
        b = (np.array(hull.equations)[:, -1]).reshape((A.shape[0], 1))
        facets_to_keep = [i for i in range(A.shape[0]) if A[i,-1] < 0.]
        A_new = np.zeros((0, A.shape[1]-1))
        b_new = np.zeros((0, 1))
        for i in facets_to_keep:
            A_new = np.vstack((A_new, - A[i:i+1,:-1]/A[i,-1]))
            b_new = np.vstack((b_new, - b[i:i+1,:]/A[i,-1]))
        return A_new, b_new






class parametric_lp:

    def __init__(self, F_u, F_x, F, C_u, C_x, C):
        """
        LP in the form:
        min  \sum_i | (F_u u + F_x x + F)_i |
        s.t. C_u u <= C_x x + C
        """
        self.F_u = F_u
        self.F_x = F_x
        self.F = F
        self.C_u = C_u
        self.C_x = C_x
        self.C = C
        self.add_slack_variables()
        return

    def add_slack_variables(self):
        """
        Reformulates the LP as:
        min f^T z
        s.t. A z <= B x + c
        """
        n_slack = self.F.shape[0]
        n_u = self.F_u.shape[1]
        self.f = np.vstack((
            np.zeros((n_u,1)),
            np.ones((n_slack,1))
            ))
        self.A = np.vstack((
            np.hstack((self.C_u, np.zeros((self.C_u.shape[0], n_slack)))),
            np.hstack((self.F_u, -np.eye(n_slack))),
            np.hstack((-self.F_u, -np.eye(n_slack)))
            ))
        self.B = np.vstack((self.C_x, -self.F_x, self.F_x))
        self.c = np.vstack((self.C, -self.F, self.F))
        self.n_var = n_u + n_slack
        self.n_cons = self.A.shape[0]
        return

    def solve(self, x0, u_length=None):
        x0 = np.reshape(x0, (x0.shape[0], 1))
        sol = linear_program(self.f, self.A, self.B.dot(x0)+self.c)
        u_star = sol.argmin[0:self.F_u.shape[1]]
        if u_length is not None:
            if not float(u_star.shape[0]/u_length).is_integer():
                raise ValueError('Uncoherent dimension of the input u_length.')
            u_star = [u_star[i*u_length:(i+1)*u_length,:] for i in range(u_star.shape[0]/u_length)]
        return u_star, sol.min


class parametric_qp:

    def __init__(self, F_uu, F_xu, F_xx, F_u, F_x, F, C_u, C_x, C):
        """
        Multiparametric QP in the form:
        min  .5 u' F_{uu} u + x0' F_{xu} u + F_u' u + .5 x0' F_{xx} x0 + F_x' x0 + F
        s.t. C_u u <= C_x x + C
        """
        self.F_uu = F_uu
        self.F_xx = F_xx
        self.F_xu = F_xu
        self.F_u = F_u
        self.F_x = F_x
        self.F = F
        self.C_u = C_u
        self.C_x = C_x
        self.C = C
        self.remove_linear_terms()
        self._feasible_set = None
        return

    def solve(self, x0, u_length=None):
        x0 = np.reshape(x0, (x0.shape[0], 1))
        H = self.F_uu
        f = x0.T.dot(self.F_xu) + self.F_u.T
        A = self.C_u
        b = self.C + self.C_x.dot(x0)
        u_star, cost = quadratic_program(H, f, A, b)
        cost += .5*x0.T.dot(self.F_xx).dot(x0) + self.F_x.T.dot(x0) + self.F
        if u_length is not None:
            if not float(u_star.shape[0]/u_length).is_integer():
                raise ValueError('Uncoherent dimension of the input u_length.')
            u_star = [u_star[i*u_length:(i+1)*u_length,:] for i in range(u_star.shape[0]/u_length)]
        return u_star, cost[0,0]

    def get_active_set(self, x, u, tol=1.e-6):
        u = np.vstack(u)
        return tuple(np.where((self.C_u.dot(u) - self.C - self.C_x.dot(x)) > -tol)[0])

    def remove_linear_terms(self):
        """
        Applies the change of variables z = u + F_uu^-1 (F_xu' x + F_u')
        that puts the cost function in the form
        V = 1/2 z' H z + 1/2 x' F_xx_q x + F_x_q' x + F_q
        and the constraints in the form:
        G u <= W + S x
        """
        self.H_inv = np.linalg.inv(self.F_uu)
        self.H = self.F_uu
        self.F_xx_q = self.F_xx - self.F_xu.dot(self.H_inv).dot(self.F_xu.T)
        self.F_x_q = self.F_x - self.F_xu.dot(self.H_inv).dot(self.F_u)
        self.F_q = self.F - .5*self.F_u.T.dot(self.H_inv).dot(self.F_u)
        self.G = self.C_u
        self.S = self.C_x + self.C_u.dot(self.H_inv).dot(self.F_xu.T)
        self.W = self.C + self.C_u.dot(self.H_inv).dot(self.F_u)
        return

    @property
    def feasible_set(self):
        if self._feasible_set is None:
            augmented_polytope = Polytope(np.hstack((- self.C_x, self.C_u)), self.C)
            augmented_polytope.assemble()
            self._feasible_set = augmented_polytope.orthogonal_projection(range(self.C_x.shape[1]))
        return self._feasible_set

    def get_cost_sensitivity(self, x_list, active_set):

        # clean active set
        G_A = self.G[active_set,:]
        if active_set and np.linalg.matrix_rank(G_A) < G_A.shape[0]:
            lir = linearly_independent_rows(G_A)
            active_set = [active_set[i] for i in lir]

        # multipliers explicit solution
        inactive_set = sorted(list(set(range(self.C.shape[0])) - set(active_set)))
        [G_A, W_A, S_A] = [self.G[active_set,:], self.W[active_set,:], self.S[active_set,:]]
        [G_I, W_I, S_I] = [self.G[inactive_set,:], self.W[inactive_set,:], self.S[inactive_set,:]]
        H_A = np.linalg.inv(G_A.dot(self.H_inv).dot(G_A.T))
        lambda_A_offset = - H_A.dot(W_A)
        lambda_A_linear = - H_A.dot(S_A)

        # primal variables explicit solution
        z_offset = - self.H_inv.dot(G_A.T.dot(lambda_A_offset))
        z_linear = - self.H_inv.dot(G_A.T.dot(lambda_A_linear))

        # optimal value function explicit solution: V_star = .5 x' V_quadratic x + V_linear x + V_offset
        V_quadratic = z_linear.T.dot(self.H).dot(z_linear) + self.F_xx_q
        V_linear = z_offset.T.dot(self.H).dot(z_linear) + self.F_x_q.T
        V_offset = self.F_q + .5*z_offset.T.dot(self.H).dot(z_offset)

        # primal original variables explicit solution
        u_offset = z_offset - self.H_inv.dot(self.F_u)
        u_linear = z_linear - self.H_inv.dot(self.F_xu.T)

        # tangent approximation
        plane_list = []
        for x in x_list:
            A = x.T.dot(V_quadratic) + V_linear
            b = -.5*x.T.dot(V_quadratic).dot(x) + V_offset
            plane_list.append([A, b])

        return plane_list

    def solve_free_x(self):
        H = np.vstack((
            np.hstack((self.F_uu, self.F_xu.T)),
            np.hstack((self.F_xu, self.F_xx))
            ))
        f = np.vstack((self.F_u, self.F_x))
        A = np.hstack((self.C_u, -self.C_x))
        b = self.C
        z_star, cost = quadratic_program(H, f, A, b)
        u_star = z_star[0:self.F_uu.shape[0],:]
        x_star = z_star[self.F_uu.shape[0]:,:]
        return u_star, x_star, cost




### AUXILIARY FUNCTIONS

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def clean_matrix(A, threshold=1.e-6):
    A_abs = np.abs(A)
    elements_to_remove = np.where(np.abs(A) < threshold)
    for i in range(elements_to_remove[0].shape[0]):
        A[elements_to_remove[0][i], elements_to_remove[1][i]] = 0.
    return A

def linearly_independent_rows(A, tol=1.e-6):
    R = linalg.qr(A.T)[1]
    R_diag = np.abs(np.diag(R))
    return list(np.where(R_diag > tol)[0])

def OCP_condenser(sys, objective_norm, Q, R, P, X_N, switching_sequence):
    tic = time.time()
    N = len(switching_sequence)
    Q_bar = linalg.block_diag(*[Q for i in range(N)] + [P])
    R_bar = linalg.block_diag(*[R for i in range(N)])
    G, W, E = constraint_condenser(sys, X_N, switching_sequence)
    if objective_norm == 'one':
        F_u, F_x, F = linear_objective_condenser(sys, Q_bar, R_bar, switching_sequence)
        parametric_program = parametric_lp(F_u, F_x, F, G, E, W)
    elif objective_norm == 'two':
        F_uu, F_xu, F_xx, F_u, F_x, F = quadratic_objective_condenser(sys, Q_bar, R_bar, switching_sequence)
        parametric_program = parametric_qp(F_uu, F_xu, F_xx, F_u, F_x, F, G, E, W)
    print 'total condensing time is', str(time.time()-tic),'s.'
    return parametric_program

# def constraint_condenser(sys, X_N, switching_sequence):
#     N = len(switching_sequence)
#     G_u, W_u, E_u = input_constraint_condenser(sys, switching_sequence)
#     G_x, W_x, E_x = state_constraint_condenser(sys, X_N, switching_sequence)
#     G = np.vstack((G_u, G_x))
#     W = np.vstack((W_u, W_x))
#     E = np.vstack((E_u, E_x))
#     p = Polytope(np.hstack((G, -E)), W)
#     p.assemble()
#     if not p.empty:
#         G = p.lhs_min[:,:sys.n_u*N]
#         E = - p.lhs_min[:,sys.n_u*N:]
#         W = p.rhs_min
#     else:
#         G = None
#         W = None
#         E = None
#     return G, W, E

# def input_constraint_condenser(sys, switching_sequence):
#     N = len(switching_sequence)
#     U_sequence = [sys.input_domains[switching_sequence[i]] for i in range(N)]
#     G_u = linalg.block_diag(*[U.lhs_min for U in U_sequence])
#     W_u = np.vstack([U.rhs_min for U in U_sequence])
#     E_u = np.zeros((W_u.shape[0], sys.n_x))
#     return G_u, W_u, E_u

# def state_constraint_condenser(sys, X_N, switching_sequence):
#     N = len(switching_sequence)
#     X_sequence = [sys.state_domains[switching_sequence[i]] for i in range(N)]
#     lhs_x = linalg.block_diag(*[X.lhs_min for X in X_sequence] + [X_N.lhs_min])
#     rhs_x = np.vstack([X.rhs_min for X in X_sequence] + [X_N.rhs_min])
#     A_bar, B_bar, c_bar = sys.condense(switching_sequence)
#     G_x = lhs_x.dot(B_bar)
#     W_x = rhs_x - lhs_x.dot(c_bar)
#     E_x = - lhs_x.dot(A_bar)
#     return G_x, W_x, E_x

def constraint_condenser(sys, X_N, switching_sequence):
    N = len(switching_sequence)
    D_sequence = [sys.domains[switching_sequence[i]] for i in range(N)]
    lhs_x = linalg.block_diag(*[D.lhs_min[:,:sys.n_x] for D in D_sequence] + [X_N.lhs_min])
    lhs_u = linalg.block_diag(*[D.lhs_min[:,sys.n_x:] for D in D_sequence])
    lhs_u = np.vstack((lhs_u, np.zeros((X_N.lhs_min.shape[0], lhs_u.shape[1]))))
    rhs = np.vstack([D.rhs_min for D in D_sequence] + [X_N.rhs_min])
    A_bar, B_bar, c_bar = sys.condense(switching_sequence)
    G = (lhs_x.dot(B_bar) + lhs_u)
    W = rhs - lhs_x.dot(c_bar)
    E = - lhs_x.dot(A_bar)
    ### do I need this? it might be super slow...
    tic = time.time()
    p = Polytope(np.hstack((G, -E)), W)
    p.assemble()
    if not p.empty:
        G = p.lhs_min[:,:sys.n_u*N]
        E = - p.lhs_min[:,sys.n_u*N:]
        W = p.rhs_min
    else:
        G = None
        W = None
        E = None
    print 'Redundant inequalities removed in', str(time.time()-tic),'s,',
    return G, W, E

def linear_objective_condenser(sys, Q_bar, R_bar, switching_sequence):
    """
    \sum_i | (F_u u + F_x x + F)_i |
    """
    A_bar, B_bar, c_bar = sys.condense(switching_sequence)
    F_u = np.vstack((Q_bar.dot(B_bar), R_bar))
    F_x = np.vstack((Q_bar.dot(A_bar), np.zeros((R_bar.shape[0], A_bar.shape[1]))))
    F = np.vstack((Q_bar.dot(c_bar), np.zeros((R_bar.shape[0], 1))))
    return F_u, F_x, F

def quadratic_objective_condenser(sys, Q_bar, R_bar, switching_sequence):
    """
    .5 u' F_{uu} u + x0' F_{xu} u + F_u' u + .5 x0' F_{xx} x0 + F_x' x0 + F
    """
    A_bar, B_bar, c_bar = sys.condense(switching_sequence)
    F_uu = 2*(R_bar + B_bar.T.dot(Q_bar).dot(B_bar))
    F_xu = 2*A_bar.T.dot(Q_bar).dot(B_bar)
    F_xx = 2.*A_bar.T.dot(Q_bar).dot(A_bar)
    F_u = 2.*B_bar.T.dot(Q_bar).dot(c_bar)
    F_x = 2.*A_bar.T.dot(Q_bar).dot(c_bar)
    F = c_bar.T.dot(Q_bar).dot(c_bar)
    return F_uu, F_xu, F_xx, F_u, F_x, F
