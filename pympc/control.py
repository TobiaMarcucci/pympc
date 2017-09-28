import time, sys, os
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import gurobipy as grb
from contextlib import contextmanager
from copy import copy

from optimization.pnnls import linear_program
from optimization.parametric_programs import ParametricLP, ParametricQP
from optimization.mpqpsolver import MPQPSolver, CriticalRegion
from dynamical_systems import DTAffineSystem, DTPWASystem
from algebra import clean_matrix, nullspace_basis, rangespace_basis
from geometry.polytope import Polytope

class MPCController:
    """
    Model Predictive Controller.

    VARIABLES:
        sys: DTLinearSystem (see dynamical_systems.py)
        N: horizon of the controller
        objective_norm: 'one' or 'two'
        Q, R, P: weight matrices for the controller (state stage cost, input stage cost, and state terminal cost, respectively)
        X, U: state and input domains (see the Polytope class in geometry.polytope)
        X_N: terminal constraint (Polytope)
    """

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
        """
        Depending on the norm of the controller, creates a parametric LP or QP in the initial state of the system (see ParametricLP or ParametricQP classes).
        """
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
        """
        This is needed since OCP_condenser() is the same for PWA systems anb linear systems. OCP for PWA systems have constraints also on the initial state of the system (x(0) has to belong to a domain of the state partition of the PWA system). For linear system this constraint has to be removed.
        """
        C_u_rows_norm = list(np.linalg.norm(self.condensed_program.C_u, axis=1))
        intial_state_contraints = [i for i, row_norm in enumerate(C_u_rows_norm) if row_norm < tol]
        if len(intial_state_contraints) > self.X.lhs_min.shape[0]:
            raise ValueError('Wrong number of zero rows in the constrinats')
        self.condensed_program.C_u = np.delete(self.condensed_program.C_u,intial_state_contraints, 0)
        self.condensed_program.C_x = np.delete(self.condensed_program.C_x,intial_state_contraints, 0)
        self.condensed_program.C = np.delete(self.condensed_program.C,intial_state_contraints, 0)
        return

    def feedforward(self, x0):
        """
        Given the state of the system, returns the optimal sequence of N inputs and the related cost.
        """
        u_feedforward, cost = self.condensed_program.solve(x0)
        u_feedforward = [u_feedforward[self.sys.n_u*i:self.sys.n_u*(i+1),:] for i in range(self.N)]
        # if any(np.isnan(u_feedforward).flatten()):
        #     print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
        return u_feedforward, cost

    def feedback(self, x0):
        """
        Returns the a single input vector (the first of feedforward(x0)).
        """
        return self.feedforward(x0)[0][0]

    def get_explicit_solution(self):
        """
        (Only for controllers with norm = 2).
        Returns the partition of the state space in critical regions (explicit MPC solution).
        Temporary fix: since the method remove_intial_state_contraints() modifies the variables of condensed_program, I have to call remove_linear_terms() again...
        """
        self.condensed_program.remove_linear_terms()
        mpqp_solution = MPQPSolver(self.condensed_program)
        self.critical_regions = mpqp_solution.critical_regions
        return

    def feedforward_explicit(self, x0):
        """
        Finds the critical region where the state x0 is, and returns the PWA feedforward.
        """
        if self.critical_regions is None:
            raise ValueError('Explicit solution not available, call .get_explicit_solution()')
        cr_x0 = self.critical_regions.lookup(x0)
        if cr_x0 is not None:
            u_feedforward = cr_x0.u_offset + cr_x0.u_linear.dot(x0)
            u_feedforward = [u_feedforward[self.sys.n_u*i:self.sys.n_u*(i+1),:] for i in range(self.N)]
            cost = .5*x0.T.dot(cr_x0.V_quadratic).dot(x0) + cr_x0.V_linear.dot(x0) + cr_x0.V_offset
            cost = cost[0,0]
        else:
            # print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            u_feedforward = [np.full((self.sys.n_u, 1), np.nan) for i in range(self.N)]
            cost = np.nan
        return u_feedforward, cost

    def feedback_explicit(self, x0):
        """
        Returns the a single input vector (the first of feedforward_explicit(x0)).
        """
        return self.feedforward(x0)[0][0]

    def optimal_value_function(self, x0):
        """
        Returns the optimal value function for the state x0.
        """
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
    """
    Hybrid Model Predictive Controller.

    VARIABLES:
        sys: DTPWASystem (see dynamical_systems.py)
        N: horizon of the controller
        objective_norm: 'one' or 'two'
        Q, R, P: weight matrices for the controller (state stage cost, input stage cost, and state terminal cost, respectively)
        X_N: terminal constraint (Polytope)
    """

    def __init__(self, sys, N, objective_norm, Q, R, P, X_N, gurobi_parameters={}):
        self.sys = sys
        self.N = N
        self.objective_norm = objective_norm
        self.Q = Q
        self.R = R
        self.P = P
        self.X_N = X_N
        self.gurobi_parameters = gurobi_parameters
        self._get_bigM_domains()
        self._get_bigM_dynamics()
        self._MIP_model()
        return

    def _get_bigM_domains(self, tol=1.e-8):
        """
        Computes all the bigMs for the domains of the PWA system.
        When the system is in mode j, n_i bigMs are needed to drop each one the n_i inequalities of the polytopic domain for the mode i.
        With s number of modes, M_domains is a list containing s lists, each list then constains s bigM vector of size n_i.
        M_domains[j][i] is used to drop the inequalities of the domain i when the system is in mode j.
        """
        self._M_domains = []
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
            self._M_domains.append(M_i)
        return

    def _get_bigM_dynamics(self, tol=1.e-8):
        """
        Computes all the smallMs and bigMs for the dynamics of the PWA system.
        When the system is in mode j, n_x smallMs and n_x bigMs are needed to drop the equations of motion for the mode i.
        With s number of modes, m_dynamics and M_dynamics are two lists containing s lists, each list then constains s bigM vector of size n_x.
        m_dynamics[j][i] and M_dynamics[j][i] are used to drop the equations of motion of the mode i when the system is in mode j.
        """
        self._M_dynamics = []
        self._m_dynamics = []
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
            self._M_dynamics.append(M_i)
            self._m_dynamics.append(m_i)
        return

    def _MIP_model(self):
        self._model = grb.Model()
        self._MIP_variables()
        self._MIP_objective()
        self._MIP_constraints()
        self._MIP_parameters()
        self._model.update()
        return

    def _MIP_variables(self):
        self._x = self._add_real_variable([self.N+1, self.sys.n_x], name='x')
        self._u = self._add_real_variable([self.N, self.sys.n_u], name='u')
        self._z = self._add_real_variable([self.N, self.sys.n_sys, self.sys.n_x], name='z')
        self._d = self._model.addVars(self.N, self.sys.n_sys, vtype=grb.GRB.BINARY, name='d')
        self._model.update()
        return

    def _add_real_variable(self, dimensions, name, **kwargs):
        lb_var = [-grb.GRB.INFINITY]
        for d in dimensions:
            lb_var = [lb_var * d]
        var = self._model.addVars(*dimensions, lb=lb_var, name=name, **kwargs)
        return var

    @staticmethod
    def _linear_term(M, x, k, tol=1.e-9):
        expr_list = []
        for i in range(M.shape[0]):
            expr = grb.LinExpr()
            for j in range(M.shape[1]):
                if np.abs(M[i,j]) > tol:
                    expr.add(M[i,j] * x[k,j])
            expr_list.append(expr)
        return np.vstack(expr_list)

    @staticmethod
    def _quadratic_term(M, x, k, tol=1.e-9):
        expr = grb.QuadExpr()
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.abs(M[i,j]) > tol:
                    expr.add(x[k,i] * M[i,j] * x[k,j])
        return expr

    def _MIP_objective(self):
        if self.objective_norm == 'one':
            self._MILP_objective()
        elif self.objective_norm == 'two':
            self._MIQP_objective()
        else:
            raise ValueError('Unknown norm ' + self.objective_norm + '.')
        return

    def _MILP_objective(self):
        self._phi = self._model.addVars(self.N+1, self.sys.n_x, name='phi')
        self._psi = self._model.addVars(self.N, self.sys.n_u, name='psi')
        self._model.update()
        V = grb.LinExpr()
        for k in range(self.N+1):
            for i in range(self.sys.n_x):
                V.add(self._phi[k,i])
        for k in range(self.N):
            for i in range(self.sys.n_u):
                V.add(self._psi[k,i])
        self._model.setObjective(V)
        for k in range(self.N):
            for i in range(self.sys.n_x):
                self._model.addConstr(self._phi[k,i] >= self._linear_term(self.Q[i:i+1,:], self._x, k)[0,0])
                self._model.addConstr(self._phi[k,i] >= - self._linear_term(self.Q[i:i+1,:], self._x, k)[0,0])
            for i in range(self.sys.n_u):
                self._model.addConstr(self._psi[k,i] >= self._linear_term(self.R[i:i+1,:], self._u, k)[0,0])
                self._model.addConstr(self._psi[k,i] >= - self._linear_term(self.R[i:i+1,:], self._u, k)[0,0])
        for i in range(self.sys.n_x):
            self._model.addConstr(self._phi[self.N,i] >= self._linear_term(self.P[i:i+1,:], self._x, self.N)[0,0])
            self._model.addConstr(self._phi[self.N,i] >= - self._linear_term(self.P[i:i+1,:], self._x, self.N)[0,0])
        return

    def _MIQP_objective(self):
        V = grb.QuadExpr()
        for k in range(self.N):
            V.add(self._quadratic_term(self.Q, self._x, k))
            V.add(self._quadratic_term(self.R, self._u, k))
        V.add(self._quadratic_term(self.P, self._x, self.N))
        self._model.setObjective(V)
        return

    def _MIP_constraints(self):
        self._disjunction_modes()
        self._constraint_domains()
        self._dynamic_constraints()
        self._terminal_constraint()
        return

    def _disjunction_modes(self):
        for k in range(self.N):
            self._model.addSOS(grb.GRB.SOS_TYPE1, [self._d[k,i] for i in range(self.sys.n_sys)], [1]*self.sys.n_sys)
        return

    def _constraint_domains(self):
        # clean bigMs from almost zero terms
        M_domains = [[clean_matrix(self._M_domains[i][j]) for j in range(self.sys.n_sys)] for i in range(self.sys.n_sys)]
        for i in range(self.sys.n_sys):
            lhs = self.sys.domains[i].lhs_min
            rhs = clean_matrix(self.sys.domains[i].rhs_min)
            for k in range(self.N):
                expr = self._linear_term(lhs[:,:self.sys.n_x], self._x, k)
                expr = expr + self._linear_term(lhs[:,self.sys.n_x:], self._u, k)
                expr = expr - np.sum([M_domains[i][j]*self._d[k,j] for j in range(self.sys.n_sys) if j != i], axis=0)
                expr = expr - rhs
                self._model.addConstrs((expr[j,0] <= 0. for j in range(lhs.shape[0])))
        return

    def _dynamic_constraints(self):
        # clean bigMs from almost zero terms
        M_dynamics = [[clean_matrix(self._M_dynamics[i][j]) for j in range(self.sys.n_sys)] for i in range(self.sys.n_sys)]
        m_dynamics = [[clean_matrix(self._m_dynamics[i][j]) for j in range(self.sys.n_sys)] for i in range(self.sys.n_sys)]
        for k in range(self.N):
            for j in range(self.sys.n_x):
                expr = grb.LinExpr()
                for i in range(self.sys.n_sys):
                    expr.add(self._z[k,i,j])
                self._model.addConstr(self._x[k+1,j] == expr)
        for k in range(self.N):
            for i in range(self.sys.n_sys):
                expr_M = M_dynamics[i][i]*self._d[k,i]
                expr_m = m_dynamics[i][i]*self._d[k,i]
                for j in range(self.sys.n_x):
                    self._model.addConstr(self._z[k,i,j] <= expr_M[j,0])
                    self._model.addConstr(self._z[k,i,j] >= expr_m[j,0])
        for i in range(self.sys.n_sys):
            for k in range(self.N):
                expr = self._linear_term(self.sys.affine_systems[i].A, self._x, k)
                expr = expr + self._linear_term(self.sys.affine_systems[i].B, self._u, k)
                expr = expr + clean_matrix(self.sys.affine_systems[i].c)
                expr_M = expr - np.sum([M_dynamics[i][j]*self._d[k,j] for j in range(self.sys.n_sys) if j != i], axis=0)
                expr_m = expr - np.sum([m_dynamics[i][j]*self._d[k,j] for j in range(self.sys.n_sys) if j != i], axis=0)
                for j in range(self.sys.n_x):
                    self._model.addConstr(self._z[k,i,j] >= expr_M[j,0])
                    self._model.addConstr(self._z[k,i,j] <= expr_m[j,0])
        return

    def _terminal_constraint(self):
        expr = self._linear_term(self.X_N.lhs_min, self._x, self.N) - clean_matrix(self.X_N.rhs_min)
        self._model.addConstrs((expr[i,0] <= 0. for i in range(len(self.X_N.minimal_facets))))
        return

    def _MIP_parameters(self):
        for p_key, p_value in self.gurobi_parameters.items():
            self._model.setParam(p_key, p_value)
        return

    def feedforward(self, x0, u_ws=None, x_ws=None, ss_ws=None):

        # initial condition
        for i in range(self.sys.n_x):
            if self._model.getConstrByName('intial_condition_' + str(i)) is not None:
                self._model.remove(self._model.getConstrByName('intial_condition_' + str(i)))
            self._model.addConstr(self._x[0,i] == x0[i,0], name='intial_condition_' + str(i))

        # reset the model to avoid "random" warm starts from gurobi
        self._model.reset() 

        # warm start
        if u_ws is not None:
            self._warm_start_input(u_ws)
        if x_ws is not None:
            self._warm_start_state(x_ws)
        if ss_ws is not None:
            self._warm_start_modes(ss_ws)
        if x_ws is not None and ss_ws is not None:
            self._warm_start_sum_of_states(x_ws, ss_ws)
        if self.objective_norm == 'one' and (x_ws is not None or u_ws is not None):
            self._warm_start_slack_variables_MILP(x_ws, u_ws)
        self._model.update()

        # run optimization
        self._model.optimize()

        return self._return_solution()

    def _warm_start_input(self, u_ws):
        for i in range(self.sys.n_u):
            for k in range(self.N):
                self._u[k,i].setAttr('Start', u_ws[k][i,0])
        return

    def _warm_start_state(self, x_ws):
        for i in range(self.sys.n_x):
            for k in range(self.N+1):
                self._x[k,i].setAttr('Start', x_ws[k][i,0])
        return

    def _warm_start_modes(self, ss_ws):
        for j in range(self.sys.n_sys):
            for k in range(self.N):
                if j == ss_ws[k]:
                    self._d[k,j].setAttr('Start', 1.)
                else:
                    self._d[k,j].setAttr('Start', 0.)
        return

    def _warm_start_sum_of_states(self, x_ws, ss_ws):
        for i in range(self.sys.n_x):
            for j in range(self.sys.n_sys):
                for k in range(self.N):
                    if j == ss_ws[k]:
                        self._z[k,j,i].setAttr('Start', x_ws[k+1][i,0])
                    else:
                        self._z[k,j,i].setAttr('Start', 0.)
        return

    def _warm_start_slack_variables_MILP(self, x_ws, u_ws):
        if x_ws is not None:
            for i in range(self.sys.n_x):
                for k in range(self.N):
                    self._phi[k,i].setAttr('Start', self.Q[i,:].dot(x_ws[k]))
                self._phi[self.N,i].setAttr('Start', self.P[i,:].dot(x_ws[self.N]))
        if u_ws is not None:
            for i in range(self.sys.n_u):
                for k in range(self.N):
                    self._psi[k,i].setAttr('Start', self.R[i,:].dot(u_ws[k]))
        return

    def _return_solution(self):
        if self._model.status in [grb.GRB.Status.OPTIMAL, grb.GRB.Status.INTERRUPTED, grb.GRB.Status.TIME_LIMIT, grb.GRB.Status.SUBOPTIMAL]:
            cost = self._model.objVal
            u_feedforward = [np.array([[self._u[k,i].x] for i in range(self.sys.n_u)]) for k in range(self.N)]
            x_trajectory = [np.array([[self._x[k,i].x] for i in range(self.sys.n_x)]) for k in range(self.N+1)]
            d_star = [np.array([[self._d[k,i].x] for i in range(self.sys.n_sys)]) for k in range(self.N)]
            switching_sequence = [np.where(np.isclose(d, 1.))[0][0] for d in d_star]
        else:
            u_feedforward = [np.full((self.sys.n_u,1), np.nan) for k in range(self.N)]
            x_trajectory = [np.full((self.sys.n_x,1), np.nan) for k in range(self.N+1)]
            cost = np.nan
            switching_sequence = [np.nan]*self.N
        return u_feedforward, x_trajectory, tuple(switching_sequence), cost


    def feedback(self, x0):
        """
        Reuturns the first input from feedforward().
        """
        return self.feedforward(x0)[0][0]

    def condense_program(self, switching_sequence):
        """
        Given a mode sequence, returns the condensed LP or QP (see ParametricLP or ParametricQP classes).
        """
        # tic = time.time()
        # print('\nCondensing the OCP for the switching sequence ' + str(switching_sequence) + ' ...')
        if len(switching_sequence) != self.N:
            raise ValueError('Switching sequence not coherent with the controller horizon.')
        prog = OCP_condenser(self.sys, self.objective_norm, self.Q, self.R, self.P, self.X_N, switching_sequence)
        # print('... OCP condensed in ' + str(time.time() -tic ) + ' seconds.\n')
        return prog

def reachability_standard_form(A, B):
    """
    Applies the transformation x = [T_R, T_N] [z_R; z_N] = = T [z_R; z_N] to decompose the linear system
    x (t+1) = A x (t) + B u (t)
    in the reachable and non reachable subsystems
    z_R (t+1) = A_RR z_R (t) + A_RN z_N (t) + B_R u (t)
    z_N (t+1) = A_NN z_N (t)
    where z_R \in R^n_R and z_N \in R^(n-n_R).
    """

    # reachability analysis
    n = A.shape[0]
    m = B.shape[1]
    R = np.hstack([np.linalg.matrix_power(A, i).dot(B) for i in range(n)])
    n_R = np.linalg.matrix_rank(R)
    if n_R == n:
        return {
        'n_R': n,
        'T': np.eye(n), 'T_R': np.eye(n), 'T_N': np.zeros((n, 0)),
        'T_inv': np.eye(n), 'T_inv_R': np.eye(n), 'T_inv_N': np.zeros((0, n)),
        'A': A, 'A_RR': A, 'A_RN': np.zeros((n, 0)), 'A_NR': np.zeros((0, n)), 'A_NN': np.zeros((0, 0)),
        'B': B, 'B_R': B, 'B_N': np.zeros((0, m)),
        }

    # tranformation to decomposed variables
    T_R = rangespace_basis(R)
    T_N = nullspace_basis(R.T)
    T = np.hstack((T_R, T_N))
    T_inv = np.linalg.inv(T)

    # standard form
    A_canonical = T_inv.dot(A).dot(T)
    B_canonical = T_inv.dot(B)
    A_RR = A_canonical[:n_R,:n_R]
    A_RN = A_canonical[:n_R,n_R:]
    A_NR = A_canonical[n_R:,:n_R]
    A_NN = A_canonical[n_R:,n_R:]
    B_R = B_canonical[:n_R,:]
    B_N = B_canonical[n_R:,:]

    return {
    'n_R': n_R,
    'T': T, 'T_R': T_R, 'T_N':T_N,
    'T_inv': T_inv, 'T_inv_R': T_inv[:n_R,:], 'T_inv_N': T_inv[n_R:,:],
    'A': A_canonical, 'A_RR': A_RR, 'A_RN': A_RN, 'A_NR': A_NR, 'A_NN': A_NN,
    'B': B_canonical, 'B_R': B_R, 'B_N': B_N
    }



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

def OCP_condenser(sys, objective_norm, Q, R, P, X_N, switching_sequence):
    tic = time.time()
    N = len(switching_sequence)
    Q_bar = linalg.block_diag(*[Q for i in range(N)] + [P])
    R_bar = linalg.block_diag(*[R for i in range(N)])
    G, W, E = constraint_condenser(sys, X_N, switching_sequence)
    if objective_norm == 'one':
        F_u, F_x, F = linear_objective_condenser(sys, Q_bar, R_bar, switching_sequence)
        parametric_program = ParametricLP(F_u, F_x, F, G, E, W)
    elif objective_norm == 'two':
        F_uu, F_xu, F_xx, F_u, F_x, F = quadratic_objective_condenser(sys, Q_bar, R_bar, switching_sequence)
        parametric_program = ParametricQP(F_uu, F_xu, F_xx, F_u, F_x, F, G, E, W)
    # print 'total condensing time is', str(time.time()-tic),'s.\n'
    return parametric_program

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
    # # the following might be super slow (and is not necessary)
    # n_ineq_before = G.shape[0]
    # tic = time.time()
    # p = Polytope(np.hstack((G, -E)), W)
    # p.assemble()
    # if not p.empty:
    #     G = p.lhs_min[:,:sys.n_u*N]
    #     E = - p.lhs_min[:,sys.n_u*N:]
    #     W = p.rhs_min
    #     n_ineq_after = G.shape[0]
    # else:
    #     G = None
    #     W = None
    #     E = None
    # print '\n' + str(n_ineq_before - n_ineq_after) + 'on' + str(n_ineq_before) + 'redundant inequalities removed in', str(time.time()-tic),'s,',
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

def remove_initial_state_constraints(prog, tol=1e-10):
    C_u_rows_norm = list(np.linalg.norm(prog.C_u, axis=1))
    intial_state_contraints = [i for i, row_norm in enumerate(C_u_rows_norm) if row_norm < tol]
    prog.C_u = np.delete(prog.C_u,intial_state_contraints, 0)
    prog.C_x = np.delete(prog.C_x,intial_state_contraints, 0)
    prog.C = np.delete(prog.C,intial_state_contraints, 0)
    prog.remove_linear_terms()
    return prog

def explict_solution_from_hybrid_condensing(prog, tol=1e-10):
    porg = remove_initial_state_constraints(prog)
    mpqp_solution = MPQPSolver(prog)
    critical_regions = mpqp_solution.critical_regions
    return critical_regions