from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver
import numpy as np
import scipy.linalg as linalg
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import itertools
from pyhull.halfspace import Halfspace
from pyhull.halfspace import HalfspaceIntersection
import time


def plot_input_sequence(u_sequence, t_s, N, u_max=None, u_min=None):
    """
    plots the input sequence "u_sequence" and its bounds "u_max" and "u_min" as functions of time
    INPUTS:
    t_s -> sampling time
    N -> number of steps
    u_sequence -> input sequence, list of dimension (number of steps = N)
    u_max -> upper bound on the inputs of dimension (number of inputs)
    u_min -> lower bound on the inputs of dimension (number of inputs)
    OUTPUTS:
    none
    """
    n_u = u_sequence[0].shape[0]
    t = np.linspace(0,N*t_s,N+1)
    for i in range(0, n_u):
        plt.subplot(n_u, 1, i+1)
        u_i_sequence = [u_sequence[j][i] for j in range(0,N)]
        input_plot, = plt.step(t, [u_i_sequence[0]] + u_i_sequence, 'b')
        if u_max is not None:
            bound_plot, = plt.step(t, u_max[i,0]*np.ones(t.shape), 'r')
        if u_min is not None:
            bound_plot, = plt.step(t, u_min[i,0]*np.ones(t.shape), 'r')
        plt.ylabel(r'$u_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            if u_max is not None or u_min is not None:
                plt.legend([input_plot, bound_plot], ['Optimal control', 'Control bounds'], loc=1)
            else:
                plt.legend([input_plot], ['Optimal control'], loc=1)
    plt.xlabel(r'$t$')

def plot_state_trajectory(x_trajectory, t_s, N, x_max=None, x_min=None):
    """
    plots the state trajectory "x_trajectory" and its bounds "x_max" and "x_min" as functions of time
    INPUTS:
    x_trajectory -> state trajectory, list of dimension (number of steps + 1= N + 1)
    t_s -> sampling time
    N -> number of steps
    x_max -> upper bound on the states of dimension (number of states)
    x_min -> lower bound on the states of dimension (number of states)
    OUTPUTS:
    none
    """
    n_x = x_trajectory[0].shape[0]
    t = np.linspace(0,N*t_s,N+1)
    for i in range(0, n_x):
        plt.subplot(n_x, 1, i+1)
        x_i_trajectory = [x_trajectory[j][i] for j in range(0,N+1)]
        state_plot, = plt.plot(t, x_i_trajectory, 'b')
        if x_max is not None:
            bound_plot, = plt.step(t, x_max[i,0]*np.ones(t.shape),'r')
        if x_min is not None:
            bound_plot, = plt.step(t, x_min[i,0]*np.ones(t.shape),'r')
        plt.ylabel(r'$x_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            if x_max is not None or x_min is not None:
                plt.legend([state_plot, bound_plot], ['Optimal trajectory', 'State bounds'], loc=1)
            else:
                plt.legend([state_plot], ['Optimal trajectory'], loc=1)
    plt.xlabel(r'$t$')

def linear_program(f, A, b, x_bound=None, toll= 1e-9):
    """
    solves the linear program min f^T * x s.t. { A * x <= b , ||x||_inf <= x_bound }
    INPUTS:
    f -> gradient of the cost function
    A -> left hand side of the constraints
    b -> right hand side of the constraints
    x_bound -> bound on the infinity norm of the solution
    OUTPUTS:
    x_min -> argument which minimizes
    cost_min -> minimum of the cost function
    """
    # program dimensions
    n = f.shape[0]
    m = A.shape[0]
    # build program
    prog = mp.MathematicalProgram()
    x = prog.NewContinuousVariables(n, "x")
    for i in range(0, m):
        prog.AddLinearConstraint((A[i,:] + 1e-15).dot(x) <= b[i])
    prog.AddLinearCost((f + 1e-15).T.dot(x))
    # set bounds to the solution
    if x_bound is not None:
        for i in range(0, n):
                prog.AddLinearConstraint(x[i] <= x_bound)
                prog.AddLinearConstraint(x[i] >= -x_bound)
    # solve
    solver = GurobiSolver()
    result = solver.Solve(prog)
    x_min = prog.GetSolution(x).reshape(n,1)
    # retrieve solution
    cost_min = f.T.dot(x_min)
    return [x_min, cost_min]

def quadratic_program(H, f, A, b, x_bound=None):
    """
    solves the quadratic program min x^t * H * x + f^T * x s.t. A * x <= b
    INPUTS:
    H -> Hessian of the cost function
    f -> linear term of the cost function
    A -> left hand side of the constraints
    b -> right hand side of the constraints
    OUTPUTS:
    x_min -> argument which minimizes
    cost_min -> minimum of the cost function
    """
    # program dimensions
    n = f.shape[0]
    m = A.shape[0]
    # build program
    prog = mp.MathematicalProgram()
    x = prog.NewContinuousVariables(n, "x")
    for i in range(0, m):
        prog.AddLinearConstraint((A[i,:] + 1e-15).dot(x) <= b[i])
    prog.AddQuadraticCost(H, f, x)
    # set bounds to the solution
    if x_bound is not None:
        for i in range(0, n):
            prog.AddLinearConstraint(x[i] <= x_bound)
            prog.AddLinearConstraint(x[i] >= -x_bound)
    # solve
    solver = GurobiSolver()
    result = solver.Solve(prog)
    x_min = prog.GetSolution(x).reshape(n,1)
    cost_min = .5*x_min.T.dot(H.dot(x_min)) + f.T.dot(x_min)
    return [x_min, cost_min]

def maximum_output_admissible_set(A, lhs_x, rhs_x):
    if np.max(np.absolute(np.linalg.eig(A)[0])) > 1:
        raise ValueError('Cannot compute MOAS for unstable systems')
    t = 0
    convergence = False
    while convergence == False:
        # cost function jacobians for all i
        J = lhs_x.dot(np.linalg.matrix_power(A,t+1))
        # constraints to each LP
        cons_lhs = np.vstack([lhs_x.dot(np.linalg.matrix_power(A,k)) for k in range(0,t+1)])
        cons_rhs = np.vstack([rhs_x for k in range(0,t+1)])
        # list of all minima
        s = rhs_x.shape[0]
        J_sol = [(-linear_program(-J[i,:].T, cons_lhs, cons_rhs)[1] - rhs_x[i]) for i in range(0,s)]
        if np.max(J_sol) < 0:
            convergence = True
        else:
            t += 1
    # remove redundant constraints
    moas = Polytope(cons_lhs, cons_rhs)
    moas.assemble()
    return [moas, t]

def licq_check(G, active_set, max_cond=1e9):
    """
    checks if licq holds
    INPUTS:
    G -> gradient of the constraints
    active_set -> active set
    OUTPUTS:
    licq -> flag, = True if licq holds, = False otherwise
    """
    G_A = G[active_set,:]
    licq = True
    cond = np.linalg.cond(G_A.dot(G_A.T))
    if cond > max_cond:
        licq = False
    return licq

class Polytope:
    """
    polytope lhs * x <= rhs
    """
    
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.assembled = False
        return

    def add_facets(self, lhs, rhs):
        if self.assembled:
            raise ValueError('Polytope already assembled, cannot add facets!')
        self.lhs = np.vstack((self.lhs, lhs))
        self.rhs = np.vstack((self.rhs, rhs))
        return

    def add_bounds(self, x_max, x_min):
        if self.assembled:
            raise ValueError('Polytope already assembled, cannot add bounds!')
        n_variables = x_max.shape[0]
        lhs = np.vstack((np.eye(n_variables), -np.eye(n_variables)))
        rhs = np.vstack((x_max, -x_min))
        self.add_facets(lhs, rhs)
        return

    def assemble(self):
        if self.assembled:
            raise ValueError('Polytope already assembled, cannot assemble again!')
        [self.n_facets, self.n_variables] = self.lhs.shape
        self.normalize()
        self.empty = False
        self.bounded = True
        if self.n_facets < self.n_variables+1:
            self.bounded = False
        elif self.n_variables == 1:
            self.assemble_1D()
        elif self.n_variables > 1: 
            self.assemble_multiD()
        if self.empty:
            print('Empty polytope')
        if not self.bounded:
            raise ValueError('Unbounded polyhedron: only polytopes allowed')
        return

    def normalize(self, toll=1e-6):
        for i in range(0, self.n_facets):
            norm_factor = np.linalg.norm(self.lhs[i,:])
            if norm_factor > toll:
                self.lhs[i,:] = self.lhs[i,:]/norm_factor
                self.rhs[i] = self.rhs[i]/norm_factor
        return

    def assemble_1D(self):
        upper_bounds = []
        lower_bounds = []
        for i in range(0, self.n_facets):
            if self.lhs[i] > 0:
                upper_bounds.append((self.rhs[i,0]/self.lhs[i])[0])
                lower_bounds.append(-np.inf)
            elif self.lhs[i] < 0:
                upper_bounds.append(np.inf)
                lower_bounds.append((self.rhs[i,0]/self.lhs[i])[0])
            else:
                raise ValueError('Invalid constraint!')
        upper_bound = min(upper_bounds)
        lower_bound = max(lower_bounds)
        if lower_bound > upper_bound:
            self.empty = True
            return
        self.vertices = [lower_bound, upper_bound]
        if any(np.isinf(self.vertices).flatten()):
            self.bounded = False
            return
        self.minimal_facets = sorted([upper_bounds.index(upper_bound), lower_bounds.index(lower_bound)])
        self.lhs_min = self.lhs[self.minimal_facets,:]
        self.rhs_min = self.rhs[self.minimal_facets]
        if upper_bounds.index(upper_bound) < lower_bounds.index(lower_bound):
            self.facet_centers = [upper_bound, lower_bound]
        else:
            self.facet_centers = [lower_bound, upper_bound]
        self.coincident_facets()
        return

    def assemble_multiD(self):
        interior_point = self.interior_point()
        if interior_point is None:
            self.empty = True
            return
        if any(np.isnan(interior_point).flatten()):
            self.bounded = False
            return
        halfspaces = []
        for i in range(0, self.n_facets):
            halfspace = Halfspace(self.lhs[i,:].tolist(), (-self.rhs[i,0]).tolist())
            halfspaces.append(halfspace)
        polyhedron_qhull = HalfspaceIntersection(halfspaces, interior_point.flatten().tolist())
        self.vertices = np.vstack(polyhedron_qhull.vertices)
        if any(np.isinf(self.vertices).flatten()):
            self.bounded = False
            return
        self.minimal_facets = []
        for i in range(0, self.n_facets):
            if polyhedron_qhull.facets_by_halfspace[i]:
                self.minimal_facets.append(i)
        self.lhs_min = self.lhs[self.minimal_facets,:]
        self.rhs_min = self.rhs[self.minimal_facets]
        self.facet_centers = []
        for facet in self.minimal_facets:
            facet_vertices_inidices = polyhedron_qhull.facets_by_halfspace[facet]
            facet_vertices = self.vertices[facet_vertices_inidices]
            self.facet_centers.append(np.mean(np.vstack(facet_vertices), axis=0))
        self.coincident_facets()
        return

    def interior_point(self, min_penetration=-1.e-6):
        """
        minimizes y s.t. lhs x - rhs <= y
        """
        cost_gradient_ip = np.zeros(self.n_variables+1)
        cost_gradient_ip[-1] = 1.
        lhs_ip = np.hstack((self.lhs, -np.ones((self.n_facets, 1))))
        [interior_point, penetration] = linear_program(cost_gradient_ip, lhs_ip, self.rhs)
        interior_point = interior_point[0:-1]
        if penetration > 0:
            interior_point = None
        elif penetration > min_penetration:
            raise ValueError('Polytope too small! numeric precision cannot be guaranteed...')
        return interior_point

    def coincident_facets(self, toll=1e-8):
        # coincident facets indices
        coincident_facets = []
        lrhs = np.hstack((self.lhs, self.rhs))
        for i in range(0, self.n_facets):
            coincident_facets.append(
                sorted(np.where(np.all(np.isclose(lrhs, lrhs[i,:], toll, toll), axis=1))[0].tolist())
                )
        self.coincident_facets = coincident_facets
        return

    def plot(self, dim_proj=[0,1], **kwargs):
        """
        plots a 2d projection of the polytope
        INPUTS:
        line_style -> line style
        dim_proj -> dimensions in which to project the polytope
        OUTPUTS:
        polytope_plot -> figure handle
        """
        if self.empty:
            raise ValueError('Empty polytope!')
        if len(dim_proj) != 2:
            raise ValueError('Only 2d polytopes!')
        # extract vertices components
        vertices_proj = np.vstack(self.vertices)[:,dim_proj]
        hull = spatial.ConvexHull(vertices_proj)
        for simplex in hull.simplices:
            polytope_plot, = plt.plot(vertices_proj[simplex, 0], vertices_proj[simplex, 1], **kwargs)
        plt.xlabel(r'$x_' + str(dim_proj[0]+1) + '$')
        plt.ylabel(r'$x_' + str(dim_proj[1]+1) + '$')
        return polytope_plot

    @staticmethod
    def from_bounds(x_max, x_min):
        n = x_max.shape[0]
        lhs = np.vstack((np.eye(n), -np.eye(n)))
        rhs = np.vstack((x_max, -x_min))
        p = Polytope(lhs, rhs)
        return p

class DTLinearSystem:

    def __init__(self, A, B=None):
        self.A = A
        self.n_x = np.shape(A)[0]
        if B is None:
            B = np.array([]).reshape(self.n_x, 0)
        self.B = B
        self.n_u = np.shape(B)[1]
        return

    def evolution_matrices(self, N):
        # free evolution of the system
        free_evolution = np.vstack([np.linalg.matrix_power(self.A,k) for k in range(1, N+1)])
        # forced evolution of the system
        forced_evolution = np.zeros((self.n_x*N,self.n_u*N))
        for i in range(0, N):
            for j in range(0, i+1):
                forced_evolution[self.n_x*i:self.n_x*(i+1),self.n_u*j:self.n_u*(j+1)] = np.linalg.matrix_power(self.A,i-j).dot(self.B)
        return [free_evolution, forced_evolution]

    def simulate(self, x0, N, u_sequence=None):
        u_sequence = np.vstack(u_sequence)
        if u_sequence is None:
            u_sequence = np.array([]).reshape(self.n_u*N, 0)
        [free_evolution, forced_evolution] = self.evolution_matrices(N)
        # state trajectory
        x = free_evolution.dot(x0) + forced_evolution.dot(u_sequence)
        x_trajectory = [x0]
        [x_trajectory.append(x[self.n_x*i:self.n_x*(i+1)]) for i in range(0,N)]
        return x_trajectory

    @staticmethod
    def from_continuous(t_s, A, B=None):
        n_x = np.shape(A)[0]
        if B is None:
            B = np.array([]).reshape(n_x, 0)
        n_u = np.shape(B)[1]
        mat_c = np.zeros((n_x+n_u, n_x+n_u))
        mat_c[0:n_x,:] = np.hstack((A,B))
        mat_d = linalg.expm(mat_c*t_s)
        A_d = mat_d[0:n_x, 0:n_x]
        B_d = mat_d[0:n_x, n_x:n_x+n_u]
        sys = DTLinearSystem(A_d, B_d)
        return sys

 # class DTPWASystem:

 #    def __init__(self, A_list, B_list, S_list, R_list, T_list):
 #        self.n_systems = len(A_list)
 #        self.linear_system_list = linear_system_list
 #        self.domain_list = domain_list
 #        return

 #    def check_intersections(self):
 #        # do stuff
 #        return

 #    def join_domains(self):
 #        # if facet_i = - facet_j -> remove facets
 #        return

class MPCController:

    def __init__(self, sys, N, Q, R, terminal_cost=None, terminal_constraint=None, state_constraints=None, input_constraints=None):
        self.sys = sys
        self.N = N
        self.Q = Q
        self.R = R
        self.terminal_cost = terminal_cost
        self.terminal_constraint = terminal_constraint
        self.state_constraints = state_constraints
        self.input_constraints = input_constraints
        return

    def add_state_constraint(self, lhs, rhs):
        if self.state_constraints is None:
            self.state_constraints = Polytope(lhs, rhs)
        else:
            self.state_constraints.add_facets(lhs, rhs)
        return

    def add_input_constraint(self, lhs, rhs):
        if self.input_constraints is None:
            self.input_constraints = Polytope(lhs, rhs)
        else:
            self.input_constraints.add_facets(lhs, rhs)
        return

    def add_state_bound(self, x_max, x_min):
        if self.state_constraints is None:
            self.state_constraints = Polytope.from_bounds(x_max, x_min)
        else:
            self.state_constraints.add_bounds(x_max, x_min)
        return

    def add_input_bound(self, u_max, u_min):
        if self.input_constraints is None:
            self.input_constraints = Polytope.from_bounds(u_max, u_min)
        else:
            self.input_constraints.add_bounds(u_max, u_min)
        return

    def set_terminal_constraint(self, terminal_constraint):
        self. terminal_constraint = terminal_constraint
        return

    def assemble(self):
        if self.state_constraints is not None:
            self.state_constraints.assemble()
        if self.input_constraints is not None:
            self.input_constraints.assemble()
        self.terminal_cost_matrix()
        if self.sys.__class__ == DTLinearSystem:
            self.constraint_blocks()
            self.cost_blocks()
            self.critical_regions = None
        # if self.sys.__class__ == DTPWASystem:
        #     constraint_bigM()
        return

    def terminal_cost_matrix(self):
        if self.terminal_cost is None:
            self.P = self.Q
        elif self.terminal_cost == 'dare':
            self.P = self.dare()[0]
        else:
            raise ValueError('Unknown terminal cost!')
        return

    def dare(self):
        # DARE solution
        P = linalg.solve_discrete_are(self.sys.A, self.sys.B, self.Q, self.R)
        # optimal gain
        K = - linalg.inv(self.sys.B.T.dot(P).dot(self.sys.B)+self.R).dot(self.sys.B.T).dot(P).dot(self.sys.A)
        return [P, K]

    def constraint_blocks(self):
        # compute each constraint
        [G_u, W_u, E_u] = self.input_constraint_blocks()
        [G_x, W_x, E_x] = self.state_constraint_blocks()
        [G_xN, W_xN, E_xN] = self.terminal_constraint_blocks()
        # gather constraints
        G = np.vstack((G_u, G_x, G_xN))
        W = np.vstack((W_u, W_x, W_xN))
        E = np.vstack((E_u, E_x, E_xN))
        # remove redundant constraints
        constraint_polytope = Polytope(np.hstack((G, -E)), W)
        constraint_polytope.assemble()
        self.G = constraint_polytope.lhs_min[:,:self.sys.n_u*self.N]
        self.E = - constraint_polytope.lhs_min[:,self.sys.n_u*self.N:]
        self.W = constraint_polytope.rhs_min
        return

    def input_constraint_blocks(self):
        if self.input_constraints is None:
            G_u = np.array([]).reshape((0, self.sys.n_u*self.N))
            W_u = np.array([]).reshape((0, 1))
            E_u = np.array([]).reshape((0, self.sys.n_x))
        else:
            G_u = linalg.block_diag(*[self.input_constraints.lhs_min for i in range(0, self.N)])
            W_u = np.vstack([self.input_constraints.rhs_min for i in range(0, self.N)])
            E_u = np.zeros((W_u.shape[0],self.sys.n_x))
        return [G_u, W_u, E_u]

    def state_constraint_blocks(self):
        if self.state_constraints is None:
            G_x = np.array([]).reshape((0, self.sys.n_u*self.N))
            W_x = np.array([]).reshape((0, 1))
            E_x = np.array([]).reshape((0, self.sys.n_x))
        else:
            [free_evolution, forced_evolution] = self.sys.evolution_matrices(self.N)
            lhs_x_diag = linalg.block_diag(*[self.state_constraints.lhs_min for i in range(0, self.N)])
            G_x = lhs_x_diag.dot(forced_evolution)
            W_x = np.vstack([self.state_constraints.rhs_min for i in range(0, self.N)])
            E_x = - lhs_x_diag.dot(free_evolution)
        return [G_x, W_x, E_x]

    def terminal_constraint_blocks(self):
        if self.terminal_constraint is None:
            G_xN = np.array([]).reshape((0, self.sys.n_u*self.N))
            W_xN = np.array([]).reshape((0, 1))
            E_xN = np.array([]).reshape((0, self.sys.n_x))
        else:
            if self.terminal_constraint == 'moas':
                # solve dare
                K = self.dare()[1]
                # closed loop dynamics
                A_cl = self.sys.A + self.sys.B.dot(K)
                # constraints for the maximum output admissible set
                lhs_cl = np.vstack((self.state_constraints.lhs_min, self.input_constraints.lhs_min.dot(K)))
                rhs_cl = np.vstack((self.state_constraints.rhs_min, self.input_constraints.rhs_min))
                # compute maximum output admissible set
                moas = maximum_output_admissible_set(A_cl, lhs_cl, rhs_cl)[0]
                lhs_xN = moas.lhs_min
                rhs_xN = moas.rhs_min
            elif self.terminal_constraint == 'origin':
                lhs_xN = np.vstack((np.eye(self.sys.n_x), - np.eye(self.sys.n_x)))
                rhs_xN = np.zeros((2*self.sys.n_x,1))
            else:
                raise ValueError('Unknown terminal constraint!')
            forced_evolution = self.sys.evolution_matrices(self.N)[1]
            G_xN = lhs_xN.dot(forced_evolution[-self.sys.n_x:,:])
            W_xN = rhs_xN
            E_xN = - lhs_xN.dot(np.linalg.matrix_power(self.sys.A, self.N))
        return [G_xN, W_xN, E_xN]

    def cost_blocks(self):
        # quadratic term in the state sequence
        H_x = linalg.block_diag(*[self.Q for i in range(0, self.N-1)])
        H_x = linalg.block_diag(H_x, self.P)
        # quadratic term in the input sequence
        H_u = linalg.block_diag(*[self.R for i in range(0, self.N)])
        # evolution of the system
        [free_evolution, forced_evolution] = self.sys.evolution_matrices(self.N)
        # quadratic term
        self.H = 2*(H_u+forced_evolution.T.dot(H_x.dot(forced_evolution)))
        # linear term
        F = 2*forced_evolution.T.dot(H_x.T).dot(free_evolution)
        self.F = F.T
        return

    def feedforward(self, x0):
        u_feedforward = quadratic_program(self.H, (x0.T.dot(self.F)).T, self.G, self.W + self.E.dot(x0))[0]
        if any(np.isnan(u_feedforward).flatten()):
            raise ValueError('Unfeasible initial condition x_0 = ' + str(x0.tolist()))

        return u_feedforward

    def feedback(self, x0):
        u_feedback = self.feedforward(x0)[0:self.sys.n_u]
        return u_feedback

    def compute_explicit_solution(self):
        tic = time.clock()
        # change variable for exeplicit MPC (z := u_seq + H^-1 F^T x0)
        H_inv = np.linalg.inv(self.H)
        self.S = self.E + self.G.dot(H_inv.dot(self.F.T))
        # start from the origin
        active_set = []
        cr0 = CriticalRegion(active_set, self.H, self.G, self.W, self.S)
        cr_to_be_explored = [cr0]
        explored_cr = []
        tested_active_sets =[cr0.active_set]
        # explore the state space
        while cr_to_be_explored:
            # choose the first candidate in the list and remove it
            cr = cr_to_be_explored[0]
            cr_to_be_explored = cr_to_be_explored[1:]
            if cr.polytope.empty:
                print('Empty critical region detected')
            else:
                # explore CR
                explored_cr.append(cr)
                # for all the facets of the CR
                for facet_index in range(0, len(cr.polytope.minimal_facets)):
                    # for all the candidate active sets across each facet
                    for active_set in cr.candidate_active_sets[facet_index]:
                        if active_set not in tested_active_sets:
                            tested_active_sets.append(active_set)
                            # check LICQ for the given active set
                            licq_flag = licq_check(self.G, active_set)
                            # if LICQ holds  
                            if licq_flag:
                                cr_to_be_explored.append(CriticalRegion(active_set, self.H, self.G, self.W, self.S))
                            # correct active set if LICQ doesn't hold 
                            else:
                                print('LICQ does not hold for the active set ' + str(active_set))
                                active_set = active_set_if_not_licq(active_set, facet_index, cr, self.H, self.G, self.W, self.S)
                                if active_set:
                                    print('    corrected active set ' + str(active_set))
                                    cr_to_be_explored.append(CriticalRegion(active_set, self.H, self.G, self.W, self.S))
                                else:
                                    print('    unfeasible critical region detected')
        self.critical_regions = explored_cr
        toc = time.clock()
        print('\nExplicit solution successfully computed in ' + str(toc-tic) + ' s:')
        print('parameter space partitioned in ' + str(len(self.critical_regions)) + ' critical regions.')

    def evaluate_explicit_solution(self, x):
        if self.critical_regions is None:
            raise ValueError('Explicit solution not computed yet! First run .compute_explicit_solution() ...')
        # find the CR to which the test point belongs
        for cr in self.critical_regions:
            if np.max(cr.polytope.lhs_min.dot(x) - cr.polytope.rhs_min) <= 0:
                break
        # derive explicit solution
        z = cr.z_optimal(x)
        u = z - np.linalg.inv(self.H).dot(self.F.T.dot(x))
        lam = cr.lambda_optimal(x)
        cost_to_go = u.T.dot(self.H.dot(u)) + x.T.dot(self.F).dot(u)
        return [u, cost_to_go, lam] 

class CriticalRegion:
    # this is from:
    # Tondel, Johansen, Bemporad - An algorithm for multi-parametric quadratic programming and explicit MPC solutions

    def __init__(self, active_set, H, G, W, S):
        print 'Computing critical region for the active set ' + str(active_set)
        self.active_set = active_set
        self.inactive_set = list(set(range(0, G.shape[0])) - set(active_set))
        self.boundaries(H, G, W, S)
        # if the critical region is empty return 
        if self.polytope.empty:
            return
        self.candidate_active_sets()

    def boundaries(self, H, G, W, S):
        # optimal solution as a function of x
        H_inv = np.linalg.inv(H)
        # active and inactive constraints
        G_A = G[self.active_set,:]
        W_A = W[self.active_set,:]
        S_A = S[self.active_set,:]
        G_I = G[self.inactive_set,:]
        W_I = W[self.inactive_set,:]
        S_I = S[self.inactive_set,:]
        # multipliers explicit solution
        H_A = np.linalg.inv(G_A.dot(H_inv.dot(G_A.T)))
        self.lambda_A_constant = - H_A.dot(W_A)
        self.lambda_A_linear = - H_A.dot(S_A)
        # primal variable explicit solution 
        self.z_constant = - H_inv.dot(G_A.T.dot(self.lambda_A_constant))
        self.z_linear = - H_inv.dot(G_A.T.dot(self.lambda_A_linear))
        # equation (12) (revised, only inactive indices...)
        lhs_type_1 = G_I.dot(self.z_linear) - S_I
        rhs_type_1 = - G_I.dot(self.z_constant) + W_I
        # equation (13)
        lhs_type_2 = - self.lambda_A_linear
        rhs_type_2 = self.lambda_A_constant
        # gather facets of type 1 and 2 to define the polytope
        lhs = np.array([]).reshape((0,S.shape[1]))
        rhs = np.array([]).reshape((0,1))
        # gather the equations such that the ith facet is the one generated by the ith constraint
        for i in range(G.shape[0]):
            if i in self.active_set:
                lhs = np.vstack((lhs, lhs_type_2[self.active_set.index(i),:]))
                rhs = np.vstack((rhs, rhs_type_2[self.active_set.index(i),0]))
            elif i in self.inactive_set:
                lhs = np.vstack((lhs, lhs_type_1[self.inactive_set.index(i),:]))
                rhs = np.vstack((rhs, rhs_type_1[self.inactive_set.index(i),0]))
        # construct polytope
        self.polytope = Polytope(lhs, rhs)
        self.polytope.assemble()
        return

    def candidate_active_sets(self):
        # without considering weakly active constraints
        candidate_active_sets = candidate_active_sets_generator(self.active_set, self.polytope)
        # detect weakly active constraints
        weakly_active_constraints = detect_weakly_active_constraints(self.active_set, - self.lambda_A_linear, self.lambda_A_constant)
        # correct if any weakly active constraint has been detected
        if weakly_active_constraints:
            # add all the new candidate sets to the list
            candidate_active_sets = expand_candidate_active_sets(weakly_active_constraints, candidate_active_sets)
        self.candidate_active_sets = candidate_active_sets
        return

    def z_optimal(self, x):
        """
        Return the explicit solution of the mpQP as a function of the parameter
        INPUTS:
        x -> value of the parameter
        OUTPUTS:
        z_optimal -> solution of the QP
        """
        z_optimal = self.z_constant + self.z_linear.dot(x).reshape(self.z_constant.shape)
        return z_optimal

    def lambda_optimal(self, x):
        """
        Return the explicit value of the multipliers of the mpQP as a function of the parameter
        INPUTS:
        x -> value of the parameter
        OUTPUTS:
        lambda_optimal -> optimal multipliers
        """
        lambda_A_optimal = self.lambda_A_constant + self.lambda_A_linear.dot(x)
        lambda_optimal = np.zeros(len(self.active_set + self.inactive_set))
        for i in range(0, len(self.active_set)):
            lambda_optimal[self.active_set[i]] = lambda_A_optimal[i]
        return lambda_optimal

def candidate_active_sets_generator(active_set, polytope):
    """
    returns a condidate active set for each facet of a critical region
    Theorem 2 and Corollary 1 are here applied
    INPUTS:
    active_set  -> active set of the parent CR
    polytope -> polytope describing the parent CR
    OUTPUTS:
    candidate_active_sets -> list of candidate active sets (ordered as the facets of the parent polytope, i.e. lhs_min)
    """
    candidate_active_sets = []
    # for each facet of the polytope
    for facet in polytope.minimal_facets:
        # start with the active set of the parent CR
        candidate_active_set = active_set[:]
        # check if this facet has coincident facets (this list includes the facet itself)
        coincident_facets = polytope.coincident_facets[facet]
        # for each coincident facet
        for facet in coincident_facets:
            if facet in candidate_active_set:
                candidate_active_set.remove(facet)
            else:
                candidate_active_set.append(facet)
            candidate_active_set.sort()
            candidate_active_sets.append([candidate_active_set])
    return candidate_active_sets

def detect_weakly_active_constraints(active_set, lhs_type_2, rhs_type_2, toll=1e-8):
    """
    returns the list of constraints that are weakly active in the whole critical region
    enumerated in the as in the equation G z <= W + S x ("original enumeration")
    (by convention weakly active constraints are included among the active set,
    so that only constraints of type 2 are anlyzed)
    INPUTS:
    active_set          -> active set of the parent critical region
    [lhs_type_2, rhs_type_2] -> left- and right-hand side of the constraints of type 2 of the parent CR
    toll             -> tollerance in the detection
    OUTPUTS:
    weakly_active_constraints -> list of weakly active constraints
    """
    weakly_active_constraints = []
    # weakly active constraints are included in the active set
    for i in range(0, len(active_set)):
        # to be weakly active in the whole region they can only be in the form 0^T x <= 0
        if np.linalg.norm(lhs_type_2[i,:]) + np.absolute(rhs_type_2[i,:]) < toll:
            print('Weakly active constraint detected!')
            weakly_active_constraints.append(active_set[i])
    return weakly_active_constraints

def expand_candidate_active_sets(weakly_active_constraints, candidate_active_sets):
    """
    returns the additional condidate active sets that are caused by weakly active constraints (theorem 5)
    INPUTS:
    weakly_active_constraints    -> indices of the weakly active contraints
    candidate_active_sets -> list of candidate neighboring active sets
    OUTPUTS:
    candidate_active_sets -> complete list of candidate active sets
    """
    for i in range(0,len(candidate_active_sets)):
        # for every possible combination of the weakly active constraints
        for n_weakly_act in range(1,len(weakly_active_constraints)+1):
            for comb_weakly_act in itertools.combinations(weakly_active_constraints, n_weakly_act):
                candidate_active_sets_weak_i = []
                # remove each combination from each candidate active set to create a new candidate active set
                if set(candidate_active_sets[i][0]).issuperset(comb_weakly_act):
                    # new candidate active set
                    candidate_active_sets_weak_i.append([j for j in candidate_active_sets[i][0] if j not in list(comb_weakly_act)])
                # update the list of candidate active sets generated because of wekly active constraints
                candidate_active_sets[i].append(candidate_active_sets_weak_i)
    return candidate_active_sets

def active_set_if_not_licq(candidate_active_set, ind, parent, H, G, W, S, dist=1e-6, lambda_bound=1e6, toll=1e-6):
    """
    returns the active set in case that licq does not hold (theorem 4 and some more...)
    INPUTS:
    parent       -> citical region that has generated this degenerate active set hypothesis
    ind          -> index of this active set hypothesis in the parent's list of neighboring active sets
    [H, G, W, S] -> cost and constraint matrices of the mp-QP
    OUTPUTS:
    active_set_child -> real active set of the child critical region (= False if the region is unfeasible)
    """
    x_center = parent.polytope.facet_centers[ind]
    active_set_change = list(set(parent.active_set).symmetric_difference(set(candidate_active_set)))
    if len(active_set_change) > 1:
        print 'Cannot solve degeneracy with multiple active set changes! The solution of a QP is required...'
        # just sole the QP inside the new critical region to derive the active set
        x = x_center + dist*parent.polytope.lhs_min[ind,:]
        x = x.reshape(x_center.shape[0],1)
        z = quadratic_program(H, np.zeros((H.shape[0],1)), G, W+S.dot(x))[0]
        cons_val = G.dot(z) - W - S.dot(x)
        # new active set for the child
        active_set_child = [i for i in range(0,cons_val.shape[0]) if cons_val[i] > -toll]
        # convert [] to False to avoid confusion with the empty active set...
        if not active_set_child:
            active_set_child = False
    else:
        # compute optimal solution in the center of the shared facet
        z_center = parent.z_optimal(x_center)
        # solve lp from theorem 4
        G_A = G[candidate_active_set,:]
        n_lam = G_A.shape[0]
        cost = np.zeros(n_lam)
        cost[candidate_active_set.index(active_set_change[0])] = -1.
        cons_lhs = np.vstack((G_A.T, -G_A.T, -np.eye(n_lam)))

        cons_rhs = np.vstack((-H.dot(z_center), H.dot(z_center), np.zeros((n_lam,1))))
        lambda_sol = linear_program(cost, cons_lhs, cons_rhs, lambda_bound)[0]
        # if the solution in unbounded the region is not feasible
        if np.max(lambda_sol) > lambda_bound - toll:
            active_set_child = False
        # if the solution in bounded look at the indices of the solution
        else:
            active_set_child = []
            for i in range(0,n_lam):
                if lambda_sol[i,0] > toll:
                    active_set_child += [candidate_active_set[i]]
    return active_set_child
