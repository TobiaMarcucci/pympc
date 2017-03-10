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
#from utils.ndpiecewise import NDPiecewise



def plot_input_sequence(u_sequence, t_s, N, u_max=None, u_min=None):
    """
    Plots the input sequence and its bounds as functions of time.

    INPUTS:
        u_sequence: list with N inputs (2D numpy vectors) of dimension (n_u,1) each
        t_s: sampling time
        N: number of steps
        u_max: upper bound on the input (2D numpy vectors of dimension (n_u,1))
        u_min: lower bound on the input (2D numpy vectors of dimension (n_u,1))
    """

    # dimension of the input
    n_u = u_sequence[0].shape[0]

    # time axis
    t = np.linspace(0,N*t_s,N+1)

    # plot each input element separately
    for i in range(0, n_u):
        plt.subplot(n_u, 1, i+1)

        # plot input sequence
        u_i_sequence = [u_sequence[j][i] for j in range(0,N)]
        input_plot, = plt.step(t, [u_i_sequence[0]] + u_i_sequence, 'b')

        # plot bounds iff provided
        if u_max is not None:
            bound_plot, = plt.step(t, u_max[i,0]*np.ones(t.shape), 'r')
        if u_min is not None:
            bound_plot, = plt.step(t, u_min[i,0]*np.ones(t.shape), 'r')

        # miscellaneous options
        plt.ylabel(r'$u_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            if u_max is not None or u_min is not None:
                plt.legend([input_plot, bound_plot], ['Optimal control', 'Control bounds'], loc=1)
            else:
                plt.legend([input_plot], ['Optimal control'], loc=1)
    plt.xlabel(r'$t$')

    return



def plot_state_trajectory(x_trajectory, t_s, N, x_max=None, x_min=None):
    """
    Plots the state trajectory and its bounds as functions of time.

    INPUTS:
        x_trajectory: list with N+1 states (2D numpy vectors) of dimension (n_x,1) each
        t_s: sampling time
        N: number of steps
        x_max: upper bound on the state (2D numpy vectors of dimension (n_x,1))
        x_min: lower bound on the state (2D numpy vectors of dimension (n_x,1))
    """
 
    # dimension of the state
    n_x = x_trajectory[0].shape[0]

    # time axis
    t = np.linspace(0,N*t_s,N+1)

    # plot each state element separately
    for i in range(0, n_x):
        plt.subplot(n_x, 1, i+1)

        # plot state trajectory
        x_i_trajectory = [x_trajectory[j][i] for j in range(0,N+1)]
        state_plot, = plt.plot(t, x_i_trajectory, 'b')

        # plot bounds iff provided
        if x_max is not None:
            bound_plot, = plt.step(t, x_max[i,0]*np.ones(t.shape),'r')
        if x_min is not None:
            bound_plot, = plt.step(t, x_min[i,0]*np.ones(t.shape),'r')

        # miscellaneous options
        plt.ylabel(r'$x_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            if x_max is not None or x_min is not None:
                plt.legend([state_plot, bound_plot], ['Optimal trajectory', 'State bounds'], loc=1)
            else:
                plt.legend([state_plot], ['Optimal trajectory'], loc=1)
    plt.xlabel(r'$t$')

    return



def linear_program(f, A, b, x_bound=1.e9, toll=1.e-3):
    """
    Solves the linear program
    minimize f^T * x
    s. t.    A * x <= b
             ||x||_inf <= x_bound

    INPUTS:
        f: gradient of the cost function (2D numpy array)
        A: left hand side of the constraints (2D numpy array)
        b: right hand side of the constraints (2D numpy array)
        x_bound: bound on the infinity norm of the solution (used to detect unbounded solutions!)
        toll: tollerance in the detection of unbounded solutions
    
    OUTPUTS:
        x_min: argument which minimizes the cost (its elements are nan if unfeasible and inf if unbounded)
        cost_min: minimum of the cost function (nan if unfeasible and inf if unbounded)
        status: status of the solution (=0 if solved, =1 if unfeasible, =2 if unbounded)
    """

    # program dimensions
    n_variables = f.shape[0]
    n_constraints = A.shape[0]

    # build program
    prog = mp.MathematicalProgram()
    x = prog.NewContinuousVariables(n_variables, "x")
    for i in range(0, n_constraints):
        prog.AddLinearConstraint((A[i,:] + 1e-15).dot(x) <= b[i])
    prog.AddLinearCost((f.flatten() + 1e-15).dot(x))

    # set bounds to the solution
    if x_bound is not None:
        for i in range(0, n_variables):
                prog.AddLinearConstraint(x[i] <= x_bound)
                prog.AddLinearConstraint(x[i] >= -x_bound)

    # solve
    solver = GurobiSolver()
    result = solver.Solve(prog)
    x_min = np.reshape(prog.GetSolution(x), (n_variables,1))
    cost_min = f.T.dot(x_min)
    status = 0

    # unfeasible
    if any(np.isnan(x_min)) or np.isnan(cost_min):
        status = 1
        return [x_min, cost_min, status]

    # unbounded
    x_min[np.where(x_min > x_bound - toll)] = np.inf
    x_min[np.where(x_min < - x_bound + toll)] = -np.inf
    if any(f[np.where(np.isinf(x_min))] != 0.):
        cost_min = -np.inf
        status = 2

    return [x_min, cost_min, status]



def quadratic_program(H, f, A, b, C=None, d=None):
    """
    Solves the convex (i.e., H > 0) quadratic program
    minimize x^T * H * x + f^T * x
    s. t.    A * x <= b
             C * x = d

    INPUTS:
        H: Hessian of the cost function (bidimensional numpy array)
        f: linear term of the cost function (monodimensional numpy array)
        A: left hand side of the inequalities (bidimensional numpy array)
        b: right hand side of the inequalities (monodimensional numpy array)
        C: left hand side of the equalities (bidimensional numpy array)
        d: right hand side of the equalities (monodimensional numpy array)

    OUTPUTS:
        x_min: argument which minimizes the cost (its elements are nan if unfeasible and inf if unbounded)
        cost_min: minimum of the cost function (nan if unfeasible and inf if unbounded)
        status: status of the solution (=0 if solved, =1 if unfeasible)
    """

    # program dimensions
    n_variables = f.shape[0]
    n_constraints = A.shape[0]

    # build program
    prog = mp.MathematicalProgram()
    x = prog.NewContinuousVariables(n_variables, "x")
    for i in range(0, n_constraints):
        prog.AddLinearConstraint((A[i,:] + 1e-15).dot(x) <= b[i])
    if C is not None:
        for i in range(C.shape[0]):
            prog.AddLinearConstraint(C[i, :].dot(x) == d[i])
    prog.AddQuadraticCost(H, f, x)

    # solve
    solver = GurobiSolver()
    result = solver.Solve(prog)
    x_min = np.reshape(prog.GetSolution(x), (n_variables,1))
    cost_min = .5*x_min.T.dot(H.dot(x_min)) + f.T.dot(x_min)
    status = 0

    # unfeasible
    if any(np.isnan(x_min)) or np.isnan(cost_min):
        status = 1

    return [x_min, cost_min, status]



def maximum_output_admissible_set(A, lhs, rhs):
    """
    Returns the maximum output admissible set (see Gilbert, Tan - Linear Systems with State and
    Control Constraints, The Theory and Application of Maximal Output Admissible Sets) for a
    non-actuated linear system with state constraints (the output vector is supposed to be the
    entire state of the system , i.e. y=x and C=I).

    INPUTS:
        A: state transition matrix
        lhs: left-hand side of the constraints lhs * x <= rhs
        rhs: right-hand side of the constraints lhs * x <= rhs

    OUTPUTS:
        moas: maximum output admissible set (instatiated as a polytope)
        t: minimum number of steps in the future that define the moas
    """

    # ensure that the system is stable (otherwise the algorithm doesn't converge)
    eig_max = np.max(np.absolute(np.linalg.eig(A)[0]))
    if eig_max > 1:
        raise ValueError('Cannot compute MOAS for unstable systems')

    # Gilber and Tan algorithm
    [n_constraints, n_variables] = lhs.shape
    t = 0
    convergence = False
    while convergence == False:

        # cost function gradients for all i
        J = lhs.dot(np.linalg.matrix_power(A,t+1))

        # constraints to each LP
        cons_lhs = np.vstack([lhs.dot(np.linalg.matrix_power(A,k)) for k in range(0,t+1)])
        cons_rhs = np.vstack([rhs for k in range(0,t+1)])

        # list of all minima
        J_sol = [] 
        for i in range(0, n_constraints):
            J_sol_i = linear_program(np.reshape(-J[i,:], (n_variables,1)), cons_lhs, cons_rhs)[1]
            J_sol.append(-J_sol_i - rhs[i])

        # convergence check
        if np.max(J_sol) < 0:
            convergence = True
        else:
            t += 1

    # define polytope
    moas = Polytope(cons_lhs, cons_rhs)
    moas.assemble()

    return [moas, t]



def licq_check(G, active_set, max_cond=1e9):
    """
    Checks if LICQ holds for the given active set

    INPUTS:
        G: gradient of the constraints
        active_set: active set
        max_cond: maximum condition number of the squared active constraints
    
    OUTPUTS:
        licq -> flag (True if licq holds, False if licq doesn't hold)
    """
    
    # select active constraints
    G_A = G[active_set,:]

    # check condion number of the squared active constraints
    licq = True
    cond = np.linalg.cond(G_A.dot(G_A.T))
    if cond > max_cond:
        licq = False

    return licq



class Polytope:
    """
    Defines a polytope as {x | lhs * x <= rhs}.

    VARIABLES:
        lhs: left-hand side of redundant description of the polytope {x | lhs * x <= rhs}
        rhs: right-hand side of redundant description of the polytope {x | lhs * x <= rhs}
        assembled: flag that determines when it isn't possible to add constraints
        empty: flag that determines if the polytope is empty
        bounded: flag that determines if the polytope is bounded (if not a ValueError is thrown)
        coincident_facets: list of of lists of coincident facets (one list for each facet)
        vertices: list of vertices of the polytope (each one is a 1D array)
        minimal_facets: list of indices of the non-redundant facets
        lhs_min: left-hand side of non-redundant facets
        rhs_min: right-hand side of non-redundant facets
        facet_centers: list of centers of each non-redundant facet
            (i.e.: lhs_min[i,:].dot(facet_centers[i]) = rhs_min[i])
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

    def normalize(self, toll=1e-8):
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
        if any(np.isnan(interior_point)):
            self.empty = True
            return
        if any(np.isinf(interior_point).flatten()):
            self.bounded = False
            return
        halfspaces = []
        for i in range(0, self.n_facets):
            halfspace = Halfspace(self.lhs[i,:].tolist(), (-self.rhs[i,0]).tolist())
            halfspaces.append(halfspace)
        polyhedron_qhull = HalfspaceIntersection(halfspaces, interior_point.flatten().tolist())
        self.vertices = polyhedron_qhull.vertices
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

    def interior_point(self):
        """
        Finds an interior point solving the linear program
        minimize y
        s.t.     lhs * x - rhs <= y
        Returns nan if the polyhedron is empty
        Might return inf if the polyhedron is unbounded
        """
        cost_gradient_ip = np.zeros((self.n_variables+1, 1))
        cost_gradient_ip[-1] = 1.
        lhs_ip = np.hstack((self.lhs, -np.ones((self.n_facets, 1))))
        [interior_point, penetration] = linear_program(cost_gradient_ip, lhs_ip, self.rhs)[0:2]
        interior_point = interior_point[0:-1]
        if penetration > 0:
            interior_point[:] = np.nan
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
        Plots a 2d projection of the polytope.

        INPUTS:
            dim_proj: dimensions in which to project the polytope

        OUTPUTS:
            polytope_plot: figure handle
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

    def __init__(self, A, B):
        self.A = A
        self.B = B
        [self.n_x, self.n_u] = np.shape(B)
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
        if u_sequence is None:
            u_sequence = np.zeros((self.n_u*N, 1))
        else:
            u_sequence = np.vstack(u_sequence)
        [free_evolution, forced_evolution] = self.evolution_matrices(N)
        # state trajectory
        if x0.ndim == 1:
            x0 = np.reshape(x0, (x0.shape[0],1))
        x = free_evolution.dot(x0) + forced_evolution.dot(u_sequence)
        x_trajectory = [x0]
        [x_trajectory.append(x[self.n_x*i:self.n_x*(i+1)]) for i in range(0,N)]
        return x_trajectory

    @staticmethod
    def from_continuous(t_s, A, B):
        n_x = np.shape(A)[0]
        n_u = np.shape(B)[1]
        mat_c = np.zeros((n_x+n_u, n_x+n_u))
        mat_c[0:n_x,:] = np.hstack((A, B))
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
            E_u = np.zeros((W_u.shape[0], self.sys.n_x))
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
            print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
        return u_feedforward

    def feedback(self, x0):
        u_feedback = self.feedforward(x0)[0:self.sys.n_u]
        return u_feedback

    def compute_explicit_solution(self):

        # change variable for exeplicit MPC (z := u_seq + H^-1 F^T x0)
        tic = time.clock()
        H_inv = np.linalg.inv(self.H)
        self.S = self.E + self.G.dot(H_inv.dot(self.F.T))

        # initialize the search with the origin (to which the empty AS is associated)
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

            # if the CR is not empty, find all the potential neighbors
            if cr.polytope.empty:
                print('Empty critical region detected')
            else:
                [cr_to_be_explored, tested_active_sets] = self.spread_critical_region(cr, cr_to_be_explored, tested_active_sets)
                explored_cr.append(cr)

        # collect all the critical regions and report the result
        self.critical_regions = NDPiecewise(explored_cr)
        toc = time.clock()
        print('\nExplicit solution successfully computed in ' + str(toc-tic) + ' s:')
        print('parameter space partitioned in ' + str(len(self.critical_regions)) + ' critical regions.')

    def spread_critical_region(self, cr, cr_to_be_explored, tested_active_sets):

        # for all the facets of the CR and all candidate active sets across each facet
        for facet_index in range(0, len(cr.polytope.minimal_facets)):
            for active_set in cr.candidate_active_sets[facet_index]:

                # add active set to the list of tested be sure to not explore an active set twice
                if active_set not in tested_active_sets:
                    tested_active_sets.append(active_set)

                    # check LICQ for the given active set
                    licq_flag = licq_check(self.G, active_set)

                    # if LICQ holds, determine the critical region
                    if licq_flag:
                        cr_to_be_explored.append(CriticalRegion(active_set, self.H, self.G, self.W, self.S))
                    
                    # if LICQ doesn't hold, correct the active set and determine the critical region
                    else:
                        print('LICQ does not hold for the active set ' + str(active_set))
                        active_set = self.active_set_if_not_licq(active_set, facet_index, cr)
                        if active_set:
                            print('    corrected active set ' + str(active_set))
                            cr_to_be_explored.append(CriticalRegion(active_set, self.H, self.G, self.W, self.S))
                        else:
                            print('    unfeasible critical region detected')
        return [cr_to_be_explored, tested_active_sets]

    def active_set_if_not_licq(self, candidate_active_set, facet_index, cr, dist=1e-6, lambda_bound=1e6, toll=1e-6):
        """
        Returns the active set of a critical region in case that licq does not hold (Theorem 4 revisited)

        INPUTS:
            candidate_active_set: candidate active set for which LICQ doesn't hold
            facet_index: index of this active set hypothesis in the parent's list of neighboring active sets
            cr: critical region from which the new active set is generated

        OUTPUTS:
            active_set: real active set of the child critical region ([] if the region is unfeasible)
        """

        # differences between the active set of the parent and the candidate active set
        active_set_change = list(set(cr.active_set).symmetric_difference(set(candidate_active_set)))

        # if there is more than one change, nothing can be done...
        if len(active_set_change) > 1:
            print 'Cannot solve degeneracy with multiple active set changes! The solution of a QP is required...'
            active_set = self.solve_qp_beyond_facet(facet_index, cr)

        # if there is one change solve the lp from Theorem 4
        else:
            active_set = self.solve_lp_on_facet(candidate_active_set, facet_index, cr)
            
        return active_set

    def solve_qp_beyond_facet(self, facet_index, cr, dist=1e-6, toll=1e-6):
        """
        Solves a QP a step of length "dist" beyond the facet wich index is "facet_index"
        to determine the active set in that region.

        INPUTS:
            facet_index: index of this active set hypothesis in the parent's list of neighboring active sets
            cr: critical region from which the new active set is generated

        OUTPUTS:
            active_set: real active set of the child critical region ([] if the region is unfeasible)
        """

        # center of the facet in the parameter space
        x_center = cr.polytope.facet_centers[facet_index]

        # solve the QP inside the new critical region to derive the active set
        x_beyond = x_center + dist*cr.polytope.lhs_min[facet_index,:]
        x_beyond = x_beyond.reshape(x_center.shape[0],1)
        z = quadratic_program(self.H, np.zeros((self.H.shape[0],1)), self.G, self.W + self.S.dot(x_beyond))[0]
        
        # new active set for the child
        constraints_residuals = self.G.dot(z) - self.W - self.S.dot(x_beyond)
        active_set = [i for i in range(0,self.G.shape[0]) if constraints_residuals[i] > -toll]

        return active_set

    def solve_lp_on_facet(self, candidate_active_set, facet_index, cr, lambda_bound=1e6, toll=1e-6):
        """
        Solves a LP on the center of the facet wich index is "facet_index" to determine
        the active set in that region (Theorem 4)

        INPUTS:
            candidate_active_set: candidate active set for which LICQ doesn't hold
            facet_index: index of this active set hypothesis in the parent's list of neighboring active sets
            cr: critical region from which the new active set is generated

        OUTPUTS:
            active_set: real active set of the child critical region ([] if the region is unfeasible)
        """

        # differences between the active set of the parent and the candidate active set
        active_set_change = list(set(cr.active_set).symmetric_difference(set(candidate_active_set)))

        # compute optimal solution in the center of the shared facet
        x_center = cr.polytope.facet_centers[facet_index]
        z_center = cr.z_optimal(x_center)

        # solve lp from Theorem 4
        G_A = self.G[candidate_active_set,:]
        n_lam = G_A.shape[0]
        cost = np.zeros((n_lam,1))
        cost[candidate_active_set.index(active_set_change[0])] = -1.
        cons_lhs = np.vstack((G_A.T, -G_A.T, -np.eye(n_lam)))
        cons_rhs = np.vstack((-self.H.dot(z_center), self.H.dot(z_center), np.zeros((n_lam,1))))
        lambda_sol = linear_program(cost, cons_lhs, cons_rhs, lambda_bound)[0]

        # if the solution in unbounded the region is unfeasible
        if np.max(lambda_sol) > lambda_bound - toll:
            active_set = []

        # if the solution in bounded look at the indices of the solution to derive the active set
        else:
            active_set = []
            for i in range(0, n_lam):
                if lambda_sol[i] > toll:
                    active_set += [candidate_active_set[i]]

        return active_set

    def feedforward_explicit(self, x0):

        # check that the explicit solution is available
        if self.critical_regions is None:
            raise ValueError('Explicit solution not computed yet! First run .compute_explicit_solution() ...')
        
        # find the CR to which the given state belongs
        cr_where_x0 = self.critical_regions.lookup(x0)

        # derive explicit solution
        if cr_where_x0 is not None:
            z = cr_where_x0.z_optimal(x0)
            u_feedforward = z - np.linalg.inv(self.H).dot(self.F.T.dot(x0))

        # if unfeasible return nan
        else:
            print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            u_feedforward = np.zeros((self.sys.n_u*self.N,1))
            u_feedforward[:] = np.nan

        return u_feedforward

    def feedback_explicit(self, x0):

        # select only the first control of the feedforward
        u_feedback = self.feedforward_explicit(x0)[0:self.sys.n_u]

        return u_feedback



class CriticalRegion:
    """
    Implements the algorithm from Tondel et al. "An algorithm for multi-parametric quadratic programming and explicit MPC solutions"
    """

    def __init__(self, active_set, H, G, W, S):

        # store active set
        print 'Computing critical region for the active set ' + str(active_set)
        [self.n_constraints, self.n_parameters] = S.shape
        self.active_set = active_set
        self.inactive_set = sorted(list(set(range(0, self.n_constraints)) - set(active_set)))

        # find the polytope
        self.polytope(H, G, W, S)
        if self.polytope.empty:
            return

        # find candidate active sets for the neighboiring regions
        minimal_coincident_facets = [self.polytope.coincident_facets[i] for i in self.polytope.minimal_facets]
        self.candidate_active_sets = self.candidate_active_sets(active_set, minimal_coincident_facets)

        # detect weakly active constraints
        self.weakly_active_constraints()

        # expand the candidates if there are weakly active constraints
        if self.weakly_active_constraints:
            self.candidate_active_set = self.expand_candidate_active_sets(self.candidate_active_set, self.weakly_active_constraints)

        return

    def polytope(self, H, G, W, S):
        """
        Stores a polytope that describes the critical region in the parameter space.
        """

        # multipliers explicit solution
        H_inv = np.linalg.inv(H)
        [G_A, W_A, S_A] = [G[self.active_set,:], W[self.active_set,:], S[self.active_set,:]]
        [G_I, W_I, S_I] = [G[self.inactive_set,:], W[self.inactive_set,:], S[self.inactive_set,:]]
        H_A = np.linalg.inv(G_A.dot(H_inv.dot(G_A.T)))
        self.lambda_A_offset = - H_A.dot(W_A)
        self.lambda_A_linear = - H_A.dot(S_A)

        # primal variables explicit solution
        self.z_offset = - H_inv.dot(G_A.T.dot(self.lambda_A_offset))
        self.z_linear = - H_inv.dot(G_A.T.dot(self.lambda_A_linear))

        # equation (12) (modified: only inactive indices considered)
        lhs_type_1 = G_I.dot(self.z_linear) - S_I
        rhs_type_1 = - G_I.dot(self.z_offset) + W_I

        # equation (13)
        lhs_type_2 = - self.lambda_A_linear
        rhs_type_2 = self.lambda_A_offset

        # gather facets of type 1 and 2 to define the polytope (note the order: the ith facet of the cr is generated by the ith constraint)
        lhs = np.zeros((self.n_constraints, self.n_parameters))
        rhs = np.zeros((self.n_constraints, 1))
        lhs[self.inactive_set + self.active_set, :] = np.vstack((lhs_type_1, lhs_type_2))
        rhs[self.inactive_set + self.active_set] = np.vstack((rhs_type_1, rhs_type_2))

        # construct polytope
        self.polytope = Polytope(lhs, rhs)
        self.polytope.assemble()

        return

    def weakly_active_constraints(self, toll=1e-8):
        """
        Stores the list of constraints that are weakly active in the whole critical region
        enumerated in the as in the equation G z <= W + S x ("original enumeration")
        (by convention weakly active constraints are included among the active set,
        so that only constraints of type 2 are anlyzed)
        """

        # equation (13), again...
        lhs_type_2 = - self.lambda_A_linear
        rhs_type_2 = self.lambda_A_offset

        # weakly active constraints are included in the active set
        self.weakly_active_constraints = []
        for i in range(0, len(self.active_set)):

            # to be weakly active in the whole region they can only be in the form 0^T x <= 0
            if np.linalg.norm(lhs_type_2[i,:]) + np.absolute(rhs_type_2[i,:]) < toll:
                print('Weakly active constraint detected!')
                self.weakly_active_constraints.append(self.active_set[i])

        return

    @staticmethod
    def candidate_active_sets(active_set, minimal_coincident_facets):
        """
        Computes one candidate active set for each non-redundant facet of a critical region
        (Theorem 2 and Corollary 1).

        INPUTS:
        active_set: active set of the parent critical region
        minimal_coincident_facets: list of facets coincident to the minimal facets
            (i.e.: [coincident_facets[i] for i in minimal_facets])

        OUTPUTS:
            candidate_active_sets: list of the candidate active sets for each minimal facet
        """
        
        # initialize list of condidate active sets
        candidate_active_sets = []

        # cross each non-redundant facet of the parent CR
        for coincident_facets in minimal_coincident_facets:

            # add or remove each constraint crossed to the active set of the parent CR
            candidate_active_set = set(active_set).symmetric_difference(set(coincident_facets))
            candidate_active_sets.append([sorted(list(candidate_active_set))])

        return candidate_active_sets
    
    @staticmethod
    def expand_candidate_active_sets(candidate_active_sets, weakly_active_constraints):
        """
        Expands the candidate active sets if there are some weakly active contraints (Theorem 5).

        INPUTS:
            candidate_active_sets: list of the candidate active sets for each minimal facet
            weakly_active_constraints: list of weakly active constraints (in the "original enumeration")

        OUTPUTS:
            candidate_active_sets: list of the candidate active sets for each minimal facet


        """

        # determine every possible combination of the weakly active contraints
        wac_combinations = []
        for n in range(1, len(weakly_active_constraints)+1):
            wac_combinations_n = itertools.combinations(weakly_active_constraints, n)
            wac_combinations += [list(c) for c in wac_combinations_n]

        # for each minimal facet of the CR add or remove each combination of wakly active constraints
        for i in range(0, len(candidate_active_sets)):
            active_set = candidate_active_sets[i][0]
            for combination in wac_combinations:
                further_active_set = set(active_set).symmetric_difference(combination)
                candidate_active_sets[i].append(sorted(list(further_active_set)))

        return candidate_active_sets

    def z_optimal(self, x):
        """
        Returns the explicit solution of the mpQP as a function of the parameter.

        INPUTS:
            x: value of the parameter

        OUTPUTS:
            z_optimal: solution of the QP
        """
        z_optimal = self.z_offset + self.z_linear.dot(x).reshape(self.z_offset.shape)
        return z_optimal

    def lambda_optimal(self, x):
        """
        Returns the explicit value of the multipliers of the mpQP as a function of the parameter.

        INPUTS:
            x: value of the parameter

        OUTPUTS:
            lambda_optimal: optimal multipliers
        """
        lambda_A_optimal = self.lambda_A_offset + self.lambda_A_linear.dot(x)
        lambda_optimal = np.zeros(len(self.active_set + self.inactive_set))
        for i in range(0, len(self.active_set)):
            lambda_optimal[self.active_set[i]] = lambda_A_optimal[i]
        return lambda_optimal

    def applies_to(self, x):
        """
        Determines is a given point belongs to the critical region.

        INPUTS:
            x: value of the parameter

        OUTPUTS:
            is_inside: flag (True if x is in the CR, False otherwise)
        """
        is_inside = np.max(self.polytope.lhs_min.dot(x) - self.polytope.rhs_min) <= 0
        return is_inside



class NDPiecewise(object):
    def __init__(self, elements):
        self.elements = elements

    def lookup(self, point):
        for element in self.elements:
            if element.applies_to(point):
                return element
        return None

    def __len__(self):
        return len(self.elements)