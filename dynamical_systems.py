import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from optimization.pnnls import linear_program
from geometry import Polytope

### CLASSES OF DYNAMICAL SYSTEMS

class DTLinearSystem:
    """
    Discrete time linear systems in the form:
    x_{k+1} = A x_k + B u_k.

    VARIABLES:
        A: discrete time state transition matrix
        B: discrete time input to state map
        n_x: number of sates
        n_u: number of inputs
    """

    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.n_x, self.n_u = np.shape(B)
        return

    def condense(self, N):
        c = np.zeros((self.n_x, 1))
        sys = DTAffineSystem(self.A, self.B, c)
        affine_systems = [sys]
        switching_sequence = [0]*N
        return condense_dynamical_system(affine_systems, switching_sequence)[0:2]

    def simulate(self, x0, u_list):
        N = len(u_list)
        A_bar, B_bar = self.condense(N)
        c_bar = np.zeros((self.n_x*(N+1), 1))
        return simulate_affine_dynamics(A_bar, B_bar, c_bar, x0, u_list)

    @staticmethod
    def from_continuous(A, B, t_s):
        c = np.zeros((A.shape[0], 1))
        A_d, B_d, _ = zero_order_hold(A, B, c, t_s)
        return DTLinearSystem(A_d, B_d)



class DTAffineSystem:
    """
    Discrete time affine systems in the form:
    x_{k+1} = A x_k + B u_k + c.

    VARIABLES:
        A: discrete time state transition matrix
        B: discrete time input to state map
        c: discrete time offset term
        n_x: number of sates
        n_u: number of inputs
    """

    def __init__(self, A, B, c):
        self.A = A
        self.B = B
        self.c = c
        self.n_x, self.n_u = np.shape(B)

    def condense(self, N):
        affine_systems = [self]
        switching_sequence = [0]*N
        return condense_dynamical_system(affine_systems, switching_sequence)

    def simulate(self, x0, u_list):
        N = len(u_list)
        A_bar, B_bar, c_bar = self.condense(N)
        return simulate_affine_dynamics(A_bar, B_bar, c_bar, x0, u_list)

    @staticmethod
    def from_continuous(A, B, c, t_s):
        A_d, B_d, c_d = zero_order_hold(A, B, c, t_s)
        return DTAffineSystem(A_d, B_d, c_d)



class DTPWASystem(object):
    """
    Discrete time piecewise affine systems in the form:
    x_{k+1} = A_i x_k + B_i u_k + c_i   if   x_k \in X_i and u_k \in U_i

    VARIABLES:
        affine_systems: list of affine systems
        state_domains: list of state domains for the affine systems
        input_domains: list of input domains for the affine systems
        n_x: number of sates
        n_u: number of inputs
        n_sys: number of affine subsystems
    """

    def __init__(self, affine_systems, state_domains, input_domains):
        self.affine_systems = affine_systems
        self.state_domains = state_domains
        self.input_domains = input_domains
        self.n_x = affine_systems[0].n_x
        self.n_u = affine_systems[0].n_u
        self.n_sys = len(affine_systems)
        return

    def condense(self, switching_sequence):
        return condense_dynamical_system(self.affine_systems, switching_sequence)

    def simulate(self, x0, u_list):
        N = len(u_list)
        x_list = [x0]
        switching_sequence = []
        for k in range(N):
            domain = None
            for i in range(self.n_sys):
                if self.state_domains[i].applies_to(x_list[k]) and self.input_domains[i].applies_to(u_list[k]):
                    domain = i
                    sys = self.affine_systems[i]
                    x_next = sys.A.dot(x_list[k]) + sys.B.dot(u_list[k]) + sys.c
                    x_list.append(x_next)
                    break
            if domain is None:
                raise ValueError('Unfeasible input sequence or initial state!')
            else:
                switching_sequence.append(domain)
        return x_list, switching_sequence



### AUXILIARY FUNCTIONS

def productory(matrix_list):
    prod = matrix_list[0]
    for i in range(1, len(matrix_list)):
        prod = prod.dot(matrix_list[i])
    return prod

def simulate_affine_dynamics(A_bar, B_bar, c_bar, x0, u_list):

    # reshape initial state (from vector to matrix)
    n_x = x0.shape[0]
    if x0.ndim == 1:
        x0 = np.reshape(x0, (n_x, 1))

    # derive state trajectory including initial state
    x_vec = A_bar.dot(x0) + B_bar.dot(np.vstack(u_list)) + c_bar
    N = len(u_list)
    x_list = []
    [x_list.append(x_vec[n_x*i:n_x*(i+1)]) for i in range(N+1)]

    return x_list

def condense_dynamical_system(affine_systems, switching_sequence):

    # system dimensions
    n_x = affine_systems[0].n_x
    n_u = affine_systems[0].n_u
    N = len(switching_sequence)

    # matrix sequence
    A_sequence = [affine_systems[switching_sequence[i]].A for i in range(N)]
    B_sequence = [affine_systems[switching_sequence[i]].B for i in range(N)]
    c_sequence = [affine_systems[switching_sequence[i]].c for i in range(N)]

    # free evolution of the system
    A_bar = np.vstack([productory(A_sequence[i::-1]) for i in range(N)])
    A_bar = np.vstack((np.eye(n_x), A_bar))

    # forced evolution of the system
    B_bar = np.zeros((n_x*N,n_u*N))
    for i in range(N):
        for j in range(i):
            B_bar[n_x*i:n_x*(i+1), n_u*j:n_u*(j+1)] = productory(A_sequence[i:j:-1]).dot(B_sequence[j])
        B_bar[n_x*i:n_x*(i+1), n_u*i:n_u*(i+1)] = B_sequence[i]
    B_bar = np.vstack((np.zeros((n_x, n_u*N)), B_bar))

    # evolution related to the offset term
    c_bar = np.vstack((np.zeros((n_x,1)), c_sequence[0]))
    for i in range(1, N):
        offset_i = sum([productory(A_sequence[i:j:-1]).dot(c_sequence[j]) for j in range(i)]) + c_sequence[i]
        c_bar = np.vstack((c_bar, offset_i))

    return A_bar, B_bar, c_bar

def zero_order_hold(A, B, c, t_s):

    # system dimensions
    n_x = np.shape(A)[0]
    n_u = np.shape(B)[1]

    # zero order hold (see Bicchi - Fondamenti di Automatica 2)
    mat_c = np.zeros((n_x+n_u+1, n_x+n_u+1))
    mat_c[0:n_x,:] = np.hstack((A, B, c))
    mat_d = linalg.expm(mat_c*t_s)

    # discrete time dynamics
    A_d = mat_d[0:n_x, 0:n_x]
    B_d = mat_d[0:n_x, n_x:n_x+n_u]
    c_d = mat_d[0:n_x, n_x+n_u:n_x+n_u+1]

    return A_d, B_d, c_d

def dare(A, B, Q, R):
    # cost to go
    P = linalg.solve_discrete_are(A, B, Q, R)
    # optimal gain
    K = - linalg.inv(B.T.dot(P).dot(B)+R).dot(B.T).dot(P).dot(A)
    return P, K

def moas_closed_loop(A, B, K, X, U):
    # closed loop dynamics
    A_cl = A + B.dot(K)
    # constraints for the maximum output admissible set
    lhs_cl = np.vstack((X.lhs_min, U.lhs_min.dot(K)))
    rhs_cl = np.vstack((X.rhs_min, U.rhs_min))
    X_cl = Polytope(lhs_cl, rhs_cl)
    X_cl.assemble()
    # compute maximum output admissible set
    return moas(A_cl, X_cl)

def moas(A, X):
    """
    Returns the maximum output admissible set (see Gilbert, Tan - Linear Systems with State and
    Control Constraints, The Theory and Application of Maximal Output Admissible Sets) for a
    non-actuated linear system with state constraints (the output vector is supposed to be the
    entire state of the system, i.e. y=x and C=I).

    INPUTS:
        A: state transition matrix
        X: constraint polytope X.lhs * x <= X.rhs

    OUTPUTS:
        moas: maximum output admissible set (instatiated as a polytope)
    """

    # ensure that the system is stable (otherwise the algorithm doesn't converge)
    eig_max = np.max(np.absolute(np.linalg.eig(A)[0]))
    if eig_max > 1:
        raise ValueError('Cannot compute MOAS for unstable systems')

    # Gilber and Tan algorithm
    [n_constraints, n_variables] = X.lhs_min.shape
    t = 0
    convergence = False
    while convergence == False:

        # cost function gradients for all i
        J = X.lhs_min.dot(np.linalg.matrix_power(A,t+1))

        # constraints to each LP
        cons_lhs = np.vstack([X.lhs_min.dot(np.linalg.matrix_power(A,k)) for k in range(0,t+1)])
        cons_rhs = np.vstack([X.rhs_min for k in range(0,t+1)])

        # list of all minima
        J_sol = []
        for i in range(0, n_constraints):
            J_sol_i = linear_program(np.reshape(-J[i,:], (n_variables,1)), cons_lhs, cons_rhs)[1]
            J_sol.append(-J_sol_i - X.rhs_min[i])

        # convergence check
        if np.max(J_sol) < 0:
            convergence = True
        else:
            t += 1

    # define polytope
    moas = Polytope(cons_lhs, cons_rhs)
    moas.assemble()

    return moas


### PLOT FUNCTIONS ###

def plot_input_sequence(u_sequence, t_s, N, u_bounds=None):
    """
    Plots the input sequence and its bounds as functions of time.

    INPUTS:
        u_sequence: list with N inputs (2D numpy vectors) of dimension (n_u,1) each
        t_s: sampling time
        N: number of steps
        u_bounds: list of bound on the input (2D numpy vectors of dimension (n_u,1))
    """

    # dimension of the input
    n_u = u_sequence[0].shape[0]

    # time axis
    t = np.linspace(0,N*t_s,N+1)

    # plot each input element separately
    for i in range(n_u):
        plt.subplot(n_u, 1, i+1)

        # plot input sequence
        u_i_sequence = [u_sequence[j][i] for j in range(0,N)]
        input_plot, = plt.step(t, [u_i_sequence[0]] + u_i_sequence, 'b')

        # plot bounds if provided
        if u_bounds is not None:
            for bound in u_bounds:
                bound_plot, = plt.step(t, bound[i,0]*np.ones(t.shape), 'r')

        # miscellaneous options
        plt.ylabel(r'$u_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            if u_bounds is not None:
                plt.legend([input_plot, bound_plot], ['Optimal control', 'Control bounds'], loc=1)
            else:
                plt.legend([input_plot], ['Optimal control'], loc=1)
    plt.xlabel(r'$t$')

    return

def plot_state_trajectory(x_trajectory, t_s, N, x_bounds=None):
    """
    Plots the state trajectory and its bounds as functions of time.

    INPUTS:
        x_trajectory: list with N+1 states (2D numpy vectors) of dimension (n_x,1) each
        t_s: sampling time
        N: number of steps
        x_bounds: list of bound on the state (2D numpy vectors of dimension (n_x,1))
    """

    # dimension of the state
    n_x = x_trajectory[0].shape[0]

    # time axis
    t = np.linspace(0,N*t_s,N+1)

    # plot each state element separately
    for i in range(n_x):
        plt.subplot(n_x, 1, i+1)

        # plot state trajectory
        x_i_trajectory = [x_trajectory[j][i] for j in range(0,N+1)]
        state_plot, = plt.plot(t, x_i_trajectory, 'b')

        # plot bounds if provided
        if x_bounds is not None:
            for bound in x_bounds:
                bound_plot, = plt.step(t, bound[i,0]*np.ones(t.shape),'r')

        # miscellaneous options
        plt.ylabel(r'$x_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            if x_bounds is not None:
                plt.legend([state_plot, bound_plot], ['Optimal trajectory', 'State bounds'], loc=1)
            else:
                plt.legend([state_plot], ['Optimal trajectory'], loc=1)
    plt.xlabel(r'$t$')

    return


def plot_state_space_trajectory(x_trajectory, state_components=[0,1], **kwargs):
    """
    Plots the state trajectories as functions of time (2d plot).

    INPUTS:
        x_trajectory: state trajectory \in R^((N+1)*n_x)
        N: time steps
        state_components: components of the state vector to be plotted.
    """
    for k in range(len(x_trajectory)-1):
        plt.plot([x_trajectory[k][state_components[0]], x_trajectory[k+1][state_components[0]]], [x_trajectory[k][state_components[1]], x_trajectory[k+1][state_components[1]]], **kwargs)
        # plt.text(x_trajectory[k][0], x_trajectory[k][1], r'$x('+str(k)+')$')
    # ax = plt.axes()
    x_0 = (x_trajectory[0][state_components[0]][0], x_trajectory[0][state_components[1]][0])
    plt.scatter(x_0[0], x_0[1], color='b')
    plt.text(x_0[0], x_0[1], r'$x(0)$')
    plt.xlabel(r'$x_{' + str(state_components[0]+1) + '}$')
    plt.ylabel(r'$x_{' + str(state_components[1]+1) + '}$')
    return

