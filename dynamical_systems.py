import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from optimization.pnnls import linear_program
from geometry import Polytope

class DTLinearSystem:
    """
    Discrete time linear systems in the form x_{k+1} = A*x_k + B*u_k.

    VARIABLES:
        A: discrete time state transition matrix
        B: discrete time input to state map
        n_x: number of sates
        n_u: number of inputs
    """

    def __init__(self, A, B):
        self.A = A
        self.B = B
        [self.n_x, self.n_u] = np.shape(B)
        return

    def evolution_matrices(self, N):
        """
        Returns the free and forced evolution matrices for the linear system
        (i.e. [x_1^T, ...,  x_N^T]^T = free_evolution*x_0 + forced_evolution*[u_0^T, ...,  u_{N-1}^T]^T)

        INPUTS:
            N: number of steps

        OUTPUTS:
            free_evolution: free evolution matrix
            forced_evolution: forced evolution matrix
        """

        # free evolution of the system
        free_evolution = np.vstack([np.linalg.matrix_power(self.A,k) for k in range(1, N+1)])

        # forced evolution of the system
        forced_evolution = np.zeros((self.n_x*N,self.n_u*N))
        for i in range(0, N):
            for j in range(0, i+1):
                forced_evolution[self.n_x*i:self.n_x*(i+1),self.n_u*j:self.n_u*(j+1)] = np.linalg.matrix_power(self.A,i-j).dot(self.B)

        return [free_evolution, forced_evolution]

    def simulate(self, x0, N, u_sequence=None):
        """
        Returns the list of states obtained simulating the system dynamics.

        INPUTS:
            x0: initial state of the system
            N: number of steps
            u_sequence: list of inputs [u_1, ..., u_{N-1}]

        OUTPUTS:
            x_trajectory: list of states [x_0, ..., x_N]
        """

        # reshape input list if provided
        if u_sequence is None:
            u_sequence = np.zeros((self.n_u*N, 1))
        else:
            u_sequence = np.vstack(u_sequence)

        # derive evolution matrices
        [free_evolution, forced_evolution] = self.evolution_matrices(N)

        # derive state trajectory includion initial state
        if x0.ndim == 1:
            x0 = np.reshape(x0, (x0.shape[0],1))
        x = free_evolution.dot(x0) + forced_evolution.dot(u_sequence)
        x_trajectory = [x0]
        [x_trajectory.append(x[self.n_x*i:self.n_x*(i+1)]) for i in range(0,N)]

        return x_trajectory

    def dare(self, Q, R):
        # cost to go
        P = linalg.solve_discrete_are(self.A, self.B, Q, R)
        # optimal gain
        K = - linalg.inv(self.B.T.dot(P).dot(self.B)+R).dot(self.B.T).dot(P).dot(self.A)
        return [P, K]

    def moas(self, K, X, U):
        # closed loop dynamics
        A_cl = self.A + self.B.dot(K)
        # constraints for the maximum output admissible set
        lhs_cl = np.vstack((X.lhs_min, U.lhs_min.dot(K)))
        rhs_cl = np.vstack((X.rhs_min, U.rhs_min))
        X_cl = Polytope(lhs_cl, rhs_cl)
        X_cl.assemble()
        # compute maximum output admissible set
        return self.moas_closed_loop(A_cl, X_cl)

    @staticmethod
    def moas_closed_loop(A, X):
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
            t: minimum number of steps in the future that define the moas
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

    @staticmethod
    def from_continuous(t_s, A, B):
        """
        Defines a discrete time linear system starting from the continuous time dynamics \dot x = A*x + B*u
        (the exact zero order hold method is used for the discretization).

        INPUTS:
            t_s: sampling time
            A: continuous time state transition matrix
            B: continuous time input to state map

        OUTPUTS:
            sys: discrete time linear system
        """

        # system dimensions
        n_x = np.shape(A)[0]
        n_u = np.shape(B)[1]

        # zero order hold (see Bicchi - Fondamenti di Automatica 2)
        mat_c = np.zeros((n_x+n_u, n_x+n_u))
        mat_c[0:n_x,:] = np.hstack((A, B))
        mat_d = linalg.expm(mat_c*t_s)

        # discrete time dynamics
        A_d = mat_d[0:n_x, 0:n_x]
        B_d = mat_d[0:n_x, n_x:n_x+n_u]

        sys = DTLinearSystem(A_d, B_d)
        return sys

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

        # plot bounds if provided
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

        # plot bounds if provided
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


def plot_state_space_trajectory(x_trajectory, state_components=[0,1], **kwargs):
    """
    Plots the state trajectories as functions of time (2d plot).

    INPUTS:
        x_trajectory: state trajectory \in R^((N+1)*n_x)
        N: time steps
        state_components: components of the state vector to be plotted.
    """
    ax = plt.axes()
    x_0 = (x_trajectory[0][state_components[0]][0], x_trajectory[0][state_components[1]][0])
    plt.scatter(x_0[0], x_0[1], color='b')
    plt.text(x_0[0], x_0[1], r'$x(0)$')
    for i in range(len(x_trajectory)-1):
        head_size = np.linalg.norm(x_trajectory[i+1]-x_trajectory[i])/10.
        x_i = x_trajectory[i][state_components[0]][0]
        y_i = x_trajectory[i][state_components[1]][0]
        x_ip1 = x_trajectory[i+1][state_components[0]][0] - x_i
        y_ip1 = x_trajectory[i+1][state_components[1]][0] - y_i
        traj = ax.arrow(x_i, y_i, x_ip1, y_ip1, head_width=head_size, head_length=head_size*2., fc='b', ec='b')
    # x_min = min([x[state_components[0]][0] for x in x_trajectory])
    # x_max = max([x[state_components[0]][0] for x in x_trajectory])
    # y_min = min([x[state_components[1]][0] for x in x_trajectory])
    # y_max = max([x[state_components[1]][0] for x in x_trajectory])
    # x_frame = (x_max - x_min)/5.
    # y_frame = (y_max - y_min)/5.
    # plt.xlim([x_min-x_frame, x_max+x_frame])
    # plt.ylim([y_min-y_frame, y_max+y_frame])
    plt.xlabel(r'$x_{' + str(state_components[0]+1) + '}$')
    plt.ylabel(r'$x_{' + str(state_components[1]+1) + '}$')
    return
