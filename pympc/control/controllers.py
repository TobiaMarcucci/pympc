# external imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# internal inputs
from pympc.dynamics.discrete_time_systems import AffineSystem, PieceWiseAffineSystem
from pympc.optimization.parametric_programs import MultiParametricQuadraticProgram

class ModelPredictiveController:
    """
    Model predictive controller for linear systems, it solves the optimal control problem
    V*(x(0)) := min_{x(.), u(.)} 1/2 sum_{t=0}^{N-1} x'(t) Q x(t) + u'(t) R u(t) + \frac{1}{2} x'(N) P x(N)
               s.t. x(t+1) = A x(t) + B u(t), t=0, ..., N-1,
                    (x(t), u(t)) in D, t=0, ..., N-1,
                    x(N) in X_N,
    in order to get the optimal control seuquence (u(0), ..., u(N-1)).
    """

    def __init__(self, S, N, Q, R, P, D, X_N):
        """
        Initilizes the controller.

        Arguments
        ----------
        S : instance of LinerSystem
            Linear system to be controlled.
        N : int
            Horizon of the optimal control problem.
        Q : numpy.ndarray
            Quadratic cost for the state.
        R : numpy.ndarray
            Quadratic cost for the input.
        P : numpy.ndarray
            Quadratic cost for the terminal state.
        D : instance of Polyhedron
            Stage constraint for state and inputs.
        X_N : instance of Polyhedron
            Terminal set.
        """

        # store inputs
        self.S = S
        self.N = N
        self.Q = Q
        self.R = R
        self.P = P
        self.D = D
        self.X_N = X_N

        # initilize explicit solution
        self.explicit_solution = None

        # condense mpqp
        self.mpqp = self.condense_program()

    def condense_program(self):
        """
        Generates and stores the optimal control problem in condensed form.

        Returns
        ----------
        instance of MultiParametricQuadraticProgram
            Condensed mpQP.
        """

        # create fake PWA system and use PWA condenser
        c = np.zeros((self.S.nx, 1))
        S = AffineSystem(self.S.A, self.S.B, c)
        S = PieceWiseAffineSystem([S], [self.D])
        mode_sequence = [0]*self.N

        return condense_optimal_control_problem(S, self.Q, self.R, self.P, self.X_N, mode_sequence)

    def feedforward(self, x):
        """
        Given the state x of the system, returns the optimal sequence of N inputs and the related cost.

        Arguments
        ----------
        x : numpy.ndarray
            State of the system.

        Returns
        ----------
        u_feedforward : list of numpy.ndarray
            Optimal control signals for t = 0, ..., N-1.
        V : float
            Optimal value function for the given state.
        """
        sol = self.mpqp.implicit_solve_fixed_point(x)
        if sol['min'] is None:
            return None, None
        u_feedforward = [sol['argmin'][self.S.nu*i : self.S.nu*(i+1), :] for i in range(self.N)]
        V = sol['min']
        return u_feedforward, V

    def feedback(self, x):
        """
        Returns the optimal feedback for the given state x.

        Arguments
        ----------
        x : numpy.ndarray
            State of the system.

        Returns
        ----------
        u_feedback : numpy.ndarray
            Optimal feedback.
        """

        # get feedforward and extract first input
        u_feedforward = self.feedforward(x)[0]
        if u_feedforward is None:
            return None

        return u_feedforward[0]

    def store_explicit_solution(self, **kwargs):
        """
        Solves the mpqp (condensed optimal control problem) explicitly.

        Returns
        ----------
        instance of ExplicitSolution
            Explicit solution of the underlying mpqp problem.
        """

        self.explicit_solution = self.mpqp.solve(**kwargs)

    def feedforward_explicit(self, x):
        """
        Finds the critical region where the state x is and returns the optimal feedforward and the cost to go.

        Arguments
        ----------
        x : numpy.ndarray
            State of the system.

        Returns
        ----------
        u_feedforward : list of numpy.ndarray
            Optimal control signals for t = 0, ..., N-1.
        V : float
            Optimal value function for the given state.
        """

        # check that the explicit solution has been found
        if self.explicit_solution is None:
            raise ValueError('explicit solution not stored.')

        # evaluate lookup table
        u = self.explicit_solution.u(x)
        if u is not None:
            u = [u[t*self.S.nu:(t+1)*self.S.nu, :] for t in range(self.N)]

        return u, self.explicit_solution.V(x)

    def feedback_explicit(self, x):
        """
        Finds the critical region where the state x is and returns the optimal feedback for the given state x.

        Arguments
        ----------
        x : numpy.ndarray
            State of the system.

        Returns
        ----------
        u_feedback : numpy.ndarray
            Optimal feedback.
        """

        # get feedforward and extract first input
        u_feedforward = self.feedforward_explicit(x)[0]
        if u_feedforward is None:
            return None

        return u_feedforward[0]

    def plot_state_space_partition(self, print_active_set=False, **kwargs):
        """
        Finds the critical region where the state x is, and returns the PWA feedforward.

        Arguments
        ----------
        print_active_set : bool
            If True it prints the active set of each critical region in its center.
        """

        # check that the required plot is 2d and that the solution is available
        if self.S.nx != 2:
            raise ValueError('can plot only 2-dimensional partitions.')
        if self.explicit_solution is None:
            raise ValueError('explicit solution not stored.')

        # plot every critical region with random colors
        for cr in self.explicit_solution.critical_regions:
            cr.polyhedron.plot(facecolor=np.random.rand(3), **kwargs)

            # if required print active sets
            if print_active_set:
                plt.text(cr.polyhedron.center[0], cr.polyhedron.center[1], str(cr.active_set))

    def plot_optimal_value_function(self, resolution=100., **kwargs):
        """
        Plots the level sets of the optimal value function V*(x).

        Arguments
        ----------
        resolution : float
            Size of the grid for the contour plot.
        """

        # check dimension of the state
        if self.S.nx != 2:
            raise ValueError('can plot only 2-dimensional value functions.')
        if self.explicit_solution is None:
            raise ValueError('explicit solution not stored.')

        # get feasible set
        feasible_set = self.mpqp.get_feasible_set()

        # create box containing the feasible set
        x_max = max([v[0,0] for v in feasible_set.vertices])
        x_min = min([v[0,0] for v in feasible_set.vertices])
        y_max = max([v[1,0] for v in feasible_set.vertices])
        y_min = min([v[1,0] for v in feasible_set.vertices])

        # create grid
        x = np.arange(x_min, x_max, (x_max-x_min)/resolution)
        y = np.arange(y_min, y_max, (y_max-y_min)/resolution)
        X, Y = np.meshgrid(x, y)

        # evaluate grid
        zs = np.array([self.explicit_solution.V(np.array([[x],[y]])) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)

        # plot
        feasible_set.plot(**kwargs)
        cp = plt.contour(X, Y, Z)
        plt.colorbar(cp)
        plt.title(r'$V^*(x)$')

def condense_optimal_control_problem(S, Q, R, P, X_N, mode_sequence):
    """
    For a given mode sequences, condenses the optimal control problem for a PWA affine system
    min_{x(.), u(.)} 1/2 sum_{t=0}^{N-1} (x'(t) Q x(t) + u'(t) R u(t)) + 1/2 x'(N) P x'(N)
                s.t. x(t+1) = A_{z(t)} x(t) + B_{z(t)} u(t) + c_{z(t)}
                     F_{z(t)} x(t) + G_{z(t)} u(t) <= h_{z(t)}
                     F_N x(N) <= h_N
    where z(t) denotes the mode of the PWA system at time t.
    The problem is then stated as a mpQP with parametric initial state x(0).

    Arguments
    ----------
    S : instance of PieceWiseAffineSystem
        PWA system of the optimal control problem.
    Q : numpy.ndarray
        Hessian of the state cost.
    R : numpy.ndarray
        Hessian of the input cost.
    P : numpy.ndarray
        Hessian of the terminal state cost.
    X_N : instance of Polyhedron
        Terminal state constraint.
    mode_sequence : list of int
        Sequence of the modes of the PWA system.

    Returns
    ----------
    instance of MultiParametricQuadraticProgram
        Condensed mpQP.
    """

    # condense dynamics
    A_bar, B_bar, c_bar = S.condense(mode_sequence)

    # stack cost matrices
    N = len(mode_sequence)
    Q_bar = block_diag(*[Q for i in range(N)] + [P])
    R_bar = block_diag(*[R for i in range(N)])

    # get blocks for condensed objective
    Huu = R_bar + B_bar.T.dot(Q_bar).dot(B_bar)
    Hux = B_bar.T.dot(Q_bar).dot(A_bar)
    Hxx = A_bar.T.dot(Q_bar).dot(A_bar)
    fu = B_bar.T.dot(Q_bar).dot(c_bar)
    fx = A_bar.T.dot(Q_bar).dot(c_bar)
    g = c_bar.T.dot(Q_bar).dot(c_bar)

    # stack constraint matrices
    D_sequence = [S.domains[m]for m in mode_sequence]
    F_bar = block_diag(*[D.A[:,:S.nx] for D in D_sequence] + [X_N.A])
    G_bar = block_diag(*[D.A[:,S.nx:] for D in D_sequence])
    G_bar = np.vstack((
        G_bar,
        np.zeros((X_N.A.shape[0], G_bar.shape[1]))
        ))
    h_bar = np.vstack([D.b for D in D_sequence] + [X_N.b])

    # get blocks for condensed contraints
    Au = G_bar + F_bar.dot(B_bar)
    Ax = F_bar.dot(A_bar)
    b = h_bar - F_bar.dot(c_bar)

    return MultiParametricQuadraticProgram(Huu, Hux, Hxx, fu, fx, g, Au, Ax, b)