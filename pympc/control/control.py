# external imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# internal inputs
from pympc.dynamics.discrete_time_systems import AffineSystem, PieceWiseAffineSystem
from pympc.optimization.parametric_programs import MultiParametricQuadraticProgram

class ModelPredictiveController:

    def __init__(self, S, N, Q, R, P, D, X_N):
        self.S = S
        self.N = N
        self.Q = Q
        self.R = R
        self.P = P
        self.D = D
        self.X_N = X_N
        self.mpqp = self.condense_program()
        self.explicit_solution = None
        return

    def condense_program(self):
        c = np.zeros((self.S.nx, 1))
        S = AffineSystem(self.S.A, self.S.B, c)
        S = PieceWiseAffineSystem([S], [self.D])
        mode_sequence = [0]*self.N
        return condense_optimal_control_problem(S, self.Q, self.R, self.P, self.X_N, mode_sequence)

    def feedforward(self, x):
        """
        Given the state of the system, returns the optimal sequence of N inputs and the related cost.
        """
        sol = self.mpqp.implicit_solve_fixed_point(x)
        if sol['min'] is None:
            return None, None
        u_feedforward = [sol['argmin'][self.S.nu*i : self.S.nu*(i+1), :] for i in range(self.N)]
        return u_feedforward, sol['min']

    def feedback(self, x):
        """
        Returns the a single input vector (the first of feedforward(x)).
        """
        u_feedforward = self.feedforward(x)[0]
        if u_feedforward is None:
            return None
        return u_feedforward[0]

    def store_explicit_solution(self, verbose):
        """
        Arguments
        ----------
        verbose : bool
            If True it prints the number active sets found at each iteration of the solver.
        """
        self.explicit_solution = self.mpqp.solve(verbose=verbose)

    def feedforward_explicit(self, x):
        """
        Finds the critical region where the state x is, and returns the PWA feedforward.
        """
        if self.explicit_solution is None:
            raise ValueError('explicit solution not stored.')
        u = self.explicit_solution.u(x)
        u = [u[t*self.S.nu:(t+1)*self.S.nu, :] for t in range(self.N)]
        return u, self.explicit_solution.V(x)

    def feedback_explicit(self, x):
        """
        Returns the a single input vector (the first of feedforward_explicit(x)).
        """
        u_feedforward = self.feedforward_explicit(x)[0]
        if u_feedforward is None:
            return None
        return u_feedforward[0]

    def plot_state_space_partition(self, print_active_set=False, **kwargs):
        """
        Finds the critical region where the state x is, and returns the PWA feedforward.
        """
        if self.S.nx != 2:
            raise ValueError('can plot only 2-dimensional partitions.')
        if self.explicit_solution is None:
            raise ValueError('explicit solution not stored.')
        for cr in self.explicit_solution.critical_regions:
            cr.polyhedron.plot(facecolor=np.random.rand(3), **kwargs)
            if print_active_set:
                plt.text(cr.polyhedron.center[0], cr.polyhedron.center[1], str(cr.active_set))

    def plot_optimal_value_function(self, resolution=200., **kwargs):

        # check dimension of the state
        if self.S.nx != 2:
            raise ValueError('can plot only 2-dimensional value functions.')

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