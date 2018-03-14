# external imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# internal inputs
from pympc.dynamics.discrete_time_systems import AffineSystem, PieceWiseAffineSystem
from pympc.optimization.parametric_programs import MultiParametricQuadraticProgram
from pympc.optimization.convex_programs import LinearProgram

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
        self.mpqp = self._condense_program()

    def _condense_program(self):
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




















class HybridModelPredictiveController(object):

	def __init__(self, S, N, Q, R, P, X_N):
		"""
        Initilizes the controller.

        Arguments
        ----------
        S : instance of PieceWiseAffineSystem
            PWA system to be controlled.
        N : int
            Horizon of the optimal control problem.
        Q : numpy.ndarray
            Quadratic cost for the state.
        R : numpy.ndarray
            Quadratic cost for the input.
        P : numpy.ndarray
            Quadratic cost for the terminal state.
        X_N : instance of Polyhedron
            Terminal set.
        """

        # store inputs
        self.S = S
        self.N = N
        self.Q = Q
        self.R = R
        self.P = P
        self.X_N = X_N

        # get bigMs
        self._alpha, self._beta = self._get_bigM_dynamics()
        self._gamma = self._get_bigM_domains()

        # condense miqp
        self.mpmiqp = self._condense_program()


    def _get_bigM_dynamics(self):
        """
        Computes all the bigMs for the dynamics of the PWA system.
        The PWA system has the dynamics
        x(t+1) = A_i x(t) + B_i u(t) + c_i if (x(t),u(t)) in D_i,
        where i in {1, ..., s}.
        In order to express it in mixed-integer form, for t = 0, ..., N-1, we introduce the auxiliary variables z_i(t), and we set
        x(t+1) = sum_{i=1}^s z_i(t).
        We now reformulate the dynamics as
        z_i(t) >= alpha_ii delta_i(t),
        z_i(t) <= beta_ii delta_i(t),
        A_i x(t) + B_i u(t) + c_i - z_i(t) >= sum_{j=1, j!=i}^s alpha_ij delta_j(t),
        A_i x(t) + B_i u(t) + c_i - z_i(t) <= sum_{j=1, j!=i}^s beta_ij delta_j(t).
        Here alpha_ij (<< 0) and beta_ij (>> 0) are both vectors of bigMs and delta_j(t) is a binary variable (equal to 1 if the system is in mode j, zero otherwise).
        If the system is in mode k at time t (i.e. delta_k(t) = 1), we have that
        z_i(t) = 0, for all i != k,
        z_k(t) >= alpha_kk,
        z_k(t) <= beta_kk,
        A_i x(t) + B_i u(t) + c_i - z_i(t) >= alpha_ik, for all i != k,
        A_i x(t) + B_i u(t) + c_i - z_i(t) <= beta_ik, for all i != k,
        A_k x(t) + B_k u(t) + c_k = z_k(t),
        that sets
        x(t+1) = z_k(t) = A_k x(t) + B_k u(t) + c_k
        as desired.
        It is very important to choose the bigMs as tight as possible, for this reason we set
        alpha_ij := min_{(x,u) in D_j} A_i x + B_i u + c_i,
        beta_ij := max_{(x,u) in D_j} A_i x + B_i u + c_i.
        (Note that the previous are a number of LPs equal to the number of states.)
        The previous ensures that when the system is mode j != i, the dynamics A_i x + B_i u + c_i is lower bounded by alpha_ik and upper bounded by beta_ij.

        Returns
        ----------
        alpha : list of lists of numpy.ndarray
            alpha[i][j] is the vector alpha_ij defined above.
        beta : list of lists of numpy.ndarray
            beta[i][j] is the vector beta_ij defined above.
        """

        # initialize list of bigMs
        alpha = []
        beta = []

        # outer loop over the number of affine systems
        for i, S_i in enumerate(self.S.affine_systems):
        	alpha_i = []
        	beta_i = []
        	A_i = np.hstack((S_i.A, S_i.B))

        	# inner loop over the number of affine systems
        	for j, S_j in enumerate(self.S.affine_systems):
        		alpha_ij = []
        	    beta_ij = []
        	    lp = LinearProgram(D_j)

        	    # solve two LPs for each component of the state vector
                for k in range(S_i.nx):
                	lp.f = A_i[k:k+1,:].T
                	sol = lp.solve()
                	alpha_ij.append(sol['min'] + S_i.c[k,0])
                	lp.f = - lp.f
                	sol = lp.solve()
                	beta_ij.append(- sol['min'] + S_i.c[k,0])

                # close inner loop appending bigMs
                alpha_i.append(np.vstack(alpha_ij))
                beta_i.append(np.vstack(beta_ij))

            # close outer loop appending bigMs
            alpha.append(np.vstack(alpha_i))
            beta.append(np.vstack(beta_i))

        return alpha, beta

    def _get_bigM_domains(self):
        """
        Computes all the bigMs for the domains of the PWA system.
        Each one of the s domains of the PWA system has the form
        D_i = {(x,u) | F_i x + G_i u <= h_i}.
        The bigM reformulation (for t = 0, ..., N-1) of this constraint is
        F_i x(t) + G_i u(t) <= h_i + sum_{j=1, j!=i}^s gamma_ij delta_j(t).
        Here gamma_ij (>> 0) is a vector of bigMs and delta_j(t) is a binary variable (equal to 1 if the system is in mode j, zero otherwise).
        If the system is in mode k at time t (i.e. delta_k(t) = 1), we have that
        F_i x(t) + G_i u(t) <= h_i + gamma_ik, for all i != k,
        F_k x(t) + G_k u(t) <= h_k,
        hence we force (x(t),u(t)) to belong to D_k, whereas the other constraints are redundant because gamma_ik >> 0.
        It is very important to choose the bigMs as tight as possible, for this reason we set
        gamma_ij := max_{(x,u) in D_j} F_i x + G_i u - h_i.
        (Note that the previous are a number of LPs equal to the number of rows of F_i.)
        The previous ensures that when the system is mode j != i, gamma_ij is always bigger than the left-hand side (i.e. F_i x + G_i u - h_i).

        Returns
        ----------
        gamma : list of lists of numpy.ndarray
            gamma[i][j] is the vector gamma_ij defined above.
        """

        # initialize list of bigMs
        gamma = []

        # outer loop over the number of affine systems
        for i, D_i in enumerate(self.S.domains):
            gamma_i = []

            # inner loop over the number of affine systems
            for j, D_j in enumerate(self.S.domains):
                gamma_ij = []
                lp = LinearProgram(D_j)

                # solve one LP for each inequality of the ith domain
                for k in range(D_i.A.shape[0]):
                    lp.f = -D_i.A[k:k+1,:].T
                    sol = lp.solve()
                    gamma_ij.append(- sol['min'] - D_i.b[k,0])

                # close inner loop appending bigMs
                gamma_i.append(np.vstack(gamma_ij))

            # close outer loop appending bigMs
            gamma.append(gamma_i)

        return gamma

    def _condense_program(self):
    	Ex, Eu, Ez, Ed, e = self._build_inequalities()
    	Ex_bar, Eu_bar, Ez_bar, Ed_bar, e_bar = self._build_inequalities(Ex, Eu, Ez, Ed, e)
    	A_bar, Bz_bar = slef._condense_equalities()
    	H, f, g, A, b = _condense_cost() # with H, f, A dictionaries
    	return MultiParametricMixedIntegerQuadraticProgram(H, f, g, A, b)

    def _build_inequalities(self):
    	pass

    def _condense_equlities(self):
    	pass

    def _condense_inequlities(self):
    	pass

    def _condense_cost(self):
    	pass

    def feedforward(self, x):
    	sol = self.mpmiqp.solve(x)
    	if sol['min'] is None:
    		return None, None
    	u_feedforward = sol['argmin_continuous'][:self.S.nu*self.N, :]
    	V = sol['min']
    	return u_feedforward, V


