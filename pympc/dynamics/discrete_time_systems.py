# external imports
import numpy as np
from scipy.linalg import solve_discrete_are
from copy import copy

# internal imports
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.programs import linear_program
from pympc.dynamics.discretization_methods import explicit_euler, zero_order_hold
from pympc.dynamics.utils import check_affine_system

class LinearSystem(object):
    """
    Discrete-time linear systems in the form x(t+1) = A x(t) + B u(t).
    """

    def __init__(self, A, B):
        """
        Initializes the discrete-time linear system.

        Arguments
        ----------
        A : numpy.ndarray
            State transition matrix (assumed to be invertible).
        B : numpy.ndarray
            Input to state map.
        """

        # check inputs
        check_affine_system(A, B)

        # store inputs
        self.A = A
        self.B = B

        # system size
        self.nx, self.nu = B.shape

        # property variables
        self._controllable = None

        return

    def simulate(self, x0, u):
        """
        Simulates the system starting from the state x0 and applying the sequence of controls u.

        Arguments
        ----------
        x0 : numpy.ndarray
            Initial state.
        u : list of numpy.ndarray
            Sequence of inputs for t = 0, 1, ..., N-1.

        Returns
        ----------
        x : list of numpy.ndarray
            Sequence of states for t = 0, 1, ..., N.
        """

        # simulate (do not call condense, too slow)
        x = [x0]
        for v in u:
            x.append(self.A.dot(x[-1]) + self.B.dot(v))

        return x

    def simulate_closed_loop(self, x0, N, K):
        """
        Simulates the system starting from the state x0 for N steps in closed loop with the feedback law u = K x.

        Arguments
        ----------
        x0 : numpy.ndarray
            Initial state.
        K : numpy.ndarray
            Feedback gain.
        N : int
            Number of simulation steps.

        Returns
        ----------
        x : list of numpy.ndarray
            Sequence of states for t = 0, 1, ..., N.
        """

        # simulate
        x = [x0]
        for t in range(N):
            x.append((self.A + self.B.dot(K)).dot(x[-1]))

        return x

    def solve_dare(self, Q, R):
        """
        Returns the solution of the Discrete Algebraic Riccati Equation (DARE).
        Consider the linear quadratic control problem V*(x(0)) = min_{x(.), u(.)} 1/2 sum_{t=0}^inf x'(t) Q x(t) + u'(t) R u(t) subject to x(t+1) = A x(t) + B u(t).
        The optimal solution is u(0) = K x(0) which leads to V*(x(0)) = 1/2 x'(0) P x(0).
        The pair A, B is assumed to be controllable.

        Arguments
        ----------
        Q : numpy.ndarray
            Quadratic cost for the state (positive semidefinite).
        R : numpy.ndarray
            Quadratic cost for the input (positive definite).

        Returns
        ----------
        P : numpy.ndarray
            Hessian of the cost-to-go (positive definite).
        K : numpy.ndarray
            Optimal feedback gain matrix.
        """

        # check controllability
        if not self.controllable:
            raise ValueError('uncontrollable system, cannot solve Riccati equation.')

        # cost to go
        P = solve_discrete_are(self.A, self.B, Q, R)

        # feedback
        K = - np.linalg.inv(self.B.T.dot(P).dot(self.B)+R).dot(self.B.T).dot(P).dot(self.A)

        return P, K

    def mcais(self, K, D, **kwargs):
        """
        Returns the maximal constraint-admissible invariant set O_inf for the closed-loop system X(t+1) = (A + B K) x(t).
        It holds that x(0) in O_inf <=> (x(t), u(t) = K x(t)) in D for all t >= 0.

        Arguments
        ----------
        K : numpy.ndarray
            Stabilizing feedback gain for the linear system.
        D : instance of Polyhedron
            Constraint set in the state and input space.

        Returns
        ----------
        O_inf : instance of Polyhedron
            Maximal constraint-admissible (positive) ivariant.
        t : int
            Determinedness index.
        """

        # closed loop dynamics
        A_cl = self.A + self.B.dot(K)

        # state-space constraint set
        X_cl = Polyhedron(
            D.A[:,:self.nx] + D.A[:,self.nx:].dot(K),
            D.b
            )
        O_inf = mcais(A_cl, X_cl, **kwargs)

        return O_inf

    def condense(self, N):
        """
        Constructs the matrices A_bar and B_bar such that x_bar = A_bar x(0) + B_bar u_bar with x_bar = (x(0), ... , x(N)) and u_bar = (u(0), ... , u(N-1)).

        Arguments
        ----------
        N : int
            Number of time-steps.

        Returns
        ----------
        A_bar : numpy.ndarray
            Condensed free evolution matrix.
        B_bar : numpy.ndarray
            Condensed input to state matrix.
        """

        # construct fake affine system
        c = np.zeros(self.A.shape[0])
        S = AffineSystem(self.A, self.B, c)

        # condense as if it was a pwa systems
        A_bar, B_bar, _ = condense_pwa_system([S], [0]*N)

        return A_bar, B_bar

    @property
    def controllable(self):

        # check if already computes
        if self._controllable is not None:
            return self._controllable

        # check controllability
        controllable = False
        R = np.hstack([np.linalg.matrix_power(self.A, i).dot(self.B) for i in range(self.nx)])
        self._controllable = np.linalg.matrix_rank(R) == self.nx

        return self._controllable

    @staticmethod
    def from_continuous(A, B, h, method='zero_order_hold'):
        """
        Instantiates a discrete-time linear system starting from its continuous time representation.

        Arguments
        ----------
        A : numpy.ndarray
            Continuous-time state transition matrix (assumed to be invertible).
        B : numpy.ndarray
            Continuous-time state input to state map.
        h : float
            Discretization time step.
        method : str
            Discretization method: 'zero_order_hold', or 'explicit_euler'.
        """

        # check inputs
        check_affine_system(A, B, None, h)

        # construct affine system
        c = np.zeros(A.shape[0])

        # discretize
        if method == 'zero_order_hold':
            A_d, B_d, _ = zero_order_hold(A, B, c, h)
        elif method == 'explicit_euler':
            A_d, B_d, _ = explicit_euler(A, B, c, h)
        else:
            raise ValueError('unknown discretization method.')

        return LinearSystem(A_d, B_d)

    @staticmethod
    def from_symbolic(x, u, x_next):
        """
        Instatiates a LinearSystem starting from the symbolic value of the next state.

        Arguments
        ----------
        x : sympy matrix filled with sympy symbols
            Symbolic state of the system.
        u : sympy matrix filled with sympy symbols
            Symbolic input of the system.
        x_next : sympy matrix filled with sympy symbolic linear expressions
            Symbolic value of the state update.
        """

        # state transition matrices
        A, B, c = get_state_transition_matrices(x, u, x_next)

        # check that offset setm is zero
        if not np.allclose(c, np.zeros(x.shape[0])):
            raise ValueError('The given system has a non zero offset.')

        return LinearSystem(A, B)

    @staticmethod
    def from_symbolic_continuous(x, u, x_dot, h, method='zero_order_hold'):
        """
        Instatiates a LinearSystem starting from the symbolic value of the next state.

        Arguments
        ----------
        x : sympy matrix filled with sympy symbols
            Symbolic state of the system.
        u : sympy matrix filled with sympy symbols
            Symbolic input of the system.
        x_dot : sympy matrix filled with sympy symbolic linear expressions
            Symbolic value of the state time derivative.
        h : float
            Discretization time step.
        method : str
            Discretization method: 'zero_order_hold', or 'explicit_euler'.
        """

        # state transition matrices
        A, B, c = get_state_transition_matrices(x, u, x_dot)

        # check that offset setm is zero
        if not np.allclose(c, np.zeros(x.shape[0])):
            raise ValueError('The given system has a non zero offset.')

        return LinearSystem.from_continuous(A, B, h, method)

class AffineSystem(object):
    """
    Discrete-time affine systems in the form x(t+1) = A x(t) + B u(t) + c.
    """

    def __init__(self, A, B, c):
        """
        Initializes the discrete-time affine system.

        Arguments
        ----------
        A : numpy.ndarray
            State transition matrix (assumed to be invertible).
        B : numpy.ndarray
            Input to state map.
        c : numpy.ndarray
            Offset term in the dynamics.
        """

        # check inputs
        check_affine_system(A, B, c)

        # store inputs
        self.A = A
        self.B = B
        self.c = c

        # system size
        self.nx, self.nu = B.shape

    def simulate(self, x0, u):
        """
        Simulates the system starting from the state x0 and applying the sequence of controls u.

        Arguments
        ----------
        x0 : numpy.ndarray
            Initial state.
        u : list of numpy.ndarray
            Sequence of inputs for t = 0, 1, ..., N-1.

        Returns
        ----------
        x : list of numpy.ndarray
            Sequence of states for t = 0, 1, ..., N.
        """

        # simulate (do not call condense, too slow)
        x = [x0]
        for v in u:
            x.append(self.A.dot(x[-1]) + self.B.dot(v) + self.c)

        return x

    def condense(self, N):
        """
        Constructs the matrices A_bar, B_bar cnd c_bar such that x_bar = A_bar x(0) + B_bar u_bar + c_bar with x_bar = (x(0), ... , x(N)) and u_bar = (u(0), ... , u(N-1)).

        Arguments
        ----------
        N : int
            Number of time-steps.

        Returns
        ----------
        A_bar : numpy.ndarray
            Condensed free evolution matrix.
        B_bar : numpy.ndarray
            Condensed input to state matrix.
        c_bar : numpy.ndarray
            Condensed offset term matrix.
        """

        # call the function for pwa systems with fake pwa system
        return condense_pwa_system([self], [0]*N)

    @staticmethod
    def from_continuous(A, B, c, h, method='zero_order_hold'):
        """
        Instantiates a discrete-time affine system starting from its continuous time representation.

        Arguments
        ----------
        A : numpy.ndarray
            Continuous-time state transition matrix (assumed to be invertible).
        B : numpy.ndarray
            Continuous-time state input to state map.
        c : numpy.ndarray
            Offset term in the dynamics.
        h : float
            Discretization time step.
        method : str
            Discretization method: 'zero_order_hold', or 'explicit_euler'.
        """

        # check inputs
        check_affine_system(A, B, c, h)

        # discretize
        if method == 'zero_order_hold':
            A_d, B_d, c_d = zero_order_hold(A, B, c, h)
        elif method == 'explicit_euler':
            A_d, B_d, c_d = explicit_euler(A, B, c, h)
        else:
            raise ValueError('unknown discretization method.')

        return AffineSystem(A_d, B_d, c_d)

    @staticmethod
    def from_symbolic(x, u, x_next):
        """
        Instatiates a AffineSystem starting from the symbolic value of the next state.

        Arguments
        ----------
        x : sympy matrix filled with sympy symbols
            Symbolic state of the system.
        u : sympy matrix filled with sympy symbols
            Symbolic input of the system.
        x_next : sympy matrix filled with sympy symbolic linear expressions
            Symbolic value of the state update.
        """

        return AffineSystem(*get_state_transition_matrices(x, u, x_next))

    @staticmethod
    def from_symbolic_continuous(x, u, x_dot, h, method='zero_order_hold'):
        """
        Instatiates a LinearSystem starting from the symbolic value of the next state.

        Arguments
        ----------
        x : sympy matrix filled with sympy symbols
            Symbolic state of the system.
        u : sympy matrix filled with sympy symbols
            Symbolic input of the system.
        x_dot : sympy matrix filled with sympy symbolic linear expressions
            Symbolic value of the state time derivative.
        h : float
            Discretization time step.
        method : str
            Discretization method: 'zero_order_hold', or 'explicit_euler'.
        """

        # get state transition matrices
        A, B, c = get_state_transition_matrices(x, u, x_dot)

        return AffineSystem.from_continuous(A, B, c, h, method)

class PieceWiseAffineSystem(object):
    """
    Discrete-time piecewise-affine systems in the form x(t+1) = A_i x(t) + B_i u(t) + c_i if (x(t), u(t)) in D_i := {(x,u) | F_i x + G_i u <= h_i}.
    """

    def __init__(self, affine_systems, domains):
        """
        Initializes the discrete-time piecewise-affine system.

        Arguments
        ----------
        affine_systems : list of instances of AffineSystem
            List of the dynamics for each mode of the system (in case a LinearSystem is passed through this list it is automatically converted to an instance of AffineSystem).
        domains : list of instances of Polyhedron
            Domains of each mode of the system.
        """

        # same number of systems and domains
        if len(affine_systems) != len(domains):
            raise ValueError('the number of affine systems has to be equal to the number of domains.')

        # same number of states for each system
        nx = set(S.nx for S in affine_systems)
        if len(nx) != 1:
            raise ValueError('all the affine systems must have the same number of states.')
        self.nx = list(nx)[0]

        # same number of inputs for each system
        nu = set(S.nu for S in affine_systems)
        if len(nu) != 1:
            raise ValueError('all the affine systems must have the same number of inputs.')
        self.nu = list(nu)[0]

        # same dimensions for each domain
        nxu = set(D.A.shape[1] for D in domains)
        if len(nxu) != 1:
            raise ValueError('all the domains must have equal dimnesionality.')

        # dimension of each domain equal too number of states plus number of inputs
        if list(nxu)[0] != self.nx + self.nu:
            raise ValueError('the domains and the affine systems must have coherent dimensions.')

        # make instances of LinearSystem instances of AffineSystem
        for i, S in enumerate(affine_systems):
            if isinstance(S, LinearSystem):
                c = np.zeros(self.nx)
                affine_systems[i] = AffineSystem(S.A, S.B, c)

        # store inputs
        self.affine_systems = affine_systems
        self.domains = domains
        self.nm = len(affine_systems)

    def condense(self, mode_sequence):
        """
        See the documentation of condense_pwa_system().
        """
        return condense_pwa_system(self.affine_systems, mode_sequence)

    def simulate(self, x0, u):
        """
        Given the initial state x0 and a list of inputs, simulates the PWA dynamics.
        If the couple (x(t), u(t)) goes out of the domains D_i raises a ValueError.

        Arguments
        ----------
        x0 : numpy.ndarray
            Initial state.
        u : list of numpy.ndarray
            Sequence of inputs for t = 0, 1, ..., N-1.

        Returns
        ----------
        x : list of numpy.ndarray
            Sequence of states for t = 0, 1, ..., N.
        mode_sequence : list of int
            Sequence of the modes that the PWA system is in for time t = 0, 1, ..., N-1.
        """

        # initialize output
        x = [x0]
        mode_sequence = []

        # simulate
        for t in range(len(u)):
            mode = self.get_mode(x[t], u[t])

            # if outside the domain, raise value error
            if mode is None:
                raise ValueError('simulation reached an unfeasible point x = ' + str(x[t]) + ', u = ' + str(u[t]) + '.')

            # compute next state and append values
            else:
                S = self.affine_systems[mode]
                x.append(S.A.dot(x[t]) + S.B.dot(u[t]) + S.c)
                mode_sequence.append(mode)

        return x, mode_sequence

    def get_mode(self, x, u):
        """
        Given (x,u) returns the i such that (x,u) in D_i.

        Arguments
        ----------
        x : numpy.ndarray
            Value for the state vector.
        u : numpy.ndarray
            Value for the input vector.

        Returns
        ----------
        i : int
            Index of the mode of the system (None ix there is no domain that contains (x,u)).
        """

        # loop over the domains
        xu = np.concatenate((x, u))
        for i, D in enumerate(self.domains):
            if D.contains(xu):
                return i

        return None

    def is_well_posed(self, tol=1.e-7):
        """
        Check if the domains of the pwa system are well posed (i.e. if the intersection of the interior of D_i with the interior of D_j is empty for all i and j != i).

        Arguments
        ----------
        tol : float
            Maximum penetration of two sets to assume that their interiors do not intersect.

        Returns:
        ----------
        well_posed : bool
            True if the domains are well posed, False otherwise.
        """

        # loop over al the combinations (avoiding to check twice)
        for i, Di in enumerate(self.domains):
            for j in range(i+1, self.nm):
                Dij = Di.intersection(self.domains[j])

                # check the Chebyshev radius of the intersection
                if Dij.radius > tol:
                    return False

        return True

class MixedLogicalDynamicalSystem(object):
    """
    Hybrid dynamical system in the form
    |x_c_+| = |A_cc A_cb| |x_c| + |B_cc B_cb| |u_c| + |C_cc C_cb| |s_c| + |d_c|,
    |x_b_+|   |A_bc A_bb| |x_b|   |B_bc B_bb| |u_b|   |C_bc C_bb| |s_b|   |d_b|
    subj. to  |F_ec F_eb| |x_c| + |G_ec G_eb| |u_c| + |H_ec H_eb| |s_c| <= |l_e|.
              |F_ic F_ib| |x_b|   |G_ic G_ib| |u_b|   |H_ic H_ib| |s_b|  = |l_i|
    Here x is the continuous and binary state, u is the input, s are the auxiliary variables.
    """

    def __init__(self, A, B, F, G, g):
        """
        Initializes the mixed logical dynamical system.

        Arguments
        ----------
        A : dict of 2d numpy arrays
            State transition matrices, keys: 'cc', 'cb', 'bc', 'bb'.
        B : dict of 2d numpy arrays
            Input matrices, keys: 'cc', 'cb', 'bc', 'bb'.
        C : dict of 2d numpy arrays
            Auxiliary variable matrices, keys: 'cc', 'cb', 'bc', 'bb'.
        d : dict of 1d numpy arrays
            Offset terms in the dynamics, keys: 'c', 'b'.
        F : dict of 2d numpy arrays
            State constraint matrices, keys: 'ec', 'eb', 'ic', 'ib'.
        G : dict of 2d numpy arrays
            Input constraint matrices, keys: 'ec', 'eb', 'ic', 'ib'.
        H : dict of 2d numpy arrays
            Auxiliary variable constraint matrices, keys: 'e', 'i'.
        l : 1d numpy array
            Right hand side of the constraints.
        """

        # store data
        self.A = A
        self.B = B
        self.b = b
        self.F = F
        self.G = G
        self.g = g