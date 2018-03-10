# external imports
import numpy as np
from scipy.linalg import block_diag, solve_discrete_are

# internal imports
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.mathematical_programs import LinearProgram
from pympc.dynamics.discretization_methods import explicit_euler, zero_order_hold
from pympc.dynamics.utils import check_affine_system

class LinearSystem:
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
        Consider the linear quadratic control problem V*(x(0)) = min_{x(.), u(.)} 1/2 sum_{t=0}^inf x(t)' Q x(t) + u(t)' R u(t) subject to x(t+1) = A x(t) + B u(t).
        The optimal solution is u(0) = K x(0) which leads to V*(x(0)) = 1/2 x(0)' P x(0).
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

        # cost to go
        P = solve_discrete_are(self.A, self.B, Q, R)

        # feedback
        K = - np.linalg.inv(self.B.T.dot(P).dot(self.B)+R).dot(self.B.T).dot(P).dot(self.A)

        return P, K
        
    def get_mcais(self, X):
        """
        See the documentation of mcais().
        """
        return mcais(self.A, X)

    def get_mcais_closed_loop(self, K, D):
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
        O_inf, t = mcais(A_cl, X_cl)

        return O_inf, t

    def get_mcais_closed_loop_orthogonal_domains(self, K, X, U):
        """
        Returns the maximal constraint-admissible invariant set O_inf for the closed-loop system X(t+1) = (A + B K) x(t).
        It holds that x(0) in O_inf <=> x(t) in X and u(t) = K x(t) in U for all t >= 0.

        Arguments
        ----------
        K : numpy.ndarray
            Stabilizing feedback gain for the linear system.
        X : instance of Polyhedron
            Constraint set in the state space.
        U : instance of Polyhedron
            Constraint set in the input space.

        Returns
        ----------
        O_inf : instance of Polyhedron
            Maximal constraint-admissible (positive) ivariant.
        t : int
            Determinedness index.
        """

        # state- and input-space constraint set
        D = Polyhedron(
            block_diag(X.A, U.A),
            np.vstack((X.b, U.b))
            )
        O_inf, t = self.get_mcais_closed_loop(K, D)

        return O_inf, t

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
        c = np.zeros((self.A.shape[0], 1))
        S = AffineSystem(self.A, self.B, c)

        # condense as if it was a pwa systems
        A_bar, B_bar, _ = condense_pwa_system([S], [0]*N)

        return A_bar, B_bar

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
        c = np.zeros((A.shape[0], 1))

        # discretize
        if method == 'zero_order_hold':
            A_d, B_d, _ = zero_order_hold(A, B, c, h)
        elif method == 'explicit_euler':
            A_d, B_d, _ = explicit_euler(A, B, c, h)
        else:
            raise ValueError('unknown discretization method.')

        return LinearSystem(A_d, B_d)

class AffineSystem:
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

        # construct affine system
        c = np.zeros((A.shape[0], 1))

        # discretize
        if method == 'zero_order_hold':
            A_d, B_d, c_d = zero_order_hold(A, B, c, h)
        elif method == 'explicit_euler':
            A_d, B_d, c_d = explicit_euler(A, B, c, h)
        else:
            raise ValueError('unknown discretization method.')

        return AffineSystem(A_d, B_d, c_d)

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
            List of the dynamics for eache mode of the system.
        domains : list of instances of Polyhedron
            Domains of each mode of the system.
        """

        # check affine systems
        if len(affine_systems) != len(domains):
            raise ValueError('the number of affine systems has to be equal to the number of domains.')
        nx = set(S.nx for S in affine_systems)
        if len(nx) != 1:
            raise ValueError('all the affine systems must have the same number of states.')
        self.nx = list(nx)[0]
        nu = set(S.nu for S in affine_systems)
        if len(nu) != 1:
            raise ValueError('all the affine systems must have the same number of inputs.')
        self.nu = list(nu)[0]

        # check domains
        nxu = set(D.A.shape[1] for D in domains)
        if len(nxu) != 1:
            raise ValueError('all the domains must have equal dimnesionality.')
        if list(nxu)[0] != self.nx + self.nu:
            raise ValueError('the domains and the affine systems must have coherent dimensions.')

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
                raise ValueError('simulation reached an unfeasible point x = ' + str(x[t].flatten()) + ', u = ' + str(u[t].flatten()) + '.')

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
        xu = np.vstack((x, u))
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
                Dij = Di.get_intersection_with(self.domains[j])

                # check the Chebyshev radius of the intersection
                if Dij.radius > tol:
                    return False

        return True

    @staticmethod
    def from_orthogonal_domains(affine_systems, state_domains, input_domains):
        """
        Instantiates a PWA system with orthogonal domains for state and input, i.e.: D_i = X_i x U_i.

        Arguments
        ----------
        affine_systems : list of instances of AffineSystem
            List of the dynamics for eache mode of the system.
        state_domains : list of instances of Polyhedron
            Domains of each mode of the system for the state vector.
        input_domains : list of instances of Polyhedron
            Domains of each mode of the system for the input vector.
        """

        # check affine systems
        if len(state_domains) != len(input_domains):
            raise ValueError('the number of state domains has to be equal to the number of input domains.')
        nx = set(X.A.shape[1] for X in state_domains)
        if len(nx) != 1:
            raise ValueError('all the state domains must have the same dimensionality.')
        nu = set(U.A.shape[1] for U in input_domains)
        if len(nu) != 1:
            raise ValueError('all the input domains must have the same dimensionality.')

        # direct product of the domains
        domains = []
        for i in range(len(state_domains)):
            A_i = linalg.block_diag(
                state_domains[i].A,
                input_domains[i].A
                )
            b_i = np.vstack((
                state_domains[i].b,
                input_domains[i].b
                ))
            domains.append(Polyhedron(A_i, b_i))

        return PieceWiseAffineSystem(affine_systems, domains)

def mcais(A, X, tol=1.e-9):
    """
    Returns the maximal constraint-admissible (positive) invariant set O_inf for the system x(t+1) = A x(t) subject to the constraint x in X.
    O_inf is also known as maximum output admissible set.
    It holds that x(0) in O_inf <=> x(t) in X for all t >= 0.
    (Implementation of Algorithm 3.2 from: Gilbert, Tan - Linear Systems with State and Control Constraints, The Theory and Application of Maximal Output Admissible Sets.)
    Sufficient conditions for this set to be finitely determined (i.e. defined by a finite number of facets) are: A stable, X bounded and containing the origin.

    Arguments
    ----------
    A : numpy.ndarray
        State transition matrix.
    X : instance of Polyhedron
        State-space domain of the dynamical system.
    tol : float
        Threshold for the checks in the algorithm.

    Returns:
    ----------
    O_inf : instance of Polyhedron
        Maximal constraint-admissible (positive) ivariant.
    t : int
        Determinedness index.
    """

    # ensure convergence of the algorithm
    eig_max = np.max(np.absolute(np.linalg.eig(A)[0]))
    if eig_max > 1.:
        raise ValueError('unstable system, cannot derive maximal constraint-admissible set.')
    [nc, nx] = X.A.shape
    if not X.contains(np.zeros((nx, 1))):
        raise ValueError('the origin is not contained in the constraint set, cannot derive maximal constraint-admissible set.')
    if not X.bounded:
        raise ValueError('unbounded constraint set, cannot derive maximal constraint-admissible set.')

    # Gilber and Tan algorithm
    t = 0
    convergence = False
    while not convergence:

        # cost function gradients for all i
        J = X.A.dot(np.linalg.matrix_power(A,t+1))

        # constraints to each LP
        F = np.vstack([X.A.dot(np.linalg.matrix_power(A,k)) for k in range(t+1)])
        g = np.vstack([X.b for k in range(t+1)])
        O_inf = Polyhedron(F, g)

        # list of all minima
        J_sol = []
        lp = LinearProgram(O_inf)
        for i in range(nc):
            lp.f = - J[i:i+1,:].T
            sol = lp.solve()
            J_sol.append(-sol['min'] - X.b[i])
        if np.max(J_sol) < tol:
            convergence = True
        else:
            t += 1

    return O_inf, t

def condense_pwa_system(affine_systems, mode_sequence):
    """
    For the PWA system
    x(t+1) = A_i x(t) + B_i u(t) + c_i    if    (x(t), u(t)) \in D_i,
    given the mode sequence z = (z(0), ... , z(N-1)), returns the matrices A_bar, B_bar, c_bar such that
    x_bar = A_bar x(0) + B_bar u_bar + c_bar
    with x_bar = (x(0), ... , x(N)) and u_bar = (u(0), ... , u(N-1)).

    Arguments
    ----------
    affine_systems : list of instances of AffineSystem
        State transition matrix (assumed to be invertible).
    mode_sequence : list of int
        Sequence of the modes that the PWA system is in for time t = 0, 1, ..., N-1.

    Returns
    ----------
    A_bar : numpy.ndarray
        Condensed free evolution matrix.
    B_bar : numpy.ndarray
        Condensed input to state matrix.
    c_bar : numpy.ndarray
        Condensed offset term matrix.
    """

    # system dimensions
    nx = affine_systems[0].nx
    nu = affine_systems[0].nu
    N = len(mode_sequence)

    # matrix sequence
    A_sequence = [affine_systems[mode_sequence[i]].A for i in range(N)]
    B_sequence = [affine_systems[mode_sequence[i]].B for i in range(N)]
    c_sequence = [affine_systems[mode_sequence[i]].c for i in range(N)]

    # free evolution of the system
    A_bar = np.vstack([productory(A_sequence[i::-1]) for i in range(N)])
    A_bar = np.vstack((np.eye(nx), A_bar))

    # forced evolution of the system
    B_bar = np.zeros((nx*N,nu*N))
    for i in range(N):
        for j in range(i):
            B_bar[nx*i:nx*(i+1), nu*j:nu*(j+1)] = productory(A_sequence[i:j:-1]).dot(B_sequence[j])
        B_bar[nx*i:nx*(i+1), nu*i:nu*(i+1)] = B_sequence[i]
    B_bar = np.vstack((np.zeros((nx, nu*N)), B_bar))

    # evolution related to the offset term
    c_bar = np.vstack((np.zeros((nx,1)), c_sequence[0]))
    for i in range(1, N):
        offset_i = sum([productory(A_sequence[i:j:-1]).dot(c_sequence[j]) for j in range(i)]) + c_sequence[i]
        c_bar = np.vstack((c_bar, offset_i))

    return A_bar, B_bar, c_bar

def productory(matrix_list):
    """
    Multiplies from lest to right the matrices in the list matrix_list.

    Arguments
    ----------
    matrix_list : list of numpy.ndarray
        List of matrices to be multiplied.

    Returns
    ----------
    M : numpy.ndarray
        Product of the matrices in matrix_list.
    """

    # start wiht the first elment and multy all the others
    A = matrix_list[0]
    for B in matrix_list[1:]:
        A = A.dot(B)

    return A