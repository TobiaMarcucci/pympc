# external imports
import numpy as np
from scipy.linalg import block_diag, solve_discrete_are
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

def mcais(A, X, verbose=False):
    """
    Returns the maximal constraint-admissible (positive) invariant set O_inf for the system x(t+1) = A x(t) subject to the constraint x in X.
    O_inf is also known as maximum output admissible set.
    It holds that x(0) in O_inf <=> x(t) in X for all t >= 0.
    (Implementation of Algorithm 3.2 from: Gilbert, Tan - Linear Systems with State and Control Constraints, The Theory and Application of Maximal Output Admissible Sets.)
    Sufficient conditions for this set to be finitely determined (i.e. defined by a finite number of facets) are: A stable, X bounded and containing the origin.

    Math
    ----------
    At each time step t, we want to verify if at the next time step t+1 the system will go outside X.
    Let's consider X := {x | D_i x <= e_i, i = 1,...,n} and t = 0.
    In order to ensure that x(1) = A x(0) is inside X, we need to consider one by one all the constraints and for each of them, the worst-case x(0).
    We can do this solvin an LP
    V(t=0, i) = max_{x in X} D_i A x - e_i for i = 1,...,n
    if all these LPs has V < 0 there is no x(0) such that x(1) is outside X.
    The previous implies that all the time-evolution x(t) will lie in X (see Gilbert and Tan).
    In case one of the LPs gives a V > 0, we iterate and consider
    V(t=1, i) = max_{x in X, x in A X} D_i A^2 x - e_i for i = 1,...,n
    where A X := {x | D A x <= e}.
    If now all V < 0, then O_inf = X U AX, otherwise we iterate until convergence
    V(t, i) = max_{x in X, x in A X, ..., x in A^t X} D_i A^(t+1) x - e_i for i = 1,...,n
    Once at convergence O_Inf = X U A X U ... U A^t X.

    Arguments
    ----------
    A : numpy.ndarray
        State transition matrix.
    X : instance of Polyhedron
        State-space domain of the dynamical system.
    verbose : bool
        If True prints at each iteration the convergence parameters.

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

    # initialize mcais
    O_inf = copy(X)

    # loop over time
    t = 1
    convergence = False
    while not convergence:

        # solve one LP per facet
        J = X.A.dot(np.linalg.matrix_power(A,t))
        residuals = []
        for i in range(X.A.shape[0]):
            sol = linear_program(- J[i], O_inf.A, O_inf.b)
            residuals.append(- sol['min'] - X.b[i])

        # print status of the algorithm
        if verbose:
            print('Time horizon: ' + str(t) + '.'),
            print('Convergence index: ' + str(max(residuals)) + '.'),
            print('Number of facets: ' + str(O_inf.A.shape[0]) + '.   \r'),

        # convergence check
        new_facets = [i for i, r in enumerate(residuals) if r > 0.]
        if len(new_facets) == 0:
            convergence = True
        else:

            # add (only non-redundant!) facets
            O_inf.add_inequality(J[new_facets], X.b[new_facets])
            t += 1

    # remove redundant facets
    if verbose:
        print('\nMaximal constraint-admissible invariant set found.')
        print('Removing redundant facets ...'),
    O_inf.remove_redundant_inequalities()
    if verbose:
        print('minimal facets are ' + str(O_inf.A.shape[0]) + '.')

    return O_inf

def condense_pwa_system(affine_systems, mode_sequence):
    """
    For the PWA system
    x(t+1) = A_i x(t) + B_i u(t) + c_i    if    (x(t), u(t)) in D_i,
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
    c_bar = np.concatenate((np.zeros(nx), c_sequence[0]))
    for i in range(1, N):
        offset_i = sum([productory(A_sequence[i:j:-1]).dot(c_sequence[j]) for j in range(i)]) + c_sequence[i]
        c_bar = np.concatenate((c_bar, offset_i))

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

def get_state_transition_matrices(x, u, x_next):
    """
    Extracts from the symbolic expression of the state at the next time step the matrices A, B, and c.
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
    A = np.array(x_next.jacobian(x)).astype(np.float64)
    B = np.array(x_next.jacobian(u)).astype(np.float64)

    # offset term
    origin = {xi:0 for xi in x}
    origin.update({ui:0 for ui in u})
    c = np.array(x_next.subs(origin)).astype(np.float64).flatten()
    
    return A, B, c