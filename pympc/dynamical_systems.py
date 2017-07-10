import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from optimization.pnnls import linear_program
from geometry.polytope import Polytope

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
    def from_continuous(A, B, t_s, method='zoh'):
        c = np.zeros((A.shape[0], 1))
        if method == 'zoh':
            A_d, B_d, _ = zero_order_hold(A, B, c, t_s)
        elif method == 'explicit_euler':
            A_d, B_d, _ = explicit_euler(A, B, c, t_s)
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
    def from_continuous(A, B, c, t_s, method='zoh'):
        if method == 'zoh':
            A_d, B_d, c_d = zero_order_hold(A, B, c, t_s)
        elif method == 'explicit_euler':
            A_d, B_d, c_d = explicit_euler(A, B, c, t_s)
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

    def __init__(self, affine_systems, domains):
        self.affine_systems = affine_systems
        self.domains = domains
        self.n_x = affine_systems[0].n_x
        self.n_u = affine_systems[0].n_u
        self.n_sys = len(affine_systems)
        self._x_min = None
        self._x_max = None
        return

    def condense(self, switching_sequence):
        return condense_dynamical_system(self.affine_systems, switching_sequence)

    def simulate(self, x0, u_list):
        N = len(u_list)
        x_list = [x0]
        switching_sequence = []
        for k in range(N):
            domain = self.find_domain(x_list[k], u_list[k])
            if domain is None:
                raise ValueError('Unfeasible state x = ' + str(x_list[k].flatten()) + ' or input u = ' + str(u_list[k].flatten()))
            else:
                sys = self.affine_systems[domain]
                x_list.append(sys.A.dot(x_list[k]) + sys.B.dot(u_list[k]) + sys.c)
                switching_sequence.append(domain)
        return x_list, switching_sequence

    def find_domain(self, x, u):
        for i in range(self.n_sys):
            if self.domains[i].applies_to(np.vstack((x, u))):
                return i
        return None

    @property
    def x_min(self):
        if self._x_min is None:
            min_list = [domain.x_min[0:self.n_x,:] for domain in self.domains]
            self._x_min = np.array([[min([x[i,0] for x in min_list])] for i in range(self.n_x)])
        return self._x_min

    @property
    def x_max(self):
        if self._x_max is None:
            max_list = [domain.x_max[0:self.n_x,:] for domain in self.domains]
            self._x_max = np.array([[max([x[i,0] for x in max_list])] for i in range(self.n_x)])
        return self._x_max

    @staticmethod
    def from_orthogonal_domains(affine_systems, state_domains, input_domains):
        domains = []
        for i in range(len(state_domains)):
            A_i = linalg.block_diag(state_domains[i].lhs_min, input_domains[i].lhs_min)
            b_i = np.vstack((state_domains[i].rhs_min, input_domains[i].rhs_min))
            domain_i = Polytope(A_i, b_i)
            domain_i.assemble()
            domains.append(domain_i)
        return DTPWASystem(affine_systems, domains)





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

def explicit_euler(A, B, c, t_s):
    A_d = A*t_s + np.eye(A.shape[0])
    B_d = B*t_s
    c_d = c*t_s
    return A_d, B_d, c_d


def dare(A, B, Q, R):
    # cost to go
    P = linalg.solve_discrete_are(A, B, Q, R)
    # optimal gain
    K = - linalg.inv(B.T.dot(P).dot(B)+R).dot(B.T).dot(P).dot(A)
    return P, K

def moas_closed_loop_from_orthogonal_domains(A, B, K, X, U):
    # closed loop dynamics
    A_cl = A + B.dot(K)
    # constraints for the maximum output admissible set
    lhs_cl = np.vstack((X.lhs_min, U.lhs_min.dot(K)))
    rhs_cl = np.vstack((X.rhs_min, U.rhs_min))
    X_cl = Polytope(lhs_cl, rhs_cl)
    X_cl.assemble()
    # compute maximum output admissible set
    return moas(A_cl, X_cl)

def moas_closed_loop(A, B, K, D):
    # closed loop dynamics
    A_cl = A + B.dot(K)
    # constraints for the maximum output admissible set
    lhs_cl = D.lhs_min[:,:A.shape[0]] + D.lhs_min[:,A.shape[0]:].dot(K)
    rhs_cl = D.rhs_min
    X_cl = Polytope(lhs_cl, rhs_cl)
    X_cl.assemble()
    # compute maximum output admissible set
    return moas(A_cl, X_cl)

def moas(A, X):
    """
    Returns the maximum output admissible set (see Gilbert, Tan - Linear Systems with State and Control Constraints, The Theory and Application of Maximal Output Admissible Sets) for a non-actuated linear system with state constraints (the output vector is supposed to be the entire state of the system, i.e. y=x and C=I).

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
    print 'Computation of MOAS started...',
    [n_constraints, n_variables] = X.lhs_min.shape
    t = 0
    convergence = False
    while convergence == False:

        # cost function gradients for all i
        J = X.lhs_min.dot(np.linalg.matrix_power(A,t+1))

        # constraints to each LP
        cons_lhs = np.vstack([X.lhs_min.dot(np.linalg.matrix_power(A,k)) for k in range(t+1)])
        cons_rhs = np.vstack([X.rhs_min for k in range(t+1)])

        # list of all minima
        J_sol = []
        for i in range(n_constraints):
            sol = linear_program(np.reshape(-J[i,:], (n_variables,1)), cons_lhs, cons_rhs)
            J_sol.append(-sol.min - X.rhs_min[i])

        # convergence check
        if np.max(J_sol) < 0:
            convergence = True
        else:
            t += 1

    # define polytope
    print 'MOAS found.'
    print 'MOAS facets are ' + str(cons_lhs.shape[0]) + ', removing redundant ones...',
    moas = Polytope(cons_lhs, cons_rhs)
    moas.assemble()
    print 'redundant factes removed, MOAS facets are ' + str(moas.lhs_min.shape[0]) + '.'

    return moas