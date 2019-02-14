# external imports
import numpy as np
from scipy.linalg import block_diag

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

def check_affine_system(A, B, c=None, h=None):
    """
    Check that the matrices A, B, and c of an affine system have compatible sizes.

    Arguments
    ----------
    A : numpy.ndarray
        State transition matrix.
    B : numpy.ndarray
        Input to state map.
    c : numpy.ndarray
        Offset term.
    h : float
        Discretization time step.
    """

    # A square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix.')

    # equal number of rows for A and B
    if A.shape[0] != B.shape[0]:
        raise ValueError('A and B must have the same number of rows.')

    # check c
    if c is not None:
        if c.ndim > 1:
            raise ValueError('c must be a 1-dimensional array.')
        if A.shape[0] != c.size:
            raise ValueError('A and c must have the same number of rows.')

    # check h
    if h is not None:
        if h < 0:
            raise ValueError('the time step h must be positive.')

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

def pwa_to_mld_convex_hull(pwa):
    """
    Converts a Piecewise Affine (PWA) system in a Mixed Logical Dynamical (MLD) system.
    It uses the convex hull method from Sec.3.2.1 of Marcucci, Tedrake - 'Mixed-Integer Formulations for Optimal Control of Piecewise-Affine Systems'.

    Math
    ----------
    Given the PWA system
    x_+ = A^i x + B^i u + c^i if F^i x + G^i u <= h^i, i = 1, ..., s,
    we derive a mixed-integer formulation of this dynamics as
    F^i x^i + G^i u^i <= m^i h^i, i = 1, ..., s,
    sum_{i=1}^n m^i = 1,
    sum_{i=1}^n x^i = x,
    sum_{i=1}^n u^i = u,
    sum_{i=1}^n A^i x^i + B^i u^i + m^i c^i = x+.
    Where
    x^i and u^i are auxiliary continuous variables in the MLD formulation,
    m^i are auxiliary binary variables.

    Arguments
    ----------
    pwa : instance of PieceWiseAffineSystem
        PWA system to be converted in MLD form.

    Returns
    ----------
    mld : instance of MixedLogicalDynamicalSystem
        Equivalent MLD representation of the PWA system in input.
    """

    # state transition matrices
    A = dict()
    A['cc'] = np.zeros((pwa.nx, pwa.nx))
    A['cb'] = np.zeros((pwa.nx, 0))
    A['bc'] = np.zeros((0, pwa.nx))
    A['bb'] = np.zeros((0, 0))

    # input matrices
    B = dict()
    B['cc'] = np.zeros((pwa.nx, pwa.nu))
    B['cb'] = np.zeros((pwa.nx, 0))
    B['bc'] = np.zeros((0, pwa.nu))
    B['bb'] = np.zeros((0, 0))

    # auxiliary variable matrices
    C = dict()
    C['cc'] = np.hstack([np.hstack([a.A, a.B]) for a in pwa.affine_systems])
    C['cb'] = np.hstack([a.c for a in pwa.affine_systems])
    C['bc'] = np.zeros((0, pwa.nm*(pwa.nx+pwa.nu)))
    C['bb'] = np.zeros((0, pwa.nm))

    # offset term
    b = dict()
    b['c'] = np.zeros(pwa.nx)
    b['b'] = np.zeros(0)

    # state constraint matrices
    F = dict()
    F['ec'] = np.vstack((
        np.zeros((1, pwa.nx)),      # sum_{i=1}^n m^i = 1
        - np.eye(pwa.nx),           # sum_{i=1}^n x^i - x = 0
        np.zeros((pwa.nu, pwa.nx)), # sum_{i=1}^n u^i - u = 0
        ))
    F['eb'] = np.vstack((
        np.zeros((1, 0)),      # sum_{i=1} m^i = 1
        np.zeros((pwa.nx, 0)), # sum_{i=1}^n x^i - x = 0
        np.zeros((pwa.nu, 0)), # sum_{i=1}^n u^i - u = 0
        ))
    # F['ic'] = np.zeros((0, pwa.nx))
    # F['ib'] = np.zeros((0, 0))

    # input constraint matrices
    G = dict()
    G['ec'] = np.vstack((
        np.zeros((1, pwa.nu)),      # sum_{i=1}^n m^i = 1
        np.zeros((pwa.nx, pwa.nu)), # sum_{i=1}^n x^i - x = 0
        - np.eye(pwa.nu),           # sum_{i=1}^n u^i - u = 0
        ))
    G['eb'] = np.vstack((
        np.zeros((1, 0)),      # sum_{i=1}^n m^i = 1
        np.zeros((pwa.nx, 0)), # sum_{i=1}^n x^i - x = 0
        np.zeros((pwa.nu, 0)), # sum_{i=1}^n u^i - u = 0
        ))

    # auxiliary variable constraint matrices
    H = dict()
    H['ec'] = np.vstack((
        np.zeros((1, pwa.nm*(pwa.nx+pwa.nu))),                                           # sum_{i=1}^n m^i = 1
        np.hstack([[np.eye(pwa.nx), np.zeros((pwa.nu,pwa.nu))] for i in range(pwa.nm)]), # sum_{i=1}^n x^i - x = 0
        ))
    H['ec'] = np.vstack((
        np.ones((1, pwa.nm)), # sum_{i=1}^n m^i = 1
        ))

    # right hand side of the constraints
    l = dict()
    l['e'] = np.concatenate((
        np.ones(1), # sum_{i=1}^n m^i = 1
        ))




















