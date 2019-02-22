# external imports
import numpy as np
import sympy as sp

class MixedLogicalDynamicalSystem(object):
    '''
    Discrete-time Mixed Logical Dynamical (MLD) system in the form
    x(t+1) = A x(t) + Buc uc(t) + Bub ub(t) + Bsc sc(t) + Bsb sb(t) + b
             F x(t) + Guc uc(t) + Gub ub(t) + Gsc sc(t) + Gsb sb(t) <= g
    where:
    - x(t) continuous state at time t
    - uc(t) continuous input at time t
    - ub(t) binary input at time t
    - sc(t) continuous auxiliary variables at time t
    - sb(t) binary auxiliary variables at time t
    - A, F, Buc, Guc, Bub, Gub, Bsc, Gsc, Bsb, Gsb matrices of appropriate size
    - b, g vectors of appropriate size
    '''

    def __init__(self, A, B, b, F, G, g):
        '''
        Initializes the MLD system.

        Arguments
        ---------
        A : np.array
            State transition matrix.
        B : dict with entries 'uc', 'ub', 'sc', 'sb'
            Collects the matrices Buc, Bub, Bsc, Bsb in the linear dynamics.
        b : np.array
            Offset term in the dynamics.
        F : np.array
            Matrix of the state coefficients in the constraints.
        G : dict with entries 'uc', 'ub', 'sc', 'sb'
            Collects the matrices Guc, Gub, Gsc, Gsb in the linear constraints.
        g : np.array
            Offset term in the constraints.
        '''

        # store the data
        self.A = A
        self.b = b
        self.F = F
        self.g = g
        [self.Buc, self.Bub, self.Bsc, self.Bsb] = [B['uc'], B['ub'], B['sc'], B['sb']]
        [self.Guc, self.Gub, self.Gsc, self.Gsb] = [G['uc'], G['ub'], G['sc'], G['sb']]
        
        # store sizes of the system
        self.nx = A.shape[0]
        self.nuc = B['uc'].shape[1]
        self.nub = B['ub'].shape[1]
        self.nsc = B['sc'].shape[1]
        self.nsb = B['sb'].shape[1]
        self.m = F.shape[0]

        # store also stacked matrices for convenience
        self.Bu = np.hstack((B['uc'], B['ub']))
        self.Bs = np.hstack((B['sc'], B['sb']))
        self.Gu = np.hstack((G['uc'], G['ub']))
        self.Gs = np.hstack((G['sc'], G['sb']))
        self.Wu = np.hstack((np.zeros((self.nub,self.nuc)), np.eye(self.nub)))
        self.Ws = np.hstack((np.zeros((self.nsb,self.nsc)), np.eye(self.nsb)))

        # check size of the input matrices
        self._check_input_sizes()

    def _check_input_sizes(self):
        '''
        Checks the size of the matrices passed as inputs in the initialization of the class.
        '''

        # check dynamics
        assert self.A.shape[0] == self.A.shape[1]
        assert self.Buc.shape[0] == self.nx
        assert self.Bub.shape[0] == self.nx
        assert self.Bsc.shape[0] == self.nx
        assert self.Bsb.shape[0] == self.nx
        assert self.b.size == self.nx

        # check constraints
        assert self.F.shape[1] == self.nx
        assert self.Guc.shape == (self.m, self.nuc)
        assert self.Gub.shape == (self.m, self.nub)
        assert self.Gsc.shape == (self.m, self.nsc)
        assert self.Gsb.shape == (self.m, self.nsb)
        assert self.g.size == self.m

    @staticmethod
    def from_symbolic(variables, x_next, constraints):
        '''
        Instatiates a MixedLogicalDynamicalSystem starting from the symbolic value of the dynamics and the constraints.

        Arguments
        ---------
        variables : dict of sympy matrix filled with sympy symbols
            Symbolic variables of the system, entries: 'x', 'uc', 'ub', 'sc', 'sb'.
        x_next : sympy matrix of sympy symbolic linear expressions
            Symbolic value of the next state (affine function of x, uc, ub, sc, sb).
        constraints : sympy matrix of sympy symbolic linear expressions
            Symbolic constraints in the form constraints <= 0.
            Hence, constraints = F x + Guc uc + Gub ub + Gsc sc + Gsb sb - g.

        Returns
        -------
        mld : instance of MixedLogicalDynamicalSystem
            MLD system extracted from the symbolic expressions.
        '''

        # collect variables
        variable_labels = ['x','uc','ub','sc','sb']
        v = sp.Matrix([variables[l] for l in variable_labels])
        blocks = [variables[l].shape[0] for l in variable_labels]

        # state transition matrices
        J, b = get_matrices_affine_expression(v, x_next)
        A, Buc, Bub, Bsc, Bsb = unpack_block_matrix(J, blocks, 'h')
        B = {'uc':Buc, 'ub':Bub, 'sc':Bsc, 'sb':Bsb}

        # constraints
        J, d = get_matrices_affine_expression(v, constraints)
        F, Guc, Gub, Gsc, Gsb = unpack_block_matrix(J, blocks, 'h')
        G = {'uc':Guc, 'ub':Gub, 'sc':Gsc, 'sb':Gsb}
        g = - d

        # construct MLD system
        mld = MixedLogicalDynamicalSystem(A, B, b, F, G, g)

        return mld

def get_matrices_affine_expression(x, expr):
    '''
    Extracts from the symbolic affine expression the matrices such that expr(x) = A x + b.

    Arguments
    ---------
    x : sympy matrix of sympy symbols
        Variables of the affine expression.
    expr : sympy matrix of sympy symbolic affine expressions
        Left hand side of the inequality constraint.

    Returns
    -------
    A : np.array
        Jacobian of the affine expression.
    b : np.array
        Offset term of the affine expression.
    '''

    # state transition matrices
    A = np.array(expr.jacobian(x)).astype(np.float64)

    # offset term
    b = np.array(expr.subs({xi:0 for xi in x})).astype(np.float64).flatten()
    
    return A, b

def unpack_block_matrix(A, indices, direction):
    '''
    Unpacks a matrix in blocks.

    Arguments
    ---------
    A : np.array
        Matrix to be unpacked.
    indices : list of int
        Set of indices at which the matrix has to be cut.
    direction : string
        'h' to unpack horizontally and 'v' to unpack vertically.

    Returns
    -------
    blocks : list of np.array
        Blocks extracted from the matrix A.
    '''

    # initialize blocks
    blocks = []

    # unpack
    i = 0
    for j in indices:

        # horizontally
        if direction == 'h':
            blocks.append(A[:,i:i+j])

        # vertically
        elif direction == 'v':
            blocks.append(A[i:i+j,:])

        # raise error if uknown key
        else:
            raise ValueError('unknown direction ' + direction)

        # increase index by j
        i += j

    return blocks