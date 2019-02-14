# external imports
import numpy as np
from copy import copy

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