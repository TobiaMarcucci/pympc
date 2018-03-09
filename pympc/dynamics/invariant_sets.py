# external imports
import numpy as np

# pympc imports
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.mathematical_programs import LinearProgram

def mcais(A, X, tol=1.e-9):
    """
    Returns the maximal constraint-admissible (positive) ivariant set O_inf for the system x(t+1) = A x(t) subject to the constraint x in X.
    O_inf is also known as maximum output admissible set.
    It holds that x(0) in O_inf <=> x(t) in X for all t > 0.
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

# def mcais_closed_loop(A, B, K, D):
#     # closed loop dynamics
#     A_cl = A + B.dot(K)
#     # constraints for the maximum output admissible set
#     lhs_cl = D.A[:,:A.shape[0]] + D.A[:,A.shape[0]:].dot(K)
#     rhs_cl = D.b
#     X_cl = Polytope(lhs_cl, rhs_cl)
#     X_cl.assemble()
#     # compute maximum output admissible set
#     return mcais(A_cl, X_cl)

# def mcais_closed_loop_orthogonal_domains(A, B, K, X, U):
#     lhs = linalg.block_diag(X.A, U.lhs_min)
#     rhs = np.vstack((X.b, U.b))
#     D = Polytope(lhs, rhs)
#     D.assemble()
#     return mcais_closed_loop(A, B, K, D)
