
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

def moas_closed_loop_from_orthogonal_domains(A, B, K, X, U):
    lhs = linalg.block_diag(X.lhs_min, U.lhs_min)
    rhs = np.vstack((X.rhs_min, U.rhs_min))
    D = Polytope(lhs, rhs)
    D.assemble()
    return moas_closed_loop(A, B, K, D)

def moas(A, X, tol=1.e-9):
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
    if eig_max > 1.:
        raise ValueError('Cannot compute MOAS for unstable systems')

    # Gilber and Tan algorithm
    print('Computation of Maximal Invariant Constraint-Admissible Set started.')
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
        print 'Determinedness index: ' + str(t) + ', Convergence index: ' + str(np.max(J_sol)) + ', Number of facets: ' + str(cons_lhs.shape[0]) + '.                \r',
        if np.max(J_sol) < tol:
            convergence = True
        else:
            t += 1

    # define polytope
    print '\nMaximal Invariant Constraint-Admissible Set found.'
    print 'Removing redundant facets ...',
    moas = Polytope(cons_lhs, cons_rhs)
    moas.assemble()
    print 'minimal facets are ' + str(moas.lhs_min.shape[0]) + '.'

    return moas