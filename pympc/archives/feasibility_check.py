import numpy as np
from pympc.optimization.gurobi import linear_program
from pympc.geometry.chebyshev_center import chebyshev_center

def check_feasibility_the_dumb_way(A, B, c, x):
    lhs = B
    rhs = c - A.dot(x)
    cost = np.zeros((B.shape[1], 1))
    sol = linear_program(cost, lhs, rhs)
    if np.isnan(sol.min):
        return False
    return True

def check_feasibility(A, B, c, x, x_internal, u_internal, tol=1.e-5):
    # step 0
    lhs = A.dot(x - x_internal)
    rhs = c - A.dot(x_internal) - B.dot(u_internal)
    cost = -np.ones((1,1))
    sol = linear_program(cost, lhs, rhs)
    alpha = - sol.min
    # step 1
    free_u = []
    new_u = 0
    while alpha < 1. - tol:
        print 'alpha:', alpha
        if alpha < tol or len(free_u) == B.shape[1]:
            return False
        x_internal = alpha*(x - x_internal) + x_internal

        ###
        contact_force = B[sol.active_set,:].T.dot(sol.inequality_multipliers[sol.active_set,:])
        biggest_components = np.argsort(np.absolute(contact_force).flatten())
        for ind in biggest_components:
        	if ind not in free_u:
        		free_u.append(ind)
        		break
        ###

        #free_u.append(new_u)
        fixed_u = [i for i in range(B.shape[1]) if i not in free_u]
        lhs = np.hstack((
            A.dot(x - x_internal),
            B[:,free_u]
        ))
        rhs = c - A.dot(x_internal) - B[:,fixed_u].dot(u_internal[fixed_u,:])
        cost = np.vstack((cost, np.zeros((1,1))))
        sol = linear_program(cost, lhs, rhs)
        alpha = - sol.min
        new_u += 1
    return True