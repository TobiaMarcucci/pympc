import numpy as np
import matplotlib.pyplot as plt
from mpc_tools.geometry import Polytope
from mpc_tools.optimization import linear_program, linear_program_pnnls
import time

print 'Test for the (home-made 20-lines-of-code) PNNLS solver for linear programs.\n'

f = np.array([[0.], [0.]])
A = np.array([[  9.73537770e-01,   2.28526169e-01],
       [ -6.95497931e-01,  -7.18528098e-01],
       [  9.37562537e-01,   3.47816748e-01],
       [  9.62584453e-01,   2.70981865e-01],
       [  9.66326722e-01,   2.57318219e-01],
       [ -9.62584453e-01,  -2.70981865e-01]])
b = np.array([[-0.30239749],
       [ 0.48801894],
       [-0.31104601],
       [-0.31763297],
       [-0.31338212],
       [ 0.32430017]])

print 'Consider the linear program\n\n', 'min f^T x \n', 's.t. A x <= b\n', '\nwith\n', '\nf^T = ', f.flatten(), '\nA = ', A, '\nb^T =', b.flatten(), '\n'


print 'Gurobi (with minimum admissible OptimalityTol) says it is infeasible\n\n', '--- GUROBI OUTPUT START'
linear_program(f, A, b, solver='gurobi', OutputFlag=1, OptimalityTol=1.e-9)
print '--- GUROBI OUTPUT END\n'
x_inside = np.array([[-.279],[-.194]])
print 'but the point', x_inside.flatten(), 'is feasible, in fact its residual is Ax - b =', (A.dot(x_inside)-b).flatten(), '< 0.\n'

tic = time.clock()
x_inside = linear_program(f, A, b, solver='mosek')[0]
toc = time.clock()
print 'Mosek gives the feasible point', x_inside.flatten(), 'in', toc-tic, 'seconds.\n'

tic = time.clock()
x_inside = linear_program_pnnls(f, A, b)[0]
toc = time.clock()
print 'The PNNLS solver gives the feasible point', x_inside.flatten(), 'in', toc-tic, 'seconds.\n'

f = f = np.ones((2,1))
print '\nChanging the cost function to\n', 'f^T = ', f.flatten(), '\nwe get:\n'
tic = time.clock()
x_star, f_star = linear_program(f, A, b, solver='gurobi')
toc = time.clock()
print '- Gurobi (with default precision) says x^* =', x_star.flatten(), ', f^* =', f_star.flatten(), 'in', toc-tic, 'seconds;'
tic = time.clock()
x_star, f_star = linear_program(f, A, b, solver='mosek')
toc = time.clock()
print '- Mosek says x^* =', x_star.flatten(), ', f^* =', f_star.flatten(), 'in', toc-tic, 'seconds;'
tic = time.clock()
x_star, f_star = linear_program_pnnls(f, A, b)
toc = time.clock()
print '- PNNLS says x^* =', x_star.flatten(), ', f^* =', f_star.flatten(), 'in', toc-tic, 'seconds.'



