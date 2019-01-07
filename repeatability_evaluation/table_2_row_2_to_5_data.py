'''
The following code computes the data reported in Table 2 from row 2 to row 5.
For all the norms and the mixed-iteger formulations, this code samples on a 51-by-51 grid the optimal value of the problem relaxation.
Running this code takes ~10 minutes.
Alternatively, the results of the samples can be loaded running:
``cost = np.load('paper_data/cost_relaxation_different_intial_conditions.npy').item()``.
An option to check the results more quickly is to make the grid more coarse, modifying the parameter n_samples in line 19.
'''

# external imports
import numpy as np

# internal imports
from pympc.control.hscc.controllers import HybridModelPredictiveController
import numeric_parameters as params
from pwa_dynamics import S

# n_samples by n_samples grid in x1 and x3
n_samples = 51
xb_samples = np.linspace(0., params.x_max[0], n_samples)
tb_samples = np.linspace(-np.pi, np.pi, n_samples)

# cost of the relaxation for each point
cost = {'xb_samples': xb_samples, 'tb_samples':tb_samples}
gurobi_options = {'OutputFlag': 0}

# for all the norms of the objective
norms = ['inf', 'one', 'two']
for norm in norms:
    cost[norm] = {}
    
    # for all the mixed-integer formulations
    methods = ['pf', 'ch', 'bm', 'mld']
    for method in methods:
        print '\n-> norm:', norm
        print '-> method:', method
        print 'Solving convex programs on a ' + str(n_samples) + ' by ' + str(n_samples) + ' grid:'
        
        # build controller
        controller = HybridModelPredictiveController(
            S,
            params.N,
            params.Q,
            params.R,
            params.P,
            params.X_N,
            method,
            norm
        )
        
        # cost matrix
        cost_mat = np.empty([n_samples]*2)
        
        # solve one convex program per point in the grid
        for i, xb in enumerate(xb_samples):
            for j, tb in enumerate(tb_samples):
                print(str(i) + ',' + str(j) + '   \r'),
                
                # initial state
                x0 = np.array([xb,0.,tb] + [0.]*7)
                
                # cost of the relaxation
                cost_mat[i,j] = controller.solve_relaxation(x0, {}, gurobi_options)[3]
                
        # fill matrix
        cost[norm][method] = cost_mat