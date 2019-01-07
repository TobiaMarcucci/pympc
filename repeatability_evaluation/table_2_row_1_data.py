'''
The following code computes the data reported in the first row of Table 2.
We plot the optimal value function as a function of the initial state (only the horizontal position of the ball x1 and the angle of the ball x3 are considered).
For all the norms we sample the optimal value function of the MICP on a 11 by 11 grid.
This takes an EXTREMELY long time: ~3 days.
For convenience, the results reported in the paper are also saved in the paper_data folder, they can be loaded running:
``cost = np.load('paper_data/cost_micp_different_intial_conditions.npy').item()``.
An option to check the results more quickly is to reduce the value of the time_limit (line 35) and turning on the gurobi log (set to 1 the OutputFlag parameter in line 34).
Doing that one can verify that the samples reported in the paper lie between the upper and the power bound that the solver provides as the branch and bound algorithm converges.
'''

# external imports
import numpy as np
import matplotlib.pyplot as plt

# internal imports
from pympc.control.hscc.controllers import HybridModelPredictiveController
import numeric_parameters as params
from pwa_dynamics import S

# mixed-integer formulations
methods = ['pf', 'ch', 'bm', 'mld']

# norms of the objective
norms = ['inf', 'one', 'two']

# n_samples by n_samples grid in x1 and x3
n_samples = 11
xb_samples = np.linspace(0., params.x_max[0], n_samples)
tb_samples = np.linspace(-np.pi, np.pi, n_samples)

# cost of the MICP for each point
cost = {'xb_samples': xb_samples, 'tb_samples':tb_samples}
gurobi_options = {'OutputFlag': 0}
time_limit = 18000

# for all the norms of the objective
for norm in norms:
    print '\nnorm:', norm
    
    # build controller
    controller = HybridModelPredictiveController(
            S,
            params.N,
            params.Q,
            params.R,
            params.P,
            params.X_N,
            'ch',
            norm
        )
    
    # cut after 5 h
    controller.prog.setParam('TimeLimit', time_limit)
    
    # cost matrix
    cost_mat = np.empty([n_samples]*2)
        
    # solve one MICP program per point in the grid
    for i, xb in enumerate(xb_samples):
        for j, tb in enumerate(tb_samples):
            print('-> Sample: ' + str(i) + ', ' + str(j) + '   \r'),

            # initial state
            x0 = np.array([xb,0.,tb] + [0.]*7)
            
            # cost of the MICP
            cost_mat[i,j] = controller.feedforward(x0, gurobi_options)[3]
            
    # fill matrix
    cost[norm] = cost_mat