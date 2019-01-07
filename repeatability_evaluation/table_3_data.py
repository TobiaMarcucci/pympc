'''
The following code computes the data reported in Table 3 in the paper.
It solves one MICP for each norm and each mixed-integer formulation under analysis.
In total there are 12 MICPs, 7 of which require more than 1 hour to be solved.
The time limit is set to 1 hour, hence the code takes ~8 hours to run.
The results of these optimization problems are the printed running the file table_3_print_results.py.
For convenience, the results reported in the paper are also saved in the paper_data folder, they can be loaded running:
``solves = np.load('paper_data/solves_flip_benchmark.npy').item()``.
(An option to check the results more quickly is to reduce the value of the time_limit (line 40): depending on the computer, 500 seconds should be sufficient to reproduce most of the results.)
'''

# external imports
import numpy as np
import matplotlib.pyplot as plt

# internal imports
from pympc.control.hscc.controllers import HybridModelPredictiveController
import numeric_parameters as params
from pwa_dynamics import S

# initial condition
x0 = np.array([
    0., 0., np.pi,
    0., 0.,
    0., 0., 0.,
    0., 0.
])

# solves of the MICP with all the methods and the norms
solves = {}

# gurobi parameters
gurobi_options = {'OutputFlag': 1} # set OutputFlag to 0 to turn off gurobi log
time_limit = 3600

# for all the norms of the objective
norms = ['inf', 'one', 'two']
methods = ['pf', 'ch', 'bm', 'mld']
for norm in norms:
    
    solves[norm] = {}
    
    # for all the mixed-integer formulations
    for method in methods:
        print '\n-> norm:', norm
        print '-> method:', method, '\n'
        
        # build the copntroller
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
        
        # kill solution if longer than 1h
        controller.prog.setParam('TimeLimit', time_limit)
        
        # solve and store result
        u_mip, x_mip, ms_mip, cost_mip = controller.feedforward(x0, gurobi_options)
        solves[norm][method] = {
            'time': controller.prog.Runtime,
            'nodes': controller.prog.NodeCount,
            'mip_gap': controller.prog.MIPGap,
            'u': u_mip,
            'x': x_mip,
            'ms': ms_mip,
            'cost': cost_mip
        }