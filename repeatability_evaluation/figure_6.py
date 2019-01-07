'''
The following code plots Figure 6.
This represents the ratio between the objective of the convex relaxations and the objective of the mixed-integer program in case of different objectives and formulations.
This leads to the solution of N=20 convex programs per objective norm and mixed-integer formulation.
Running this code takes ~1 minutes.
This analysis requires the knowledge of the optimal mode sequence, this is loaded from the file paper_data/solves_flip_benchmark.npy (see line 20).
To check the validity of the data stored in this file, please run table_3_print_results.py.
'''

# external imports
import numpy as np
import matplotlib.pyplot as plt

# internal imports
from pympc.control.hscc.controllers import HybridModelPredictiveController
import numeric_parameters as params
from pwa_dynamics import S

# load results
solves = np.load('paper_data/solves_flip_benchmark.npy').item()

# cost of each relaxation as a function of the time step (takes approx. 2 minutes)
costs = {}
gurobi_options = {'OutputFlag': 0}

# initial condition
x0 = np.array([
    0., 0., np.pi,
    0., 0.,
    0., 0., 0.,
    0., 0.
])

# for all the norms of the objective
norms = ['inf', 'one', 'two']
methods = ['pf', 'ch', 'bm', 'mld']
for norm in norms:
    costs[norm] = {}
    
    # for all the mixed-integer formulations
    for method in methods:
        print '\n-> norm:', norm
        print '-> method:', method
        costs[norm][method] = []
        
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
        
        # fix the mode of the system to its optimal value for the initial t steps
        for ms in [solves[norm]['ch']['ms'][:t] for t in range(params.N+1)]:
            
            # solve the relaxation and normalize on the optimal value of the MICP
            cost = controller.solve_relaxation(x0, ms, gurobi_options)[3]
            if cost is not None:
                cost /= solves[norm]['ch']['cost']
            costs[norm][method].append(cost)

# change default colors and size of matplotlib
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
plt.rc('font', size=14)

# for all the norms of the objective
for norm in norms:
    
    # set line colors and styles
    colors = ['b', 'r', 'c','g']
    linestyles = ['-', '-.', '--', ':']
    
     # for all the mixed-integer formulations plot the relaxation ratio as a funtion of the time step
    for i, method in enumerate(methods):
        plt.plot(
            range(params.N+1),
            costs[norm][method],
            label=method.upper(),
            color=colors[i],
            linestyle=linestyles[i],
            linewidth=3
        )
        
    # axis limits
    plt.xlim((0, params.N))
    plt.ylim((0, 1.1))
    
    # plot titles
    if norm == 'inf':
        plt.title(r'$\infty$-norm objective')
    elif norm == 'one':
        plt.title(r'1-norm objective')
    elif norm == 'two':
        plt.title(r'Quadratic objective')
        
    # ticks
    plt.xticks(range(params.N+1)) # a tick per time step
    plt.gca().axes.xaxis.set_ticklabels([i if i%2==0 else '' for i in range(params.N+1)]) # label only even ticks
    
    # axis labels
    plt.xlabel(r'Time step $t$')
    plt.ylabel(r'Cost relaxed problem / cost MICP')

    # misc
    plt.legend(loc=4)
    plt.grid(True)
    plt.show()