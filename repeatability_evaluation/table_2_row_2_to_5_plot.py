'''
The following code plots the results presented in Table 2 from row 2 to row 5.
This calls the file table_2_row_2_to_5_data.py which takes ~10 minutes to run.
Alternatively, one can load the results from the file paper_data/cost_relaxation_different_intial_conditions.npy (to do that comment line 13 and uncomment lines 16).

'''

# external imports
import numpy as np
import matplotlib.pyplot as plt

# To solve the optimization problems uncomment the following (approx. 8 hours)
from table_2_row_2_to_5_data import cost

# To load the results uncomment the following
# cost = np.load('paper_data/cost_relaxation_different_intial_conditions.npy').item()

# load parameters
import numeric_parameters as params

# change default size of matplotlib
plt.rc('font', size=18)

# create grid x1 and x3
xb_samples = cost['xb_samples']
tb_samples = cost['tb_samples']
Xb, Tb = np.meshgrid(xb_samples, tb_samples)

# number of level curves in each plot
n_levels = 10

# for all the norms of the objective
norms = ['inf', 'one', 'two']
for norm in norms:
    
    # for all the mixed-integer formulations
    methods = ['pf', 'ch', 'bm', 'mld']
    for method in methods:
        print '\nnorm: %s, method: %s' % (norm, method)
        
        # initialize figure
        plt.figure(figsize=(6., 2.3))
        
        # cost matrix
        cm = cost[norm][method]

        # set desired levels and draw the contour plot
        levels = [(i+1)*np.nanmax(cm)/n_levels for i in range(n_levels)]
        cp = plt.contour(Xb, Tb, cm.T, levels=levels, cmap='viridis_r')
        
        # get colorbar
        cb = plt.colorbar(cp)
        
        # ticks of the colorbar (only first and last) limited to 2 decimals
        cb.set_ticks([cb.locator()[0],cb.locator()[-1]])
        cb.set_ticklabels(['%.2f'%cb.locator()[0],'%.2f'%cb.locator()[-1]])
        
        # axis ticks
        plt.xticks(np.linspace(0., params.x_max[0], 6))
        plt.yticks(np.linspace(-np.pi, np.pi, 5))
        plt.gca().axes.yaxis.set_ticklabels([r'$-\pi$',r'$-\pi/2$',0,r'$\pi/2$',r'$\pi$'])
        
        # misc
        plt.grid(True)
        plt.show()