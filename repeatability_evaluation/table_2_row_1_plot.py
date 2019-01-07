'''
The following code plots the results presented in the first row of Table 2.
This calls the file table_2_row_1_data.py which takes an EXTREMELY long time (~3 days) to run.
In order to avoid to run that code, one can load the results from the file paper_data/cost_micp_different_intial_conditions.npy (to do that comment line 16 and uncomment lines 19).
'''

# external imports
import numpy as np
import matplotlib.pyplot as plt

# load parameters
import numeric_parameters as params

# To solve the optimization problems uncomment the following (approx. 8 hours)
from table_2_row_1_data import cost

# To load the results uncomment the following
# cost = np.load('paper_data/cost_micp_different_intial_conditions.npy').item()

# change default size of matplotlib
plt.rc('font', size=18)

# create grid x1 and x3
xb_samples = np.concatenate([[i]*len(cost['tb_samples']) for i in cost['xb_samples']])
tb_samples = cost['tb_samples'].tolist()*len(cost['xb_samples'])

# number of level curves in each plot
n_levels = 10

# for all the norms of the objective
norms = ['inf', 'one', 'two']
for norm in norms:
    print '\nnorm:', norm
    
    # initialize figure
    plt.figure(figsize=(6., 2.3))
    
    # scatter the samples of the optimal value function
    c = cost[norm].flatten()
    sc = plt.scatter(
        xb_samples,
        tb_samples,
        c=c,
        cmap=plt.cm.get_cmap('viridis_r')
    )
    
    # get colorbar
    cb = plt.colorbar(sc)
    
    # colorbar ticks
    cb.set_ticks([min(c), max(c)])
    cb.set_ticklabels(['%.2f'%min(c),'%.2f'%max(c)])
    
    # axis limits
    plt.xlim(-.01,.31)
    plt.ylim(-np.pi-.3,np.pi+.3)
    
    # axis ticks
    plt.xticks(np.linspace(0., params.x_max[0], 6))
    plt.yticks(np.linspace(-np.pi, np.pi, 5))
    plt.gca().axes.yaxis.set_ticklabels([r'$-\pi$',r'$-\pi/2$',0,r'$\pi/2$',r'$\pi$'])
    
    # misc
    plt.grid(True)
    plt.gca().set_axisbelow(True)
    plt.show()