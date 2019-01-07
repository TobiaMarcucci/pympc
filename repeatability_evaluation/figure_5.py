'''
The following code plots Figure 5.
This represents the optimal mode sequence for the case with infinity norm objective function.
Note that the optimal mode sequence is loaded from the file paper_data/solves_flip_benchmark.npy (see line 17).
To check the validity of the data stored in this file, please run table_3_print_results.py.
'''

# external imports
import numpy as np
import matplotlib.pyplot as plt

# internal imports
import numeric_parameters as params
from pwa_dynamics import S

# load results
solves = np.load('paper_data/solves_flip_benchmark.npy').item()

# optimal mode sequence, e.g., in the infinity norm case
ms = solves['inf']['ch']['ms']

# double the initial mode for plot purposes
ms = [ms[0]] + ms

# names of the modes
ms_legend = [
    '',
    'Slide left on paddle',
    'Stick/roll on paddle',
    'Slide right on paddle',
    'No contact',
    'Slide left on ceiling',
    'Stick/roll on ceiling',
    'Slide right on ceiling'
]

# reorder the mode sequence in such a way that they match with the order of the names
ms_map = {0: 3, 1: 1, 2: 2, 3: 0, 4: 5, 5: 6, 6: 4}
ms_reordered = [ms_map[m] for m in ms]

# change default size of matplotlib
plt.rc('font', size=16)

# initialize figure
plt.figure(figsize=(6,2.5))

# plot mode sequence
plt.step(range(params.N+1), ms_reordered, color='b')

# axis limits
plt.xlim(0, params.N)
plt.ylim(-.2, S.nm-.8)

# axis ticks
plt.xticks(range(params.N+1))
plt.gca().axes.xaxis.set_ticklabels([i if i%2==0 else '' for i in range(params.N+1)])
plt.yticks(range(-1,S.nm))
plt.gca().set_yticklabels(ms_legend)

# axis labels
plt.xlabel(r'Time step $t$')
plt.ylabel(r'System mode $i^*(t)$')

# misc
plt.grid(True)
plt.show()