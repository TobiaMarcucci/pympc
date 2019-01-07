'''
The following code prints the results reported in Table 3.
This calls the file table_3_data.py which takes ~8 hours to run.
In order to avoid to run that code, one can load the results from the file paper_data/solves_flip_benchmark.npy (to do that comment line 9 and uncomment lines 12-13).
Alternatively one can consider to reduce the time limit_parameter as explained in table_3_data.py.
'''

# To solve the optimization problems uncomment the following (approx. 8 hours)
# from table_3_data import solves

# To load the results uncomment the following
# import numpy as np
# solves = np.load('paper_data/solves_flip_benchmark.npy').item()

norms = ['inf', 'one', 'two']
methods = ['pf', 'ch', 'bm', 'mld']
for norm in norms:
    for method in methods:
        print '\n-> norm:', norm
        print '-> method:', method
        print 'mip gap:', solves[norm][method]['mip_gap']
        print 'time:', solves[norm][method]['time']
        print 'nodes:', solves[norm][method]['nodes']