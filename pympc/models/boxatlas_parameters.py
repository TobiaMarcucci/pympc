import numpy as np

dynamics = {
        'mass': 1.,
        'moment_of_inertia': 1.,
        'normal_stiffness': 100.,
        'tangential_damping': 100.,
        'friction': .5,
        'gravity': 10.,
        'sampling_time': .1,
}

joint_limits = {
        'b': {'min': np.array([[-.2],[-.1],[-.1]]), 'max': np.array([[.2],[.1],[.1]])}, # x,y,theta
        'lf': {'min': np.array([[.0],[-.7]]), 'max': np.array([[.4],[-.3]])}, # wrt body
        'rf': {'min': np.array([[-.4],[-.7]]), 'max': np.array([[0.],[-.3]])}, # wrt body
        'lh': {'min': np.array([[.2],[-.2]]), 'max': np.array([[.4],[.2]])}, # wrt body
        'rh': {'min': np.array([[-.4],[-.2]]), 'max': np.array([[-.2],[.2]])}, # wrt body
        }

velocity_limits = {
        'b': {'min': np.array([[-2.],[-2.],[-.5]]), 'max': np.array([[2.],[2.],[.5]])}, # x,y,theta
        'lf': {'min': -np.ones((2,1)), 'max': np.ones((2,1))},
        'rf': {'min': -np.ones((2,1)), 'max': np.ones((2,1))},
        'lh': {'min': -2*np.ones((2,1)), 'max': 2*np.ones((2,1))},
        'rh': {'min': -2*np.ones((2,1)), 'max': 2*np.ones((2,1))},
        }

weight = dynamics['mass'] * dynamics['gravity']
f_min = np.array([[0.], [-dynamics['friction'] * weight]])
f_max = np.array([[weight], [dynamics['friction'] * weight]])
force_limits = {
        'lf': {'min': f_min*2., 'max': f_max*2.},
        'rf': {'min': f_min*2., 'max': f_max*2.},
        'lh': {'min': f_min, 'max': f_max},
        'rh': {'min': f_min, 'max': f_max},
        }

controller ={
        'horizon': 10,
        'objective_norm': 'two',
        'state_cost': {
                'qb': 1.,
                'tb': 1.,
                'vb': 1.,
                'ob': 1.,
                'qlf_rel': 1.,
                'qrf_rel': 1.,
                'qlh_rel': .2,
                'qrh_rel': .2,
                },
        'input_cost': {
                'vlf': 1.,
                'vrf': 1.,
                'vlh': .2,
                'vrh': .2,
                'flf': 1.,
                'frf': 1.,
                'flh': 1.,
                'frh': 1.,
                },
        'gurobi': {
                'OutputFlag': 0,
                'TimeLimit': 600.,
                'MIPFocus': 0,           # balanced: 0, feasibility: 1, optimality: 2, bounds: 3
                'NumericFocus': 0,       # min:     0, def:     0, max:     3
                'OptimalityTol': 1.e-6,  # min: 1.e-9, def: 1.e-6, max: 1.e-2 
                'FeasibilityTol': 1.e-6, # min: 1.e-9, def: 1.e-6, max: 1.e-2 
                'IntFeasTol': 1.e-5,     # min: 1.e-9, def: 1.e-5, max: 1.e-1 
                'MIPGap': 1.e-4,         # min:    0., def: 1.e-4, max:   inf
        }
}

visualizer = {
        'min': np.array([[-.6], [-.6]]),
        'max': np.array([[.6], [.4]]),
        'depth': [-.3,.3],
        'box_fixed_feet': {'tickness': .05, 'width': .1},
        'body_size': .2,
        'limbs_size': .05,
        'body_color': np.hstack((np.array([0.,0.,1.]), 1.)),
        'limbs_color': np.hstack((np.array([1.,0.,0.]), 1.)),
}