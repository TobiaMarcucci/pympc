import numpy as np

mass = 1.
moment_of_inertia = 1.
stiffness = 200.
damping = 50.
friction = .5
gravity = 10.
sampling_time = .1
# integrator = 'explicit_euler'
weight = mass * gravity
f_min = np.array([[0.], [-friction * weight]])
f_max = np.array([[weight], [friction * weight]])
visualizer_min = np.array([[-.6],[-.6]])
visualizer_max = np.array([[.6],[.4]])

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

force_limits = {
        'lf': {'min': f_min*2., 'max': f_max*2.},
        'rf': {'min': f_min*2., 'max': f_max*2.},
        'lh': {'min': f_min, 'max': f_max},
        'rh': {'min': f_min, 'max': f_max},
        }

state_cost = {
        'qb': 1.,
        'tb': 1.,
        'vb': 1.,
        'ob': 1.,
        'qlf_rel': 1.,
        'qrf_rel': 1.,
        'qlh_rel': .2,
        'qrh_rel': .2,
        }
input_cost = {
        'vlf': 1.,
        'vrf': 1.,
        'vlh': .2,
        'vrh': .2,
        'flf': 1.,
        'frf': 1.,
        'flh': 1.,
        'frh': 1.,
        }