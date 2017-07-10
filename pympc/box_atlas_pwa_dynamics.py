import numpy as np
from pympc.geometry.polytope import Polytope
from pympc.dynamical_systems import DTAffineSystem, DTPWASystem
import matplotlib.pyplot as plt
from itertools import combinations
from copy import copy

### Acronyms:
# q -> position, v -> velocity, f -> contact force
# b -> body, lf -> left foot, rf -> right foot, h -> hand
# x -> horizontal axis, y -> vertical axis

### State vector \in R^10:
# [q_b_x, q_b_y, q_lf_x, q_lf_y, q_rf_x, q_rf_y, q_h_x, q_h_y, v_b_x, v_b_y]

### Input vector \in R^9:
# [v_lf_x, v_lf_y, v_rf_x, v_rf_y, v_h_x, v_h_y, f_lf_x, f_rf_x, f_h_y]

### Numeric constants

# scalar
mass = 1.
stiffness = 100.
gravity = 10.
friction_coefficient = .5
t_s = .1

# position bounds

# body
q_b_min = np.array([[0.],[0.]])
q_b_max = np.array([[1.],[1.]])
v_b_max = np.ones((2,1))
v_b_min = - v_b_max

# left foot (limits in the body frame)
q_lf_min = np.array([[.1],[-.5]])
q_lf_max = np.array([[.5],[-.1]])

# right foot (limits in the body frame)
q_rf_min = np.array([[-.5],[-.5]])
q_rf_max = np.array([[-.1],[-.1]])

# hand (limits in the body frame)
q_h_min = np.array([[-.5],[.1]])
q_h_max = np.array([[-.1],[.5]])

# velocity bounds

# body
v_b_max = np.ones((2,1))
v_b_min = - v_b_max

# left foot
v_lf_max = np.ones((2,1))
v_lf_min = - v_lf_max

# right foot
v_rf_max = np.ones((2,1))
v_rf_min = - v_rf_max

# hand
v_h_max = np.ones((2,1))
v_h_min = - v_rf_max

# force bounds

# left foot
f_lf_x_max = - friction_coefficient * stiffness * (q_b_min[1,0]+q_lf_min[1,0])
f_lf_x_min = - f_lf_x_max

# right foot
f_rf_x_max = - friction_coefficient * stiffness * (q_b_min[1,0]+q_rf_min[1,0])
f_rf_x_min = - f_rf_x_max

# hand
f_h_y_max = - friction_coefficient * stiffness * (q_b_min[0,0]+q_h_min[0,0])
f_h_y_min = - f_h_y_max

### Dynamics

def get_A(contact_set):
    contact_map_normal_force = {'lf':[9,3], 'rf':[9,5], 'h':[8,6]}
    A = np.vstack((
        np.hstack((np.zeros((2, 8)), np.eye(2))),
        np.zeros((8, 10))
        ))
    for contact in contact_set:
        A[contact_map_normal_force[contact]] = -stiffness/mass
    return A

def get_B(contact_set):
    contact_map_tangential_force = {'lf':[8,6], 'rf':[8,7], 'h':[9,8]}
    contact_map_tangential_velocity = {'lf':[2,0], 'rf':[4,2], 'h':[7,5]}
    B = np.vstack((
        np.zeros((2, 9)),
        np.hstack((np.eye(6), np.zeros((6, 3)))),
        np.zeros((2, 9))
        ))
    for contact in contact_set:
        B[contact_map_tangential_force[contact]] = 1./mass
    for contact in contact_set:
        B[contact_map_tangential_velocity[contact]] = 0.
    return B

c = np.vstack((np.zeros((9, 1)), np.array((-gravity))))

contacts = ['lf', 'rf', 'h']
modes =  [mode for n_contacts in range(len(contacts)+1) for mode in combinations(contacts, n_contacts)]

### Kinematic limits

selection_matrix = np.vstack((np.eye(2), -np.eye(2)))

# left foot
lhs = np.hstack((-selection_matrix, selection_matrix, np.zeros((4,6))))
rhs = np.vstack((q_lf_max, -q_lf_min))
kinematic_limits = Polytope(lhs, rhs)

# right foot
lhs = np.hstack((-selection_matrix, np.zeros((4,2)), selection_matrix, np.zeros((4,4))))
rhs = np.vstack((q_rf_max, -q_rf_min))
kinematic_limits.add_facets(lhs, rhs)

# hand
lhs = np.hstack((-selection_matrix, np.zeros((4,4)), selection_matrix, np.zeros((4,2))))
rhs = np.vstack((q_h_max, -q_h_min))
kinematic_limits.add_facets(lhs, rhs)

# body
kinematic_limits.add_bounds(q_b_min, q_b_max, [0,1])
kinematic_limits.add_bounds(v_b_min, v_b_max, [8,9])

### Input limits

input_max = np.vstack((v_lf_max, v_rf_max, v_h_max, f_lf_x_max, f_rf_x_max, f_h_y_max))
input_min = np.vstack((v_lf_min, v_rf_min, v_h_min, f_lf_x_min, f_rf_x_min, f_h_y_min))
input_limits = Polytope.from_bounds(input_min, input_max)

### Mode indepenedent contraints

lhs = np.vstack((
    np.hstack((
        kinematic_limits.A,
        np.zeros((
            kinematic_limits.A.shape[0],
            input_limits.A.shape[1]
            ))
        )),
    np.hstack((
        np.zeros((
            input_limits.A.shape[0],
            kinematic_limits.A.shape[1]
            )),
        input_limits.A
        ))
    ))

rhs = np.vstack((kinematic_limits.b, input_limits.b))

mode_independent_constraints = Polytope(lhs, rhs)

### Mode dependent constraints

def contact_constraints(domain, contact, active):
    contact_indices = {
    'lf': {'q': 3, 'f': 16},
    'rf': {'q': 5, 'f': 17},
    'h': {'q': 6, 'f': 18}
    }
    if active:
        domain.add_upper_bounds(np.array([[0.]]), [contact_indices[contact]['q']])
        lhs = np.zeros((2,19))
        lhs[0, contact_indices[contact]['f']] = 1.
        lhs[0, contact_indices[contact]['q']] = friction_coefficient * stiffness
        lhs[1, contact_indices[contact]['f']] = -1.
        lhs[1, contact_indices[contact]['q']] = friction_coefficient * stiffness
        rhs = np.zeros((2,1))
        domain.add_facets(lhs, rhs)
    else:
        domain.add_lower_bounds(np.array([[0.]]), [contact_indices[contact]['q']])
    return domain

domains = []
for mode in modes:
    domain = copy(mode_independent_constraints)
    for contact in contacts:
        domain = contact_constraints(domain, contact, contact in mode)
    domains.append(domain)

### Translation to the equilibrium point

x_eq = np. array([
    [.5], # q_b_x
    [.5], # q_b_y
    [.3], # q_lf_x
    [-mass*gravity/2./stiffness], # q_lf_y
    [.7], # q_rf_x
    [-mass*gravity/2./stiffness], # q_rf_y
    [.2], # q_h_x
    [.6], # q_h_y
    [0.], # v_b_x
    [0.], # b_b_y
    ])
u_eq = np.zeros((9,1))

translated_affine_systems = [DTAffineSystem.from_continuous(get_A(mode), get_B(mode), c + get_A(mode).dot(x_eq), t_s) for mode in modes]
translated_domains = [Polytope(domain.A, domain.b - domain.A.dot(np.vstack((x_eq, u_eq)))) for domain in domains]

print translated_domains
# PWA system

pwa_system = DTPWASystem(translated_affine_systems, translated_domains)
