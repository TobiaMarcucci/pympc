import numpy as np
import scipy.linalg as linalg
from itertools import product
from copy import copy
from ad import adnumber, jacobian
from collections import OrderedDict
from pympc.geometry.polytope import Polytope
from pympc.dynamical_systems import DTAffineSystem, DTPWASystem
from pympc.control import MPCHybridController
from pympc.optimization.pnnls import linear_program
from pympc.models.box_atlas_visualizer import BoxAtlasVisualizer

class MovingLimb():
    def __init__(self, A_domains, b_domains, contact_surfaces, forbidden_transitions=[]):

        # copy inputs
        self.modes = A_domains.keys()
        self.n_domains = len(A_domains)
        self.A_domains = {}
        self.b_domains = {}
        self.contact_surfaces = contact_surfaces
        self.forbidden_transitions = forbidden_transitions

        # normalize all the A and all the b
        for mode in A_domains.keys():
            norm_factor = np.linalg.norm(A_domains[mode], axis=1)
            self.A_domains[mode] = np.divide(A_domains[mode].T, norm_factor).T
            self.b_domains[mode] = np.divide(b_domains[mode].T, norm_factor).T

        return

class FixedLimb():
    def __init__(self, position, normal):
        self.position = position
        self.normal = normal/np.linalg.norm(normal)
        return

class Trajectory:
    
    def __init__(self, x, u, Q, R, P):
        self.x = x
        self.u = u
        self.Q = Q
        self.R = R
        self.P = P
        self.cost = self._cost()
        return
        
    def _cost(self):
        cost = 0
        for u in self.u:
            cost += u.T.dot(self.R).dot(u)
        for x in self.x[:-1]:
            cost += x.T.dot(self.Q).dot(x)
        cost += self.x[-1].T.dot(self.P).dot(self.x[-1])
        return cost

class BoxAtlas():

    def __init__(
        self,
        limbs,
        parameters,
        equilibrium_configuration,
        equilibrium_limb_forces,
        joint_limits,
        velocity_limits,
        force_limits
        ):
        self.limbs = limbs
        self.parameters = parameters
        self.equilibrium_configuration = equilibrium_configuration
        self.equilibrium_limb_forces = equilibrium_limb_forces
        self.joint_limits = joint_limits
        self.velocity_limits = velocity_limits
        self.force_limits = force_limits
        self.x_eq = self._get_equilibrium_state()
        self.u_eq = self._get_equilibrium_input()
        self.n_x = self.x_eq.shape[0]
        self.n_u = self.u_eq.shape[0]
        self.x_diff = self._differentiable_state()
        self.u_diff = self._differentiable_input()
        self.contact_modes = self._get_contact_modes()
        domains = self._get_domains()
        self.contact_modes, domains = self._remove_empty_domains(domains)
        affine_systems = self._get_affine_systems()
        self.pwa_system = DTPWASystem(affine_systems, domains)
        self._visualizer = self._set_visualizer()
        return

    def _get_equilibrium_state(self):
        """
        Translates the equilibrium_configuration dictionary in the equilibrium state vector x_equilibrium.
        """

        # moving limbs position
        x_equilibrium = np.zeros((0,1))
        for limb in self.limbs['moving'].keys():
            x_equilibrium = np.vstack((x_equilibrium, self.equilibrium_configuration[limb]))

        # body state
        x_equilibrium = np.vstack((
            x_equilibrium, # limbs
            self.equilibrium_configuration['b'], # body position
            np.zeros((2,1)) # body velocity
            ))

        return x_equilibrium

    def _get_equilibrium_input(self):
        """
        Translates the equilibrium_limb_forces dictionary in the equilibrium input vector u_equilibrium.
        """

        # velocities of the moving limbs
        u_equilibrium = np.zeros((len(self.limbs['moving'])*2, 1))

        # forces of the fixed limbs
        for limb in self.limbs['fixed'].keys():
            u_equilibrium = np.vstack((u_equilibrium, self.equilibrium_limb_forces[limb]))

        return u_equilibrium

    def _get_contact_modes(self):
        """
        Returns a list of dictionaries. Each dictionary represents a mode of box-atlas: the keys of the dict are the names of the moving limbs, the values of the dict are the names given to the modes of the limbs, e.g.:
        contact_modes = [
        {'left_hand': 'not_in_contact', 'right_hand': 'not_in_contact'},
        {'left_hand': 'in_contact', 'right_hand': 'not_in_contact'},
        {'left_hand': 'not_in_contact_side', 'right_hand': 'in_contact'}
        ]
        """

        # list of modes of each moving limb
        moving_limbs_modes = [self.limbs['moving'][limb].modes for limb in self.limbs['moving'].keys()]

        # list of modes of atlas (tuples)
        modes_tuples = product(*moving_limbs_modes)

        # reorganization of atlas's modes into a list of dicts
        contact_modes = []
        for mode_tuple in modes_tuples:
            mode = {}
            for i, limb_mode in enumerate(mode_tuple):
                mode[self.limbs['moving'].keys()[i]] = limb_mode
            contact_modes.append(mode)

        return contact_modes

    def _differentiable_state(self):
        """
        Defines a dictionary with the elements of the state using differentiable arrays from the AD package.
        """

        x_diff = OrderedDict()

        # moving limbs
        for limb in self.limbs['moving'].keys():
            x_diff['q'+limb] = adnumber(np.zeros((2,1)))

        # body
        x_diff['qb'] = adnumber(np.zeros((2,1)))
        x_diff['vb'] = adnumber(np.zeros((2,1)))

        return x_diff

    def _differentiable_input(self):
        """
        Defines a dictionary with the elements of the input using differentiable arrays from the AD package.
        """

        u_diff = OrderedDict()

        # moving limbs
        for limb in self.limbs['moving'].keys():
            u_diff['v'+limb] = adnumber(np.zeros((2,1)))

        # fixed limbs
        for limb in self.limbs['fixed'].keys():
            u_diff['f'+limb] = adnumber(np.zeros((2,1)))

        return u_diff

    def _get_domains(self):
        """
        Reutrn the list of the domains in the (x,u) space for each mode of atlas. Each domain is a polytope and is derived differentiating the linear constraints. The list 'constraints' contains a set of vector inequalities in the variables (x_diff, u_diff); each inequality has to be interpreted as:
        f(x_diff, u_diff) <= 0
        and being each f(.) a linear function, we have equivalently
        grad(f) * (x_diff, u_diff) <= - f(0,0)
        """

        # mode independent constraints
        constraints = []
        constraints = self._state_constraints(constraints)
        constraints = self._input_constraints(constraints)

        # mode dependent constraints
        domains = []
        for mode in self.contact_modes:
            mode_constraint = self._contact_mode_constraints(mode, copy(constraints))
            A, B, c = self._matrices_from_linear_expression(mode_constraint)
            lhs = np.hstack((A, B))
            rhs = - c
            D = Polytope(lhs, rhs - lhs.dot(np.vstack((self.x_eq, self.u_eq))))
            D.assemble()
            domains.append(D)

        return domains

    def _state_constraints(self, constraints):
        """
        Generates differentiable expressions for the constraints on the state that don't depend on the mode.
        """

        # moving limbs
        for limb in self.limbs['moving'].keys():
            constraints.append(self.x_diff['q'+limb] - self.x_diff['qb'] - self.joint_limits[limb]['max'])
            constraints.append(self.joint_limits[limb]['min'] - self.x_diff['q'+limb] + self.x_diff['qb'])

        # fixed limbs
        for limb in self.limbs['fixed'].keys():
            constraints.append(self.equilibrium_configuration[limb] - self.x_diff['qb'] - self.joint_limits[limb]['max'])
            constraints.append(self.joint_limits[limb]['min'] - self.equilibrium_configuration[limb] + self.x_diff['qb'])

        # body
        constraints.append(self.x_diff['qb'] - self.joint_limits['b']['max'])
        constraints.append(self.joint_limits['b']['min'] - self.x_diff['qb'])
        constraints.append(self.x_diff['vb'] - self.velocity_limits['b']['max'])
        constraints.append(self.velocity_limits['b']['min'] - self.x_diff['vb'])

        return constraints

    def _input_constraints(self, constraints):
        """
        Generates differentiable expressions for the constraints on the input that don't depend on the mode.
        """

        # moving limbs
        for limb in self.limbs['moving'].keys():
            constraints.append(self.u_diff['v'+limb] - self.velocity_limits[limb]['max'])
            constraints.append(self.velocity_limits[limb]['min'] - self.u_diff['v'+limb])

        # fixed limbs
        mu  = self.parameters['friction_coefficient']
        for limb in self.limbs['fixed'].keys():
            constraints.append(self.u_diff['f'+limb] - self.force_limits[limb]['max'])
            constraints.append(self.force_limits[limb]['min'] - self.u_diff['f'+limb])

            # friction limits
            constraints.append(-mu*self.u_diff['f'+limb][0,0] + self.u_diff['f'+limb][1,0])
            constraints.append(-mu*self.u_diff['f'+limb][0,0] - self.u_diff['f'+limb][1,0])

        return constraints

    def _contact_mode_constraints(self, mode, constraints):
        """
        Adds to the list of contraints the constraints that are mode dependent.
        """
        
        mu  = self.parameters['friction_coefficient']
        for limb in self.limbs['moving'].keys():

            # moving limbs in the specified domain
            A = self.limbs['moving'][limb].A_domains[mode[limb]]
            b = self.limbs['moving'][limb].b_domains[mode[limb]]
            constraints.append(A.dot(self.x_diff['q'+limb]) - b)

            # friction constraints (surface equation: n^T x <= d)
            contact_surface = self.limbs['moving'][limb].contact_surfaces[mode[limb]]
            if contact_surface is not None:
                n = A[contact_surface, :].reshape(2,1) # normal vector
                t = np.array([[-n[1,0]],[n[0,0]]]) # tangential vector
                d = b[contact_surface, 0].reshape(1,1) # offset
                f_n = self.parameters['stiffness']*(d - n.T.dot(self.x_diff['q'+limb])) # normal force (>0)
                f_t = - self.parameters['damping']*(t.T.dot(self.u_diff['v'+limb])) # tangential force
                constraints.append(f_t - mu*f_n)
                constraints.append(- f_t - mu*f_n)

        return constraints

    def _remove_empty_domains(self, domains):
        """
        Removes from the domains and the contact modes all the modes that have an empty domain.
        """
        non_empty_contact_modes = []
        non_empty_domains = []
        for i, D in enumerate(domains):
            if not D.empty:
                non_empty_contact_modes.append(self.contact_modes[i])
                non_empty_domains.append(D)
        return non_empty_contact_modes, non_empty_domains

    def _get_affine_systems(self):
        """
        Returns the list of affine systems, one for each mode of the robot.
        """
        affine_systems = []
        for mode in self.contact_modes:
            dynamics = self._continuous_time_dynamics(mode)
            A_ct, B_ct, c_ct = self._matrices_from_linear_expression(dynamics)
            sys = DTAffineSystem.from_continuous(
                A_ct,
                B_ct,
                c_ct + A_ct.dot(self.x_eq) + B_ct.dot(self.u_eq), # traslates the equilibrium to the origin
                self.parameters['sampling_time'],
                self.parameters['integrator']
                )
            affine_systems.append(sys)
        return affine_systems

    def _continuous_time_dynamics(self, mode):
        """
        Returns the right hand side of the dynamics
        \dot x = f(x, u)
        where x and u are AD variables and f(.) is a linear function.
        """

        # position of the moving limbs (controlled in velocity)
        dynamics = []
        for limb in self.limbs['moving'].keys():
            dynamics.append(self.u_diff['v'+limb])

        # position of the body (simple integrator)
        dynamics.append(self.x_diff['vb'])

        # velocity of the body
        g = np.array([[0.],[-self.parameters['gravity']]])
        v_dot = adnumber(g)
        for limb in self.limbs['moving'].keys():
            v_dot = v_dot + self._force_moving_limb(limb, mode)/self.parameters['mass']
        for limb in self.limbs['fixed'].keys():
            v_dot = v_dot + self._force_fixed_limb(limb)/self.parameters['mass']
        dynamics.append(v_dot)

        return dynamics

    def _force_moving_limb(self, limb, mode):
        """
        For a given moving limb, in a given mode, returns the contact force f(x, u) (= f_n + f_t), where x and u are AD variables and f(.) is a linear function.
        """

        # domain
        A = self.limbs['moving'][limb].A_domains[mode[limb]]
        b = self.limbs['moving'][limb].b_domains[mode[limb]]
        contact_surface = self.limbs['moving'][limb].contact_surfaces[mode[limb]]

        # free space
        if contact_surface is None:
            f = adnumber(np.zeros((2,1)))

        # elastic wall
        else:
            n = A[contact_surface, :].reshape(2,1) # normal vector
            d = b[contact_surface, 0].reshape(1,1) # offset
            penetration = n.T.dot(self.x_diff['q'+limb]) - d
            f_n = - self.parameters['stiffness'] * penetration * n
            v_t = self.u_diff['v'+limb] - (self.u_diff['v'+limb].T.dot(n)) * n
            f_t = - self.parameters['damping'] * v_t
            f = f_n + f_t

        return f

    def _force_fixed_limb(self, limb):
        """
        For a given fixed limb, returns the contact force f(x, u) (= f_n + f_t), where x and u are AD variables and f(.) is a linear function.
        """
        n = self.limbs['fixed'][limb].normal
        t = np.array([[-n[1,0]],[n[0,0]]])
        f_n = n * self.u_diff['f'+limb][0,0]
        f_t = t * self.u_diff['f'+limb][1,0]
        f = f_n + f_t
        return f

    def _matrices_from_linear_expression(self, f):
        """
        Takes a linear expression in the form
        f(x, u)
        and returns (A, B, c) such that
        f(x, u) = A x + B u + c
        """

        # variables
        x = np.vstack(*[self.x_diff.values()])
        u = np.vstack(*[self.u_diff.values()])

        # concatenate if list
        if type(f) is list:
            f = np.vstack(f)

        # get matrices
        A = np.array(jacobian(f.flatten().tolist(), x.flatten().tolist()))
        B = np.array(jacobian(f.flatten().tolist(), u.flatten().tolist()))
        c = np.array([[el[0].x] for el in f])

        return A, B, c

    def penalize_relative_positions(self, Q):
        T = np.eye(self.n_x)
        n = len(self.limbs['moving'].keys())
        for i in range(n):
            T[2*i:2*(i+1),2*n:2*(n+1)] = -np.eye(2)
        return T.T.dot(Q).dot(T)

    def is_inside_a_domain(self, x):
        is_inside = False
        for D in self.pwa_system.domains:
            A_x = D.lhs_min[:,:self.n_x]
            A_u = D.lhs_min[:,self.n_x:]
            b_u = D.rhs_min - A_x.dot(x)
            cost = np.zeros((self.n_u, 1))
            sol = linear_program(cost, A_u, b_u)
            if not np.isnan(sol.min):
                is_inside = True
                break
        return is_inside

    def _set_visualizer(self):
        frame_min = np.array([[-.6],[-.6]])
        frame_max = np.array([[.6],[.4]])
        walls = []
        for limb in self.limbs['moving'].values():
            for mode in limb.modes:
            #for i in range(limb.n_domains):
                if limb.contact_surfaces[mode] is not None:
                    A = limb.A_domains[mode]
                    b = limb.b_domains[mode]
                    wall = Polytope(A, b)
                    wall.add_bounds(frame_min, frame_max)
                    wall.assemble()
                    walls.append(wall)
        limbs = self.limbs['moving'].keys() + self.limbs['fixed'].keys()
        visualizer = BoxAtlasVisualizer(self.equilibrium_configuration, walls, limbs)
        return visualizer

    def visualize(self, x):
        configuration = self._configuration_to_visualizer(x)
        self._visualizer.visualize(configuration)
        return

    def _configuration_to_visualizer(self, x):
        configuration = dict()
        for i, limb in enumerate(self.limbs['moving'].keys()):
            configuration[limb] = x[i*2:(i+1)*2, :]
        configuration['b'] = x[(i+1)*2:(i+2)*2, :]
        for limb in self.limbs['fixed'].keys():
            configuration[limb] = np.zeros((2,1))
        return configuration

    def state_labels(self):
        x = []
        for limb in self.limbs['moving'].keys():
            x += ['q' + limb + 'x', 'q' + limb + 'y']
        x += ['qbx', 'qby', 'vbx', 'vby']
        print 'Box-Atlas states:\n', x
        return

    def input_labels(self):
        u = []
        for limb in self.limbs['moving'].keys():
            u += ['v' + limb + 'x', 'v' + limb + 'y']
        for limb in self.limbs['fixed'].keys():
            u += ['f' + limb + 'n', 'f' + limb + 't']
        print 'Box-Atlas inputs:\n', u
        return

    def avoid_forbidden_transitions(self, controller):
        for limb_key, limb_value in self.limbs['moving'].items():
            for transition in limb_value.forbidden_transitions:
                previous_modes = [i for i, mode in enumerate(self.contact_modes) if mode[limb_key] == transition[0]]
                next_modes = [i for i, mode in enumerate(self.contact_modes) if mode[limb_key] == transition[1]]
                for previous_mode in previous_modes:
                    for next_mode in next_modes:
                        for k in range(controller.N-1):
                            expr = controller._d[k, previous_mode] + controller._d[k+1, next_mode]
                            controller._model.addConstr(expr <= 1.)
        return controller