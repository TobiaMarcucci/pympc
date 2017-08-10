import numpy as np
import scipy.linalg as linalg
from itertools import product
from copy import copy
from pympc.geometry.polytope import Polytope
from pympc.dynamical_systems import DTAffineSystem, DTPWASystem
from pympc.control import MPCHybridController
from pympc.optimization.pnnls import linear_program
from pympc.models.box_atlas_visualizer import BoxAtlasVisualizer

class MovingLimb():
    def __init__(self, A_domains, b_domains, contact_surfaces, forbidden_transitions=[]):
        self.n_domains = len(A_domains)
        self.A_domains = []
        self.b_domains = []
        for i in range(self.n_domains):
            norm_factor = np.linalg.norm(A_domains[i], axis=1)
            self.A_domains.append(np.divide(A_domains[i].T, norm_factor).T)
            self.b_domains.append(np.divide(b_domains[i].T, norm_factor).T)
        self.contact_surfaces = contact_surfaces
        self.forbidden_transitions = forbidden_transitions
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
        nominal_configuration,
        nominal_limb_forces,
        kinematic_limits,
        velocity_limits,
        force_limits
        ):
        self.limbs = limbs
        self.parameters = parameters
        self.nominal_configuration = nominal_configuration
        self.nominal_limb_forces = nominal_limb_forces
        self.kinematic_limits = kinematic_limits
        self.velocity_limits = velocity_limits
        self.force_limits = force_limits
        self.x_eq = self._get_equilibrium_state()
        self.u_eq = self._get_equilibrium_input()
        self.n_x = self.x_eq.shape[0]
        self.n_u = self.u_eq.shape[0]
        self.contact_modes = self._get_contact_modes()
        domains = self._get_domains()
        affine_systems = self._get_affine_systems()
        self.pwa_system = DTPWASystem(affine_systems, domains)
        self._visualizer = self._set_visualizer()
        return

    def _get_equilibrium_state(self):
        x_eq = np.zeros((0,1))
        for limb in self.limbs['moving'].keys():
            x_eq = np.vstack((x_eq, self.nominal_configuration[limb]))
        x_eq = np.vstack((x_eq, self.nominal_configuration['b'], np.zeros((2,1))))
        return x_eq

    def _get_equilibrium_input(self):
        u_eq = np.zeros((len(self.limbs['moving'].keys())*2, 1))
        for limb in self.limbs['fixed'].keys():
            u_eq = np.vstack((u_eq, self.nominal_limb_forces[limb]))
        return u_eq

    def _get_contact_modes(self):
        modes_tuples = product(*[range(self.limbs['moving'][limb].n_domains) for limb in self.limbs['moving'].keys()])
        contact_modes = []
        for mode_tuple in modes_tuples:
            mode = dict()
            for i, limb_mode in enumerate(mode_tuple):
                mode[self.limbs['moving'].keys()[i]] = limb_mode
            contact_modes.append(mode)
        return contact_modes

    def _get_domains(self):
        state_domains = []
        input_domains = []
        X = self._state_constraints()
        U = self._input_constraints()
        domain_list = []
        non_empty_domains = []
        for i, contact_mode in enumerate(self.contact_modes):
            D = self._contact_mode_constraints(contact_mode, X, U)
            if not D.empty:
                domain_list.append(D)
                non_empty_domains.append(i)
        self.contact_modes = [self.contact_modes[i] for i in non_empty_domains]
        return domain_list

    def _state_constraints(self):
        n = len(self.limbs['moving'].keys())
        selection_matrix = np.vstack((np.eye(2), -np.eye(2)))
        X = Polytope(np.zeros((0, 2*(n+2))), np.zeros((0, 1)))
        for i, limb in enumerate(self.limbs['moving'].keys()):
            lhs = np.hstack((
                    np.zeros((4, i*2)),
                    selection_matrix,
                    np.zeros((4, (n-1-i)*2)),
                    -selection_matrix,
                    np.zeros((4, 2))
                    ))
            rhs = np.vstack((
                self.kinematic_limits[limb]['max'],
                -self.kinematic_limits[limb]['min']
                ))
            X.add_facets(lhs, rhs)
        for limb in self.limbs['fixed'].keys():
            q_b_min = self.nominal_configuration[limb] - self.kinematic_limits[limb]['max']
            q_b_max = self.nominal_configuration[limb] - self.kinematic_limits[limb]['min']
            X.add_bounds(q_b_min, q_b_max, [2*n,2*n+1])
        X.add_bounds(self.kinematic_limits['b']['min'], self.kinematic_limits['b']['max'], [2*n, 2*n+1])
        X.add_bounds(self.velocity_limits['b']['min'], self.velocity_limits['b']['max'], [2*n+2, 2*n+3])
        return X

    def _input_constraints(self):
        # bounds
        u_min = np.zeros((0,1))
        u_max = np.zeros((0,1))
        for limb in self.limbs['moving'].keys():
            u_min = np.vstack((u_min, self.velocity_limits[limb]['min']))
            u_max = np.vstack((u_max, self.velocity_limits[limb]['max']))
        for limb in self.limbs['fixed'].keys():
            u_min = np.vstack((u_min, self.force_limits[limb]['min']))
            u_max = np.vstack((u_max, self.force_limits[limb]['max']))
        U = Polytope.from_bounds(u_min, u_max)
        # friction limits
        n_moving = len(self.limbs['moving'].keys())
        n_fixed = len(self.limbs['fixed'].keys())
        lhs_friction = np.array([
            [-self.parameters['friction_coefficient'], 1.],
            [-self.parameters['friction_coefficient'], -1.]
            ])
        lhs = np.hstack((
            np.zeros((n_fixed*2, n_moving*2)),
            linalg.block_diag(*[lhs_friction]*n_fixed)
            ))
        rhs = np.zeros((n_fixed*2,1))
        U.add_facets(lhs, rhs)
        return U

    def _contact_mode_constraints(self, contact_mode, X, U):
        
        # moving limbs in the specified domain
        X_mode = copy(X)
        n = len(self.limbs['moving'].keys())
        for i, limb in enumerate(self.limbs['moving'].keys()):
            A = self.limbs['moving'][limb].A_domains[contact_mode[limb]]
            b = self.limbs['moving'][limb].b_domains[contact_mode[limb]]
            lhs = np.hstack((
                    np.zeros((A.shape[0], i*2)),
                    A,
                    np.zeros((A.shape[0], 2*(n-i)+2))
                    ))
            X_mode.add_facets(lhs, b)

        # gather state and input constraints
        lhs = linalg.block_diag(*[X_mode.A, U.A])
        rhs = np.vstack((X_mode.b, U.b))
        D = Polytope(lhs, rhs)

        # friction constraints
        for i, limb in enumerate(self.limbs['moving'].keys()):
            contact_surface = self.limbs['moving'][limb].contact_surfaces[contact_mode[limb]]
            if contact_surface is not None:
                A = self.limbs['moving'][limb].A_domains[contact_mode[limb]]
                b = self.limbs['moving'][limb].b_domains[contact_mode[limb]]
                a = A[contact_surface, :]
                b = b[contact_surface, 0]
                lhs = np.zeros((2, self.n_x + self.n_u))
                lhs[:,i*2:(i+1)*2] = self.parameters['friction_coefficient'] * self.parameters['stiffness'] * np.array([
                    [a[0], a[1]],
                    [a[0], a[1]]
                    ])
                lhs[:,self.n_x+i*2:self.n_x+(i+1)*2] = self.parameters['damping'] * np.array([
                    [-a[1], a[0]],
                    [a[1], -a[0]]
                    ])
                rhs = self.parameters['friction_coefficient'] * self.parameters['stiffness'] * b * np.ones((2,1))
                D.add_facets(lhs, rhs)
        xu_eq = np.vstack((self.x_eq, self.u_eq))
        D = Polytope(D.A, D.b - D.A.dot(xu_eq))
        D.assemble()
        return D

    def _get_affine_systems(self):
        affine_systems = []
        for contact_mode in self.contact_modes:
            A_ct = self._get_A_ct(contact_mode)
            B_ct = self._get_B_ct(contact_mode)
            c_ct = self._get_c_ct(contact_mode)
            a_sys = DTAffineSystem.from_continuous(
                A_ct,
                B_ct,
                c_ct + A_ct.dot(self.x_eq) + B_ct.dot(self.u_eq), # traslates the equilibrium to the origin
                self.parameters['sampling_time'],
                self.parameters['integrator']
                )
            affine_systems.append(a_sys)
        return affine_systems

    def _get_A_ct(self, contact_mode):
        n = len(self.limbs['moving'].keys())
        A_upper = np.hstack((
            np.zeros((2*(n+1), 2*(n+1))),
            np.vstack((np.zeros((2*n, 2)), np.eye(2)))
            ))
        A_lower = np.zeros((2, 0))
        for limb, limb_mode in contact_mode.items():
            A_domain = self.limbs['moving'][limb].A_domains[limb_mode]
            contact_surface = self.limbs['moving'][limb].contact_surfaces[limb_mode]
            A_lower = np.hstack((
                A_lower,
                self._contact_contribution_A_ct(A_domain, contact_surface)
                ))
        A_lower = np.hstack((A_lower, np.zeros((2, 4))))
        return np.vstack((A_upper, A_lower))

    def _get_B_ct(self, contact_mode):
        n_moving = len(self.limbs['moving'].keys())
        n_fixed = len(self.limbs['fixed'].keys())
        B_upper = np.vstack((
            np.hstack((np.eye(2*n_moving), np.zeros((2*n_moving, 2*n_fixed)))),
            np.zeros((2, 2*(n_moving + n_fixed)))
            ))
        B_lower = np.zeros((2, 0))
        for limb, limb_mode in contact_mode.items():
            A_domain = self.limbs['moving'][limb].A_domains[limb_mode]
            contact_surface = self.limbs['moving'][limb].contact_surfaces[limb_mode]
            B_lower = np.hstack((
                B_lower,
                self._contact_contribution_B_ct(A_domain, contact_surface)
                ))
        for fixed_limb in self.limbs['fixed'].values():
            n = fixed_limb.normal
            B_fixed_limb = np.array([[n[0,0], -n[1,0]],[n[1,0], n[0,0]]])
            B_lower = np.hstack((
                B_lower,
                B_fixed_limb / self.parameters['mass']
                ))
        return np.vstack((B_upper, B_lower))

    def _get_c_ct(self, contact_mode):
        n = len(self.limbs['moving'].keys())
        c_upper = np.zeros((2*(n+1), 1))
        c_lower = np.array([[0.],[-self.parameters['gravity']]])
        for limb, limb_mode in contact_mode.items():
            A_domain = self.limbs['moving'][limb].A_domains[limb_mode]
            b_domain = self.limbs['moving'][limb].b_domains[limb_mode]
            contact_surface = self.limbs['moving'][limb].contact_surfaces[limb_mode]
            c_lower += self._contact_contribution_c_ct(A_domain, b_domain, contact_surface)
        return np.vstack((c_upper, c_lower))

    def _contact_contribution_A_ct(self, A_domain, contact_surface):
        if contact_surface is None:
            return np.zeros((2,2))
        else:
            a = A_domain[contact_surface,:]
            A_block = - np.array([
                [a[0]**2, a[0]*a[1]],
                [a[0]*a[1], a[1]**2]
                ]) * self.parameters['stiffness'] / self.parameters['mass']
            return A_block

    def _contact_contribution_B_ct(self, A_domain, contact_surface):
        if contact_surface is None:
            return np.zeros((2,2))
        else:
            a = A_domain[contact_surface,:]
            B_block = - np.array([
                [a[1]**2, -a[0]*a[1]],
                [-a[0]*a[1], a[0]**2]
                ]) * self.parameters['damping'] / self.parameters['mass']
            return B_block

    def _contact_contribution_c_ct(self, A_domain, b_domain, contact_surface):
        if contact_surface is None:
            return np.zeros((2,1))
        else:
            a = A_domain[contact_surface, :]
            b = b_domain[contact_surface, 0]
            c_block = self.parameters['stiffness'] * b * np.array([[a[0]],[a[1]]]) / self.parameters['mass']
            return c_block

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
        frame_min = np.array([[-.6],[-.5]])
        frame_max = np.array([[.6],[.4]])
        walls = []
        for limb in self.limbs['moving'].values():
            for i in range(limb.n_domains):
                if limb.contact_surfaces[i] is not None:
                    A = limb.A_domains[i]
                    b = limb.b_domains[i]
                    wall = Polytope(A, b)
                    wall.add_bounds(frame_min, frame_max)
                    wall.assemble()
                    walls.append(wall)
        limbs = self.limbs['moving'].keys() + self.limbs['fixed'].keys()
        visualizer = BoxAtlasVisualizer(self.nominal_configuration, walls, limbs)
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

    def print_state(self):
        x = []
        for limb in self.limbs['moving'].keys():
            x += ['q' + limb + 'x', 'q' + limb + 'y']
        x += ['qbx', 'qby', 'vbx', 'vby']
        print 'Box-Atlas states:\n', x
        return

    def print_input(self):
        u = []
        for limb in self.limbs['moving'].keys():
            u += ['v' + limb + 'x', 'v' + limb + 'y']
        for limb in self.limbs['fixed'].keys():
            u += ['f' + limb + 'n', 'f' + limb + 't']
        print 'Box-Atlas inputs:\n', u
        return

    def avoid_forbidden_transitions(self, controller):
        for limb_id, limb in self.limbs['moving'].items():
            for transition in limb.forbidden_transitions:
                previous_modes = [i for i, mode in enumerate(self.contact_modes) if mode[limb_id] == transition[0]]
                next_modes = [i for i, mode in enumerate(self.contact_modes) if mode[limb_id] == transition[1]]
                for previous_mode in previous_modes:
                    for next_mode in next_modes:
                        for k in range(controller.N-1):
                            expr = controller._d[k, previous_mode] + controller._d[k+1, next_mode]
                            controller._model.addConstr(expr <= 1.)
        return controller
