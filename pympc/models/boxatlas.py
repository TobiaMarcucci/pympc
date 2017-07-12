import numpy as np
from itertools import combinations
from copy import copy
from pympc.geometry.polytope import Polytope
from pympc.dynamical_systems import DTAffineSystem, DTPWASystem, dare
from pympc.control import MPCHybridController


class BoxAtlasKinematicLimits(object):
    def __init__(self):
        # position bounds
        # body
        self.q_b_min = np.array([[0.3],[0.3]])
        self.q_b_max = np.array([[0.7],[0.7]])

        # left foot (limits in the body frame)
        self.q_lf_min = np.array([[0.],[-.7]])
        self.q_lf_max = np.array([[.4],[-.3]])

        # right foot (limits in the body frame)
        self.q_rf_min = np.array([[-.4],[-.7]])
        self.q_rf_max = np.array([[0.],[-.3]])

        # hand (limits in the body frame)
        self.q_h_min = np.array([[-.6],[-.1]])
        self.q_h_max = np.array([[-.2],[.3]])

        # velocity bounds

        # body
        self.v_b_max = 5 * np.ones((2,1))
        self.v_b_min = - self.v_b_max

        self.polytope = None  # will be generated in self._assemble()

    def _assemble(self):
        selection_matrix = np.vstack((np.eye(2), -np.eye(2)))

        # left foot
        lhs = np.hstack((-selection_matrix, selection_matrix, np.zeros((4,6))))
        rhs = np.vstack((self.q_lf_max, -self.q_lf_min))
        self.polytope = Polytope(lhs, rhs)

        # right foot
        lhs = np.hstack((-selection_matrix, np.zeros((4,2)), selection_matrix, np.zeros((4,4))))
        rhs = np.vstack((self.q_rf_max, -self.q_rf_min))
        self.polytope.add_facets(lhs, rhs)

        # hand
        lhs = np.hstack((-selection_matrix, np.zeros((4,4)), selection_matrix, np.zeros((4,2))))
        rhs = np.vstack((self.q_h_max, -self.q_h_min))
        self.polytope.add_facets(lhs, rhs)

        # body
        self.polytope.add_bounds(self.q_b_min, self.q_b_max, [0,1])
        self.polytope.add_bounds(self.v_b_min, self.v_b_max, [8,9])

        self.polytope.assemble()
        return self.polytope


class BoxAtlasInputLimits(object):
    def __init__(self):
        # left foot
        self.v_lf_max = 5 * np.ones((2,1))
        self.v_lf_min = - self.v_lf_max

        # right foot
        self.v_rf_max = 5 * np.ones((2,1))
        self.v_rf_min = - self.v_rf_max

        # hand
        self.v_h_max = 5 * np.ones((2,1))
        self.v_h_min = - self.v_rf_max

        self.polytope = None  # will be generated in self._assemble()

    def _assemble(self, kinematic_limits, friction_coefficient, stiffness):
        # force bounds

        # left foot
        f_lf_x_max = - friction_coefficient * stiffness * (kinematic_limits.q_b_min[1,0]+kinematic_limits.q_lf_min[1,0])
        f_lf_x_min = - f_lf_x_max

        # right foot
        f_rf_x_max = - friction_coefficient * stiffness * (kinematic_limits.q_b_min[1,0]+kinematic_limits.q_rf_min[1,0])
        f_rf_x_min = - f_rf_x_max

        # hand
        f_h_y_max = - friction_coefficient * stiffness * (kinematic_limits.q_b_min[0,0]+kinematic_limits.q_h_min[0,0])
        f_h_y_min = - f_h_y_max

        input_max = np.vstack((self.v_lf_max, self.v_rf_max, self.v_h_max, f_lf_x_max, f_rf_x_max, f_h_y_max))
        input_min = np.vstack((self.v_lf_min, self.v_rf_min, self.v_h_min, f_lf_x_min, f_rf_x_min, f_h_y_min))
        self.polytope = Polytope.from_bounds(input_min, input_max).assemble()
        return self.polytope


class BoxAtlasPWAModel(object):
    contacts = ['lf', 'rf', 'h']
    modes =  [mode for n_contacts in range(len(contacts)+1) for mode in combinations(contacts, n_contacts)]

    def __init__(self,
        # scalar
        mass = 1.,
        stiffness = 100.,
        gravity = 10.,
        friction_coefficient = .5,
        t_s = .1,
        kinematic_limits=BoxAtlasKinematicLimits(),
        input_limits=BoxAtlasInputLimits()):

        self.mass = mass
        self.stiffness = stiffness
        self.gravity = gravity
        self.friction_coefficient = friction_coefficient
        self.t_s = t_s
        self.kinematic_limits = kinematic_limits
        self.input_limits = input_limits
        self.mode_independent_constraints = self._mode_independent_constraints(
            self.kinematic_limits, self.input_limits)
        self.pwa_system = self._pwa_system(self._translated_affine_systems(),
                                           self._translated_domains())

    def _pwa_system(self, translated_affine_systems, translated_domains):
        return DTPWASystem(translated_affine_systems, translated_domains)

    def _translated_affine_systems(self):
        x_eq, u_eq = self.equilibrium_point()
        return [
            DTAffineSystem.from_continuous(
                self._get_A(mode),
                self._get_B(mode),
                self._get_c() + self._get_A(mode).dot(x_eq),
                self.t_s,
                'explicit_euler') for mode in self.modes]

    def equilibrium_point(self):
        # equilibrium point
        x_eq = np. array([
            [.5], # q_b_x
            [.5], # q_b_y
            [.7], # q_lf_x
            [-self.mass*self.gravity/2./self.stiffness], # q_lf_y
            [.3], # q_rf_x
            [-self.mass*self.gravity/2./self.stiffness], # q_rf_y
            [.2], # q_h_x
            [.6], # q_h_y
            [0.], # v_b_x
            [0.], # v_b_y
            ])
        u_eq = np.zeros((9,1))
        return x_eq, u_eq

    def _mode_independent_constraints(self, kinematic_limits, input_limits):
        kinematic_limits._assemble()
        input_limits._assemble(kinematic_limits, self.friction_coefficient, self.stiffness)
        lhs = np.vstack((
            np.hstack((
                kinematic_limits.polytope.A,
                np.zeros((
                    kinematic_limits.polytope.A.shape[0],
                    input_limits.polytope.A.shape[1]
                    ))
                )),
            np.hstack((
                np.zeros((
                    input_limits.polytope.A.shape[0],
                    kinematic_limits.polytope.A.shape[1]
                    )),
                input_limits.polytope.A
                ))
            ))
        rhs = np.vstack((kinematic_limits.polytope.b, input_limits.polytope.b))
        return Polytope(lhs, rhs)

    def _contact_constraints(self, domain, contact, active):
        contact_indices = {
        'lf': {'q': 3, 'f': 16},
        'rf': {'q': 5, 'f': 17},
        'h': {'q': 6, 'f': 18}
        }
        if active:
            domain.add_upper_bounds(np.array([[0.]]), [contact_indices[contact]['q']])
            lhs = np.zeros((2,19))
            lhs[0, contact_indices[contact]['f']] = 1.
            lhs[0, contact_indices[contact]['q']] = self.friction_coefficient * self.stiffness
            lhs[1, contact_indices[contact]['f']] = -1.
            lhs[1, contact_indices[contact]['q']] = self.friction_coefficient * self.stiffness
            rhs = np.zeros((2,1))
            domain.add_facets(lhs, rhs)
        else:
            domain.add_lower_bounds(np.array([[0.]]), [contact_indices[contact]['q']])
        return domain

    def _domains(self):
        domains = []
        for mode in self.modes:
            domain = copy(self.mode_independent_constraints)
            for contact in self.contacts:
                domain = self._contact_constraints(domain, contact, contact in mode)
            domains.append(domain)
        return domains

    def _translated_domains(self):
        x_eq, u_eq = self.equilibrium_point()
        translated_domains = []
        for domain in self._domains():
            translated_domain = Polytope(domain.A, domain.b - domain.A.dot(np.vstack((x_eq, u_eq))))
            translated_domain.assemble()
            translated_domains.append(translated_domain)
        return translated_domains

    def _get_A(self, contact_set):
        contact_map_normal_force = {'lf':(9,3), 'rf':(9,5), 'h':(8,6)}
        A = np.vstack((
            np.hstack((np.zeros((2, 8)), np.eye(2))),
            np.zeros((8, 10))
            ))
        for contact in contact_set:
            A[contact_map_normal_force[contact]] = -self.stiffness/self.mass
        return A

    def _get_B(self, contact_set):
        contact_map_tangential_force = {'lf':(8,6), 'rf':(8,7), 'h':(9,8)}
        contact_map_tangential_velocity = {'lf':(2,0), 'rf':(4,2), 'h':(7,5)}
        B = np.vstack((
            np.zeros((2, 9)),
            np.hstack((np.eye(6), np.zeros((6, 3)))),
            np.zeros((2, 9))
            ))
        for contact in contact_set:
            B[contact_map_tangential_force[contact]] = 1./self.mass
        for contact in contact_set:
            B[contact_map_tangential_velocity[contact]] = 0.
        return B

    def _get_c(self):
        return np.vstack((np.zeros((9, 1)), np.array((-self.gravity))))

    def controller(self,
                   N=10,
                   Q=10 * np.eye(10),
                   R=np.eye(9),
                   objective_norm="two",
                   X_N=Polytope.from_bounds(-np.ones((10,1)), np.ones((10,1)))):
        # terminal set and cost
        # terminal_mode = 4
        #P, K = dare(translated_affine_systems[terminal_mode].A, translated_affine_systems[terminal_mode].B, Q, R)
        #X_N = ds.moas_closed_loop(translated_affine_systems[terminal_mode].A, translated_affine_systems[terminal_mode].B, K, X[1], U[1])
        P = Q
        if not X_N.assembled:
            X_N.assemble()

        # hybrid controller
        return MPCHybridController(self.pwa_system, N, objective_norm, Q, R, P, X_N)

    def random_state(self, X=None, controller=None):
        """
        Sample a random state within the lower and upper bounds of the piecewise
        affine system, and further restrict that state to some polytope X. By
        default, X is the polytope defined by the robot's kinematic limits.

        If `controller` is not None, use the given controller to check if there
        is a feasible input sequence from the given sample before returning it
        """
        if X is None:
            A = self.kinematic_limits.polytope.A
            b = self.kinematic_limits.polytope.b
            x_eq, u_eq = self.equilibrium_point()
            X = Polytope(A, b - A.dot(x_eq)).assemble()

        while True:
            x = np.random.rand(self.pwa_system.n_x, 1)
            x = np.multiply(x, (self.pwa_system.x_max - self.pwa_system.x_min)) + self.pwa_system.x_min
            if X.applies_to(x):
                if controller is not None:
                    u, xtraj, ss, cost = controller.feedforward(x)
                    if np.any(np.isnan(u[0])):
                        continue
                return x
