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
from pympc.models.boxatlas_visualizer import BoxAtlasVisualizer
from pympc.dynamical_systems import dare, moas_closed_loop
from pympc.geometry.polytope import LowerDimensionalPolytope
from pympc.control import reachability_standard_form
import boxatlas_parameters

class MovingLimb():
    def __init__(self, A_domains, b_domains, contact_surfaces, nominal_position, forbidden_transitions=[], parameters=[]):

        # copy inputs
        self.modes = A_domains.keys()
        self.n_domains = len(A_domains)
        self.A_domains = {}
        self.b_domains = {}
        self.contact_surfaces = contact_surfaces
        self.nominal_position = nominal_position
        self.forbidden_transitions = forbidden_transitions
        self.parameters = parameters

        # normalize all the A and all the b and the parameters
        for mode in A_domains.keys():
            norm_factor = np.linalg.norm(A_domains[mode], axis=1)
            self.A_domains[mode] = np.divide(A_domains[mode].T, norm_factor).T
            self.b_domains[mode] = np.divide(b_domains[mode].T, norm_factor).T
            for p in self.parameters:
                if p.has_key(mode):
                    for i, index in enumerate(p[mode]['index']):
                        p[mode]['coefficient'][i] /= norm_factor[index]

        return

class FixedLimb():
    def __init__(self, position, normal, nominal_force):
        self.position = position
        self.normal = normal/np.linalg.norm(normal)
        self. nominal_force = nominal_force
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

    def __init__(self, limbs, nominal_mode):
        self.limbs = limbs
        self.nominal_mode = nominal_mode
        self.n_p = sum([len(limb.parameters) for limb in self.limbs['moving'].values()])
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
        self.nominal_system, self.nominal_domain = self._extract_nominal_configuration()
        self._check_equilibrium_point()
        self.Q = self._state_cost_hessian()
        self.R = self._input_cost_hessian()
        self.P, self.K, self.X_N = self._terminal_lqr_controller()
        self._visualizer = BoxAtlasVisualizer(self.limbs)
        # self._visualizer = self._initialize_visualizer()
        return

    def _get_equilibrium_state(self):
        """
        Translates the equilibrium_configuration dictionary in the equilibrium state vector x_equilibrium.
        """

        # parameters values (the equilibrium is set arbitrarily to zero...)
        x_equilibrium = np.zeros((self.n_p, 1))

        # moving limbs position
        for limb in self.limbs['moving'].values():
            x_equilibrium = np.vstack((x_equilibrium, limb.nominal_position))

        # body state: limbs + body position and velocity (linear + angular)
        x_equilibrium = np.vstack((x_equilibrium, np.zeros((6,1))))

        return x_equilibrium

    def _get_equilibrium_input(self):
        """
        Translates the equilibrium_limb_forces dictionary in the equilibrium input vector u_equilibrium.
        """

        # velocities of the moving limbs
        u_equilibrium = np.zeros((len(self.limbs['moving'])*2, 1))

        # forces of the fixed limbs
        for limb in self.limbs['fixed'].values():
            u_equilibrium = np.vstack((u_equilibrium, limb.nominal_force))

        return u_equilibrium

    def _differentiable_state(self):
        """
        Defines a dictionary with the elements of the state using differentiable arrays from the AD package.
        """

        # parameters
        x_diff = OrderedDict()
        for limb in self.limbs['moving'].values():
            for parameter in limb.parameters:
                x_diff[parameter['label']] = adnumber(np.zeros((1,1)))

        # moving limbs
        for limb in self.limbs['moving'].keys():
            x_diff['q'+limb] = adnumber(np.zeros((2,1)))

        # body
        x_diff['qb'] = adnumber(np.zeros((2,1)))
        x_diff['tb'] = adnumber(np.zeros((1,1)))
        x_diff['vb'] = adnumber(np.zeros((2,1)))
        x_diff['ob'] = adnumber(np.zeros((1,1)))

        return x_diff

    def _differentiable_input(self):
        """
        Defines a dictionary with the elements of the input using differentiable arrays from the AD package.
        """

        # moving limbs
        u_diff = OrderedDict()
        for limb in self.limbs['moving'].keys():
            u_diff['v'+limb] = adnumber(np.zeros((2,1)))

        # fixed limbs
        for limb in self.limbs['fixed'].keys():
            u_diff['f'+limb] = adnumber(np.zeros((2,1)))

        return u_diff

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
            rhs = - c - lhs.dot(np.vstack((self.x_eq, self.u_eq))) # translated to the equilibrium
            D = Polytope(lhs, rhs)
            D.assemble()
            domains.append(D)

        return domains

    def _state_constraints(self, constraints):
        """
        Generates differentiable expressions for the constraints on the state that don't depend on the mode:
        constraints <= 0.
        """

        # joint limits for the moving limbs
        for limb in self.limbs['moving'].keys():
            constraints.append(self.x_diff['q'+limb] - self.x_diff['qb'] - boxatlas_parameters.joint_limits[limb]['max'])
            constraints.append(boxatlas_parameters.joint_limits[limb]['min'] - self.x_diff['q'+limb] + self.x_diff['qb'])

        # joint limits for the fixed limbs
        for limb_key, limb_value in self.limbs['fixed'].items():
            constraints.append(limb_value.position - self.x_diff['qb'] - boxatlas_parameters.joint_limits[limb_key]['max'])
            constraints.append(boxatlas_parameters.joint_limits[limb_key]['min'] - limb_value.position + self.x_diff['qb'])

        # joint limits for the body position
        pos_b = np.vstack((self.x_diff['qb'], self.x_diff['tb']))
        constraints.append(pos_b - boxatlas_parameters.joint_limits['b']['max'])
        constraints.append(boxatlas_parameters.joint_limits['b']['min'] - pos_b)

        # velocity limits for the body velocity
        vel_b = np.vstack((self.x_diff['vb'], self.x_diff['ob']))
        constraints.append(vel_b - boxatlas_parameters.velocity_limits['b']['max'])
        constraints.append(boxatlas_parameters.velocity_limits['b']['min'] - vel_b)

        # limits for the parameters
        for limb in self.limbs['moving'].values():
            for parameter in limb.parameters:
                constraints.append(self.x_diff[parameter['label']] - parameter['max'])
                constraints.append(parameter['min'] - self.x_diff[parameter['label']])

        return constraints

    def _input_constraints(self, constraints):
        """
        Generates differentiable expressions for the constraints on the input that don't depend on the mode.
        """

        # moving limbs
        for limb in self.limbs['moving'].keys():
            constraints.append(self.u_diff['v'+limb] - boxatlas_parameters.velocity_limits[limb]['max'])
            constraints.append(boxatlas_parameters.velocity_limits[limb]['min'] - self.u_diff['v'+limb])

        # fixed limbs
        mu  = boxatlas_parameters.friction
        for limb in self.limbs['fixed'].keys():
            constraints.append(self.u_diff['f'+limb] - boxatlas_parameters.force_limits[limb]['max'])
            constraints.append(boxatlas_parameters.force_limits[limb]['min'] - self.u_diff['f'+limb])

            # friction limits
            constraints.append(-mu*self.u_diff['f'+limb][0,0] + self.u_diff['f'+limb][1,0])
            constraints.append(-mu*self.u_diff['f'+limb][0,0] - self.u_diff['f'+limb][1,0])

        return constraints

    def _contact_mode_constraints(self, mode, constraints):
        """
        Adds to the list of contraints the constraints that are mode dependent.
        """
        
        mu  = boxatlas_parameters.friction
        for limb_key, limb_value in self.limbs['moving'].items():

            # moving limbs in the specified domain
            A = self.limbs['moving'][limb_key].A_domains[mode[limb_key]]
            b = self.limbs['moving'][limb_key].b_domains[mode[limb_key]]
            for parameter in limb_value.parameters:
                if parameter.has_key(mode[limb_key]):
                    for i, index in enumerate(parameter[mode[limb_key]]['index']):
                        b_parametric = np.zeros(b.shape)
                        b_parametric[index, 0] = 1.
                        b = b + b_parametric * parameter[mode[limb_key]]['coefficient'][i] * self.x_diff[parameter['label']][0,0]
            constraints.append(A.dot(self.x_diff['q'+limb_key]) - b)

            # friction constraints (surface equation: n^T x <= d)
            contact_surface = self.limbs['moving'][limb_key].contact_surfaces[mode[limb_key]]
            if contact_surface is not None:
                n = A[contact_surface, :].reshape(2,1) # normal vector
                t = np.array([[-n[1,0]],[n[0,0]]]) # tangential vector
                d = np.array([[b[contact_surface, 0]]]) # offset
                f_n = boxatlas_parameters.stiffness*(d - n.T.dot(self.x_diff['q'+limb_key])) # normal force (>0)
                f_t = - boxatlas_parameters.damping*(t.T.dot(self.u_diff['v'+limb_key])) # tangential force
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
            A_dt, B_dt, c_dt = self._semi_implicit_euler(mode)
            sys = DTAffineSystem(A_dt, B_dt, c_dt)
            affine_systems.append(sys)
        return affine_systems

    def _first_order_dynamics(self):
        """
        Right hand side of the first order dynamics.
        """

        dynamics_1 = []

        # constant parameters
        for limb in self.limbs['moving'].values():
            for parameter in limb.parameters:
                dynamics_1.append(np.zeros((1,1)))

        # position of the moving limbs (controlled in velocity)
        for limb in self.limbs['moving'].keys():
            dynamics_1.append(self.u_diff['v'+limb])

        return np.vstack(dynamics_1)

    def _second_order_dynamics(self, mode):
        """
        Right hand side of the second order dynamics.
        """

        # body linear velocity
        dynamics_2 = []
        g = np.array([[0.],[-boxatlas_parameters.gravity]])
        v_dot = adnumber(g)
        for limb in self.limbs['moving'].keys():
            v_dot = v_dot + self._force_moving_limb(limb, mode)/boxatlas_parameters.mass
        for limb in self.limbs['fixed'].keys():
            v_dot = v_dot + self._force_fixed_limb(limb)/boxatlas_parameters.mass
        dynamics_2.append(v_dot)

        # body angular velocity
        o_dot = adnumber(np.zeros((1,1)))
        for limb_key, limb_value in self.limbs['moving'].items():
            r_i = limb_value.nominal_position
            f_i = self._force_moving_limb(limb_key, mode)
            o_dot = o_dot + cross_product_2d(r_i, f_i)/boxatlas_parameters.moment_of_inertia
        for limb_key, limb_value in self.limbs['fixed'].items():
            r_i = limb_value.position
            f_i = self._force_fixed_limb(limb_key)
            o_dot = o_dot + cross_product_2d(r_i, f_i)/boxatlas_parameters.moment_of_inertia
        dynamics_2.append(o_dot)

        return np.vstack(dynamics_2)

    def _semi_implicit_euler(self, mode):
        """
        Impements the semi-impict Euler method for the dynamics of boxatlas:
        \dot  q_l = B_1 u
        \ddot q_b = A_21 q_l + A_22 q_b + A_23 \dot q_b + B_2 u + c_2
        First derives the velocity of the body at the next time step
        v_b (t+1) = v_b (t) + h * \ddot q_b (t),
        then updates the position as
        q_b (t+1) = q_b (t) + h * v_b (t+1),
        the position of the limbs is updated as
        q_l (t+1) = q_l (t) + h * \dot  q_l (t).
        This discretization scheme preserves continuity of the PWA dynamics.
        """

        # update parameter dynamics and position of the limbs
        dynamics_1 = self._first_order_dynamics()
        h = boxatlas_parameters.sampling_time
        p = [self.x_diff[parameter['label']] for limb in self.limbs['moving'].values() for parameter in limb.parameters]
        q_l = [self.x_diff['q'+limb] for limb in self.limbs['moving'].keys()]
        p_and_q_l_next =  np.vstack(p + q_l) + h*dynamics_1

        # update velocity of the body
        dynamics_2 = self._second_order_dynamics(mode)
        v_b_next = np.vstack((self.x_diff['vb'], self.x_diff['ob'])) + h*dynamics_2

        # update position of the body
        q_b_next = np.vstack((self.x_diff['qb'], self.x_diff['tb'])) + h*v_b_next

        # extract affine-system matrices
        discrete_dynamics = np.vstack([p_and_q_l_next, q_b_next, v_b_next])
        A_dt, B_dt, c_dt = self._matrices_from_linear_expression(discrete_dynamics)

        # shift the equilibrium to the origin
        c_dt = (A_dt - np.eye(self.n_x)).dot(self.x_eq) + B_dt.dot(self.u_eq) + c_dt

        return A_dt, B_dt, c_dt

    def _force_moving_limb(self, limb, mode):
        """
        For a given moving limb, in a given mode, returns the contact force f(x, u) (= f_n + f_t), where x and u are AD variables and f(.) is a linear function.
        """

        # domain
        A = self.limbs['moving'][limb].A_domains[mode[limb]]
        b = self.limbs['moving'][limb].b_domains[mode[limb]]
        for parameter in self.limbs['moving'][limb].parameters:
            if parameter.has_key(mode[limb]):
                for i, index in enumerate(parameter[mode[limb]]['index']):
                    b_parametric = np.zeros(b.shape)
                    b_parametric[index, 0] = 1.
                    b = b + b_parametric * parameter[mode[limb]]['coefficient'][i] * self.x_diff[parameter['label']][0,0]
        contact_surface = self.limbs['moving'][limb].contact_surfaces[mode[limb]]

        # free space
        if contact_surface is None:
            f = adnumber(np.zeros((2,1)))

        # elastic wall
        else:
            n = A[contact_surface, :].reshape(2,1) # normal vector
            d = np.array([[b[contact_surface, 0]]]) # offset
            penetration = n.T.dot(self.x_diff['q'+limb]) - d
            f_n = - boxatlas_parameters.stiffness * penetration * n
            v_t = self.u_diff['v'+limb] - (self.u_diff['v'+limb].T.dot(n)) * n
            f_t = - boxatlas_parameters.damping * v_t
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

    def _extract_nominal_configuration(self):
        for i, mode in enumerate(self.contact_modes):
            if mode == self.nominal_mode:
                nominal_mode_index = i
        nominal_system = self.pwa_system.affine_systems[nominal_mode_index]
        nominal_domain = self.pwa_system.domains[nominal_mode_index]
        return nominal_system, nominal_domain


    def _check_equilibrium_point(self, tol= 1.e-6):
        """
        Check that the nominal configuration of the moving limbs and the nominal forces of the fixed limbs are an equilibrium point for atlas in the nominal mode.
        """
        offset_norm = np.linalg.norm(self.nominal_system.c)
        if offset_norm > tol:
            raise ValueError('Given nominal configuration and forces are not an equilibrium state.')
        return

    def _state_cost_hessian(self):

        # position of moving limbs
        state_cost = 0.
        for limb in self.limbs['moving'].keys():
            q_rel = self.x_diff['q'+limb] - self.x_diff['qb']
            state_cost += boxatlas_parameters.state_cost['q'+limb+'_rel'] * q_rel.T.dot(q_rel)

        # body
        for label in ['qb', 'tb', 'vb', 'ob']:
            state_cost += boxatlas_parameters.state_cost[label] * self.x_diff[label].T.dot(self.x_diff[label])

        # get the hessian matrix
        x = np.vstack(*[self.x_diff.values()])
        Q = .5*np.array(state_cost[0,0].hessian(x.flatten().tolist()))

        return Q

    def _input_cost_hessian(self):

        # velocity of moving limbs
        input_cost = 0.
        for limb in self.limbs['moving'].keys():
            input_cost += boxatlas_parameters.input_cost['v'+limb] * self.u_diff['v'+limb].T.dot(self.u_diff['v'+limb])

        # force on fixed limbs
        for limb in self.limbs['fixed'].keys():
            input_cost += boxatlas_parameters.input_cost['f'+limb] * self.u_diff['f'+limb].T.dot(self.u_diff['f'+limb])

        # get the hessian matrix
        u = np.vstack(*[self.u_diff.values()])
        R = .5*np.array(input_cost[0,0].hessian(u.flatten().tolist()))

        return R

    def _terminal_lqr_controller(self):

        # reachability analysis for the system without the dynamics of the parameters
        A_non_p = self.nominal_system.A[self.n_p:,self.n_p:]
        B_non_p = self.nominal_system.B[self.n_p:,:]
        Q_non_p = self.Q[self.n_p:,self.n_p:]
        rsf = reachability_standard_form(A_non_p, B_non_p)

        # # if the system is controllable
        # if rsf['n_R'] == self.n_x - self.n_p:
        #     P_non_p, K_non_p = dare(A_non_p, B_non_p, Q_non_p, self.R)
        #     P = linalg.block_diag(np.zeros((self.n_p, self.n_p)), P_non_p)
        #     K = np.hstack((np.zeros((self.n_u, self.n_p)), K_non_p))
        #     X_N = moas_closed_loop(self.nominal_system.A, self.nominal_system.B, K, self.nominal_domain)

        # # if the system is not controllable
        # else:

        # solve lower-dimensional dare
        Q_R = rsf['T_R'].T.dot(Q_non_p).dot(rsf['T_R'])
        P_R, K_R = dare(rsf['A_RR'], rsf['B_R'], Q_R, self.R)

        # augment the lower-dimensional system with the parameter dynamics
        A_RR_aug = linalg.block_diag(np.eye(self.n_p), rsf['A_RR'])
        B_R_aug = np.vstack((np.zeros((self.n_p, self.n_u)), rsf['B_R']))
        K_R_aug = np.hstack((np.zeros((self.n_u, self.n_p)), K_R))

        # domain of the augmented lower-dimensional system
        D_lhs_p = copy(self.nominal_domain.lhs_min[:, :self.n_p])
        D_lhs_x = copy(self.nominal_domain.lhs_min[:, self.n_p:self.n_x])
        D_lhs_u = copy(self.nominal_domain.lhs_min[:, self.n_x:])
        D_lhs_R_aug = np.hstack((D_lhs_p, D_lhs_x.dot(rsf['T_R']), D_lhs_u))
        D_rhs_R_aug = copy(self.nominal_domain.rhs_min)
        D_R_aug = Polytope(D_lhs_R_aug, D_rhs_R_aug)
        D_R_aug.assemble()

        # augmented lower-dimensional terminal set
        X_N_R_aug = moas_closed_loop(A_RR_aug, B_R_aug, K_R_aug, D_R_aug)

        # back to the original coordinates
        X_N_lhs_ineq = np.hstack((
            X_N_R_aug.lhs_min[:,:self.n_p],
            X_N_R_aug.lhs_min[:,self.n_p:].dot(rsf['T_inv_R'])
            ))
        X_N_rhs_ineq = X_N_R_aug.rhs_min
        X_N_lhs_eq = np.hstack((np.zeros((rsf['T_inv_N'].shape[0], self.n_p)), rsf['T_inv_N']))
        X_N_rhs_eq = np.zeros((rsf['T_inv_N'].shape[0], 1))
        X_N = LowerDimensionalPolytope(
            X_N_lhs_ineq,
            X_N_rhs_ineq,
            X_N_lhs_eq,
            X_N_rhs_eq
            )

        # retrieve the full dimensional lqr
        P = linalg.block_diag(np.zeros((self.n_p, self.n_p)), rsf['T_inv_R'].T.dot(P_R).dot(rsf['T_inv_R']))
        K = np.hstack((np.zeros((self.n_u, self.n_p)), K_R.dot(rsf['T_inv_R'])))
        
        return P, K, X_N

    def is_inside_a_domain(self, x):
        """
        Checks if there exists an input vector u such that the given state is inside of at least one domain.
        """
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

    # def _initialize_visualizer(self):
    #     """
    #     TO BE MOVED IN boxatlas_visualizer.py!
    #     """
    #     walls = []
    #     for limb in self.limbs['moving'].values():
    #         for mode in limb.modes:
    #             if limb.contact_surfaces[mode] is not None:
    #                 wall = Polytope(limb.A_domains[mode], limb.b_domains[mode])
    #                 wall.add_bounds(boxatlas_parameters.visualizer_min, boxatlas_parameters.visualizer_max)
    #                 wall.assemble()
    #                 walls.append(wall)
    #     for limb in self.limbs['fixed'].values():
    #         support_box = self._box_from_surface(limb.position, limb.normal, limb.normal.T.dot(limb.position))
    #         walls.append(support_box)
    #     visualizer = BoxAtlasVisualizer(walls, self.limbs)
    #     return visualizer

    # @staticmethod
    # def _box_from_surface(x, n, d, depth=.05, width=.1):
    #     t = np.array([[-n[1,0]],[n[0,0]]])
    #     c = t.T.dot(x)
    #     lhs = np.vstack((n.T, -n.T, t.T, -t.T))
    #     rhs = np.vstack((d, -d+depth, c+width, -c+width))
    #     box = Polytope(lhs, rhs)
    #     box.assemble()
    #     return box

    def visualize(self, x):
        configuration = self._configuration_to_visualizer(x)
        self._visualizer.visualize(configuration)
        return

    def _configuration_to_visualizer(self, x):
        configuration = dict()
        i = 0
        for limb in self.limbs['moving'].values():
            for parameter in limb.parameters:
                configuration[parameter['label']] = x[i:i+1,:]
                i += 1
        for i, limb in enumerate(self.limbs['moving'].keys()):
            configuration[limb] = x[i*2+self.n_p:(i+1)*2+self.n_p, :]
        configuration['qb'] = x[(i+1)*2+self.n_p:(i+2)*2+self.n_p, :]
        configuration['tb'] = x[(i+2)*2+self.n_p:2*i+5+self.n_p, :]
        return configuration

    def print_state_labels(self):
        x = []
        for limb in self.limbs['moving'].values():
            for parameter in limb.parameters:
                x += [parameter['label']]
        for limb in self.limbs['moving'].keys():
            x += ['q' + limb + 'x', 'q' + limb + 'y']
        x += ['qbx', 'qby', 'tb', 'vbx', 'vby', 'ob']
        print 'Box-Atlas states:\n', x
        return

    def print_input_labels(self):
        u = []
        for limb in self.limbs['moving'].keys():
            u += ['v' + limb + 'x', 'v' + limb + 'y']
        for limb in self.limbs['fixed'].keys():
            u += ['f' + limb + 'n', 'f' + limb + 't']
        print 'Box-Atlas inputs:\n', u
        return

    def print_mode_sequence(self, mode_sequence):
        for mode in mode_sequence:
            print self.contact_modes[mode]
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

def cross_product_2d(a, b):
    return np.array([[a[0,0]*b[1,0] - a[1,0]*b[0,0]]])