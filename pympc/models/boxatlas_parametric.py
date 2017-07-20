import numpy as np
from itertools import product
from copy import copy
from pympc.geometry.polytope import Polytope
from pympc.dynamical_systems import DTAffineSystem, DTPWASystem, dare, moas_closed_loop_from_orthogonal_domains
from pympc.control import MPCHybridController
import pympc.plot as mpc_plt
import matplotlib.pyplot as plt
from director.thirdparty import transformations
import scipy.spatial as spatial
import director.viewerclient as vc

from pyhull.halfspace import Halfspace, HalfspaceIntersection



# from matplotlib.path import Path
# import matplotlib.patches as patches


class MovingLimb():
    def __init__(self, A, b, contact_surface):
        norm_factor = np.linalg.norm(A, axis=1)
        self.A = np.divide(A.T, norm_factor).T
        self.b = np.divide(b.T, norm_factor).T
        self.contact_surface = contact_surface
        return

class FixedLimb():
    def __init__(self, position, normal):
        self.position = position
        self.normal = normal/np.linalg.norm(normal)
        return

class BoxAtlas():

    def __init__(
        self,
        topology,
        parameters,
        nominal_limb_positions,
        nominal_limb_forces,
        kinematic_limits,
        velocity_limits,
        force_limits
        ):
        self.topology = topology
        self.parameters = parameters
        self.nominal_limb_positions = nominal_limb_positions
        self.nominal_limb_forces = nominal_limb_forces
        self.kinematic_limits = kinematic_limits
        self.velocity_limits = velocity_limits
        self.force_limits = force_limits
        self.moving_limbs = self.topology['moving'].keys()
        self.fixed_limbs = self.topology['fixed'].keys()
        self.x_eq = self._equilibrium_state()
        self.u_eq = self._equilibrium_input()
        self.n_x = self.x_eq.shape[0]
        self.n_u = self.u_eq.shape[0]
        self.contact_modes = self._get_contact_modes()
        self.affine_systems = self._get_affine_systems()
        self.state_domains, self.input_domains = self._domains()
        self.pwa_system = DTPWASystem.from_orthogonal_domains(
            self.affine_systems,
            self.state_domains,
            self.input_domains)
        return

    def _equilibrium_state(self):
        x_eq = np.zeros((0,1))
        for limb in self.moving_limbs:
            x_eq = np.vstack((x_eq, self.nominal_limb_positions[limb]))
        body_nominal_state = np.zeros((4,1))
        return np.vstack((x_eq, body_nominal_state))

    def _equilibrium_input(self):
        u_eq = np.zeros((len(self.moving_limbs)*2, 1))
        for limb in self.fixed_limbs:
            u_eq = np.vstack((u_eq, self.nominal_limb_forces[limb]))
        return u_eq

    def _get_contact_modes(self):
        modes_tuples = product(*[range(len(self.topology['moving'][limb])) for limb in self.moving_limbs])
        contact_modes = []
        for mode_tuple in modes_tuples:
            mode = dict()
            for i, limb_mode in enumerate(mode_tuple):
                mode[self.moving_limbs[i]] = limb_mode
            contact_modes.append(mode)
        return contact_modes

    def _contact_contribution_A_ct(self, moving_limb):
        if moving_limb.contact_surface is None:
            return np.zeros((2,2))
        else:
            a = moving_limb.A[moving_limb.contact_surface,:]
            A_block = - np.array([
                [a[0]**2, a[0]*a[1]],
                [a[0]*a[1], a[1]**2]
                ]) * self.parameters['stiffness'] / self.parameters['mass']
            return A_block

    def _contact_contribution_B_ct(self, moving_limb):
        if moving_limb.contact_surface is None:
            return np.zeros((2,2))
        else:
            a = moving_limb.A[moving_limb.contact_surface,:]
            B_block = - np.array([
                [a[1]**2, -a[0]*a[1]],
                [-a[0]*a[1], a[0]**2]
                ]) * self.parameters['damping'] / self.parameters['mass']
            return B_block

    def _contact_contribution_c_ct(self, moving_limb):
        if moving_limb.contact_surface is None:
            return np.zeros((2,1))
        else:
            a = moving_limb.A[moving_limb.contact_surface,:]
            b = moving_limb.b[moving_limb.contact_surface,0]
            c_block = self.parameters['stiffness'] * b * np.array([[a[0]],[a[1]]]) / self.parameters['mass']
            return c_block

    def _get_A_ct(self, contact_mode):
        n = len(self.moving_limbs)
        A_upper = np.hstack((
            np.zeros((2*(n+1), 2*(n+1))),
            np.vstack((np.zeros((2*n, 2)), np.eye(2)))
            ))
        A_lower = np.zeros((2, 0))
        for limb, limb_mode in contact_mode.items():
            A_lower = np.hstack((
                A_lower,
                self._contact_contribution_A_ct(self.topology['moving'][limb][limb_mode])
                ))
        A_lower = np.hstack((A_lower, np.zeros((2, 4))))
        return np.vstack((A_upper, A_lower))

    def _get_B_ct(self, contact_mode):
        n_moving = len(self.moving_limbs)
        n_fixed = len(self.fixed_limbs)
        B_upper = np.vstack((
            np.hstack((np.eye(2*n_moving), np.zeros((2*n_moving, 2*n_fixed)))),
            np.zeros((2, 2*(n_moving + n_fixed)))
            ))
        B_lower = np.zeros((2, 0))
        for limb, limb_mode in contact_mode.items():
            B_lower = np.hstack((
                B_lower,
                self._contact_contribution_B_ct(self.topology['moving'][limb][limb_mode])
                ))
        for fixed_limb in self.topology['fixed'].values():
            n = fixed_limb.normal
            B_fixed_limb = np.array([[n[0,0], -n[1,0]],[n[1,0], n[0,0]]])
            B_lower = np.hstack((
                B_lower,
                B_fixed_limb / self.parameters['mass']
                ))
        return np.vstack((B_upper, B_lower))

    def _get_c_ct(self, contact_mode):
        n = len(self.moving_limbs)
        c_upper = np.zeros((2*(n+1), 1))
        c_lower = np.array([[0.],[-self.parameters['gravity']]])
        for limb, limb_mode in contact_mode.items():
            c_lower += self._contact_contribution_c_ct(self.topology['moving'][limb][limb_mode])
        return np.vstack((c_upper, c_lower))

    def _get_affine_systems(self):
        affine_systems = []
        for contact_mode in self.contact_modes:
            A_ct = self._get_A_ct(contact_mode)
            B_ct = self._get_B_ct(contact_mode)
            c_ct = self._get_c_ct(contact_mode)
            a_sys = DTAffineSystem.from_continuous(
                A_ct,
                B_ct,
                c_ct + A_ct.dot(self.x_eq) + B_ct.dot(self.u_eq),
                self.parameters['sampling_time'],
                'explicit_euler'
                )
            affine_systems.append(a_sys)
        return affine_systems

    def _state_constraints(self):
        n = len(self.moving_limbs)
        selection_matrix = np.vstack((np.eye(2), -np.eye(2)))
        X = Polytope(np.zeros((0, 2*(n+2))), np.zeros((0, 1)))
        for i, limb in enumerate(self.moving_limbs):
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
        for limb in self.fixed_limbs:
            q_b_min = self.nominal_limb_positions[limb] - self.kinematic_limits[limb]['max']
            q_b_max = self.nominal_limb_positions[limb] - self.kinematic_limits[limb]['min']
            X.add_bounds(q_b_min, q_b_max, [2*n,2*n+1])
        X.add_bounds(self.kinematic_limits['b']['min'], self.kinematic_limits['b']['max'], [2*n, 2*n+1])
        X.add_bounds(self.velocity_limits['b']['min'], self.velocity_limits['b']['max'], [2*n+2, 2*n+3])
        X = Polytope(X.A, X.b - X.A.dot(self.x_eq))
        return X

    def _input_constraints(self):
        u_min = np.zeros((0,1))
        u_max = np.zeros((0,1))
        for limb in self.moving_limbs:
            u_min = np.vstack((u_min, self.velocity_limits[limb]['min']))
            u_max = np.vstack((u_max, self.velocity_limits[limb]['max']))
        for limb in self.fixed_limbs:
            u_min = np.vstack((u_min, self.force_limits[limb]['min']))
            u_max = np.vstack((u_max, self.force_limits[limb]['max']))
        U = Polytope.from_bounds(u_min, u_max)
        U = Polytope(U.A, U.b - U.A.dot(self.u_eq))
        return U

    def _contact_mode_constraints(self, contact_mode):
        n = len(self.moving_limbs)
        X = Polytope(np.zeros((0, 2*(n+2))), np.zeros((0, 1)))
        for i, limb in enumerate(self.moving_limbs):
            A = self.topology['moving'][limb][contact_mode[limb]].A
            b = self.topology['moving'][limb][contact_mode[limb]].b
            lhs = np.hstack((
                    np.zeros((A.shape[0], i*2)),
                    A,
                    np.zeros((A.shape[0], 2*(n-i)+2))
                    ))
            X.add_facets(lhs, b)
        X = Polytope(X.A, X.b - X.A.dot(self.x_eq))
        return X

    def _domains(self):
        state_domains = []
        input_domains = []
        X = self._state_constraints()
        U = self._input_constraints()
        U.assemble()
        input_domains = [U] * len(self.contact_modes)
        for contact_mode in self.contact_modes:
            X_mode = self._contact_mode_constraints(contact_mode)
            X_mode.add_facets(X.A, X.b)
            X_mode.assemble()
            state_domains.append(X_mode)
        return state_domains, input_domains

    def penalize_relative_positions(self, Q):
        T = np.eye(self.n_x)
        n = len(self.moving_limbs)
        for i in range(n):
            T[2*i:2*(i+1),2*n:2*(n+1)] = -np.eye(2)
        return T.T.dot(Q).dot(T)

    def _state_vector_to_dict(self, x_vec):
        x_dict = dict()
        for i, limb in enumerate(self.moving_limbs):
            x_dict[limb] = x_vec[i*2:(i+1)*2, :]
        x_dict['bq'] = x_vec[(i+1)*2:(i+2)*2, :]
        x_dict['bv'] = x_vec[(i+2)*2:(i+3)*2, :]
        return x_dict

    def visualize_environment(self, vis):
        z_lim = [-.3,.3]
        environment_limits = np.ones((2,1)) * .6
        for limb in self.moving_limbs:
            for i, moving_limb in enumerate(self.topology['moving'][limb]):
                if moving_limb.contact_surface is not None:
                    contact_domain = Polytope(moving_limb.A, moving_limb.b)
                    contact_domain.add_bounds(-environment_limits, environment_limits)
                    contact_domain.assemble()
                    contact_domain = extrude_2d_polytope(contact_domain, z_lim)
                    vis = visualize_3d_polytope(contact_domain, 'w_' + limb + '_' + str(i), vis)
        for limb in self.fixed_limbs:
            fixed_limb = self.topology['fixed'][limb]
            b = fixed_limb.normal.T.dot(fixed_limb.position)
            contact_domain = Polytope(fixed_limb.normal.T, b)
            contact_domain.add_bounds(-environment_limits, environment_limits)
            contact_domain.assemble()
            contact_domain = extrude_2d_polytope(contact_domain, z_lim)
            vis = visualize_3d_polytope(contact_domain, 'w_' + limb, vis)
        return vis

    def visualize(self, vis, x_vec):
        x_dict = self._state_vector_to_dict(x_vec)
        for limb in self.moving_limbs:
            translation = [0.] + list((self.nominal_limb_positions[limb] + x_dict[limb]).flatten())
            vis[limb].settransform(transformations.translation_matrix(translation))
        for limb in self.fixed_limbs:
            translation = [0.] + list((self.topology['fixed'][limb].position).flatten())
            vis[limb].settransform(transformations.translation_matrix(translation))
        translation = [0.] + list(x_dict['bq'].flatten())
        vis['bq'].settransform(transformations.translation_matrix(translation))
        return vis

    # def visualize(self, x_vec):
    #     x_dict = self._state_vector_to_dict(x_vec)
    #     body_vertices = [[.1, .1],[-.1,.1],[-.1,-.1],[.1,-.1],[.1, .1]]
    #     body_vertices = [[vertex[0] + x_dict['bq'][0,0], vertex[1] + x_dict['bq'][1,0]] for vertex in body_vertices]
    #     codes = [Path.MOVETO] + [Path.LINETO]*(len(body_vertices)-2) + [Path.CLOSEPOLY]
    #     path = Path(body_vertices, codes)
    #     ax = plt.gca()
    #     body_patch = patches.PathPatch(path, facecolor='b')
    #     patches = [body_patch]
    #     ax.add_patch(body)
    #     for limb in self.moving_limbs:
    #         limb_position = tuple((self.nominal_limb_positions[limb] + x_dict[limb]).flatten())
    #         limb_patch = plt.Circle(limb_position, 0.05, facecolor='g', edgecolor='k')
    #         patches.append(limb_patch)
    #         ax.add_artist(limb_patch)
    #     for limb in self.fixed_limbs:
    #         limb_position = tuple((self.topology['fixed'][limb].position).flatten())
    #         limb_patch = plt.Circle(limb_position, 0.05, facecolor='g', edgecolor='k')
    #         patches.append(limb_patch)
    #         ax.add_artist(limb_patch)
    #     return body_patch

    def print_state(self):
        x = []
        for limb in self.moving_limbs:
            x += ['q_' + limb + '_x', 'q_' + limb + '_y']
        x += ['q_b_x', 'q_b_y', 'v_b_x', 'v_b_x']
        print x
        return

    def print_input(self):
        u = []
        for limb in self.moving_limbs:
            u += ['v_' + limb + '_x', 'v_' + limb + '_y']
        for limb in self.fixed_limbs:
            u += ['f_' + limb + '_n', 'f_' + limb + '_t']
        print u
        return


class Mesh(vc.BaseGeometry):
    __slots__ = ["vertices", "triangular_faces"]

    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.triangular_faces = []
        for face in faces:
            for i in range(1, len(face)-1):
                self.triangular_faces.append([face[0], face[i], face[i+1]])

    def serialize(self):
        return {
            "type": "mesh_data",
            "vertices": self.vertices,
            "faces": self.triangular_faces
        }

def visualize_3d_polytope(p, name, visualizer):
    p = reorder_coordinates(p)
    halfspaces = []
    # change of coordinates because qhull is stupid...
    b_qhull = p.rhs_min - p.lhs_min.dot(p.center)
    for i in range(p.lhs_min.shape[0]):
        halfspace = Halfspace(p.lhs_min[i,:].tolist(), (-b_qhull[i,0]).tolist())
        halfspaces.append(halfspace)
    p_qhull = HalfspaceIntersection(halfspaces, np.zeros(p.center.shape).flatten().tolist())
    vertices = p_qhull.vertices + np.hstack([p.center]*len(p_qhull.vertices)).T
    mesh = Mesh(vertices.tolist(), p_qhull.facets_by_halfspace)
    visualizer[name].setgeometry(mesh)
    return visualizer

def extrude_2d_polytope(p_2d, z_limits):
    A = np.vstack((
        np.hstack((p_2d.lhs_min, np.zeros((p_2d.lhs_min.shape[0], 1)))),
        np.hstack((np.zeros((2, p_2d.lhs_min.shape[1])), np.array([[1.],[-1.]])))
        ))
    b = np.vstack((p_2d.rhs_min, np.array([[z_limits[1]],[-z_limits[0]]])))
    p_3d = Polytope(A, b)
    p_3d.assemble()
    return p_3d

def reorder_coordinates(p):
    T = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    A = p.lhs_min.dot(T)
    p_rotated = Polytope(A, p.rhs_min)
    p_rotated.assemble()
    return p_rotated





