import numpy as np
import scipy.linalg as linalg
from itertools import product
from copy import copy
from pympc.geometry.polytope import Polytope
from pympc.dynamical_systems import DTAffineSystem, DTPWASystem, dare, moas_closed_loop_from_orthogonal_domains
from pympc.control import MPCHybridController
from pympc.optimization.pnnls import linear_program
import pympc.plot as mpc_plt
import matplotlib.pyplot as plt
from director.thirdparty import transformations
import scipy.spatial as spatial
import director.viewerclient as vc
from pyhull.halfspace import Halfspace, HalfspaceIntersection
from boxatlas_parameters import visualizer_parameters as vp


class BoxAtlasVisualizer():

    def __init__(self, limbs):
        self.limbs = limbs
        self.vis = vc.Visualizer()['box_altas']
        self._translate_visualizer()
        self._initialize_environment()
        self._initialize_robot()
        return

    def _translate_visualizer(self):
        h_lf = 0.
        if self.limbs['moving'].has_key('lf'):
            h_lf = self.limbs['moving']['lf'].nominal_position[1,0]
        elif self.limbs['fixed'].has_key('lf'):
            h_lf = self.limbs['fixed']['lf'].position[1,0]
        h_rf = 0.
        if self.limbs['moving'].has_key('rf'):
            h_rf = self.limbs['moving']['rf'].nominal_position[1,0]
        elif self.limbs['fixed'].has_key('rf'):
            h_rf = self.limbs['fixed']['rf'].position[1,0]
        vertical_translation = - (h_lf + h_rf) / 2.
        self.vis.settransform(vc.transformations.translation_matrix([0.,0.,vertical_translation]))
        return

    def _initialize_environment(self):
        walls = []
        for limb_key, limb_value in self.limbs['moving'].items():
            for mode in limb_value.modes:
                if limb_value.contact_surfaces[mode] is not None:
                    name = 'wall_' + limb_key + '_' + mode
                    wall = Polytope(limb_value.A_domains[mode], limb_value.b_domains[mode])
                    wall.add_bounds(vp['min'], vp['max'])
                    wall.assemble()
                    walls.append([wall, name])
        for limb_key, limb_value in self.limbs['fixed'].items():
            name = 'wall_' + limb_key
            wall = self._box_from_surface(limb_value.position, limb_value.normal, limb_value.normal.T.dot(limb_value.position))
            walls.append([wall, name])
        for [wall, name] in walls:
            self._visualize_2d_polytope(wall, name)
        return

    @staticmethod
    def _box_from_surface(x, n, d):
        t = np.array([[-n[1,0]],[n[0,0]]])
        c = t.T.dot(x)
        lhs = np.vstack((n.T, -n.T, t.T, -t.T))
        rhs = np.vstack((
            d,
            -d + vp['box_fixed_feet']['tickness'],
            c + vp['box_fixed_feet']['width'],
            -c + vp['box_fixed_feet']['width']
            ))
        box = Polytope(lhs, rhs)
        box.assemble()
        return box

    def _visualize_2d_polytope(self, p2d, name):
        p3d = self._extrude_2d_polytope(p2d)
        self._visualize_3d_polytope(p3d, name)
        return

    @staticmethod
    def _extrude_2d_polytope(p2d):
        A = np.vstack((
            np.hstack((
                p2d.lhs_min,
                np.zeros((p2d.lhs_min.shape[0], 1))
                )),
            np.hstack((
                np.zeros((2, p2d.lhs_min.shape[1])),
                np.array([[1.],[-1.]])
                ))
            ))
        b = np.vstack((
            p2d.rhs_min,
            np.array([[vp['depth'][1]], [-vp['depth'][0]]])
            ))
        p3d = Polytope(A, b)
        p3d.assemble()
        return p3d

    def _visualize_3d_polytope(self, p, name):
        p = self._reorder_coordinates(p)
        halfspaces = []
        # change of coordinates because qhull is stupid...
        b_qhull = p.rhs_min - p.lhs_min.dot(p.center)
        for i in range(p.lhs_min.shape[0]):
            halfspace = Halfspace(p.lhs_min[i,:].tolist(), (-b_qhull[i,0]).tolist())
            halfspaces.append(halfspace)
        p_qhull = HalfspaceIntersection(halfspaces, np.zeros(p.center.shape).flatten().tolist())
        vertices = p_qhull.vertices + np.hstack([p.center]*len(p_qhull.vertices)).T
        mesh = Mesh(vertices.tolist(), p_qhull.facets_by_halfspace)
        self.vis[name].setgeometry(mesh)
        return

    @staticmethod
    def _reorder_coordinates(p):
        T = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
        A = p.lhs_min.dot(T)
        p_reordered = Polytope(A, p.rhs_min)
        p_reordered.assemble()
        return p_reordered

    def _initialize_robot(self):

        # body
        self.vis['b'].setgeometry(
            vc.GeometryData(
                vc.Box(
                    lengths = [vp['body_size']]*3),
                    color = vp['body_color']
                    )
            )

        # moving and fixed limbs
        for limb in self.limbs['moving'].keys() + self.limbs['fixed'].keys():
            self.vis[limb].setgeometry(
                vc.GeometryData(
                    vc.Sphere(radius = vp['limbs_size']),
                    color = vp['limbs_color']
                    )
                )

        # place fixed limbs in nominal position
        for limb_key, limb_value in self.limbs['fixed'].items():
            translation = [0.] + limb_value.position.flatten().tolist()
            self.vis[limb_key].settransform(transformations.translation_matrix(translation))

        return

    def visualize(self, x):

        # body
        translation = [0.] + x['qb'].flatten().tolist()
        self.vis['b'].settransform(
            transformations.translation_matrix(translation).dot(
            transformations.rotation_matrix(x['tb'][0,0], np.array([1.,0.,0.]))
            )
            )

        # moving limbs
        for limb_key, limb_value in self.limbs['moving'].items():
            translation = x[limb_key] + limb_value.nominal_position
            translation = [0.] + translation.flatten().tolist()
            self.vis[limb_key].settransform(transformations.translation_matrix(translation))

        # parametric walls
        for limb_key, limb_value in self.limbs['moving'].items():
            for mode in limb_value.modes:
                if limb_value.contact_surfaces[mode] is not None:
                    name = 'wall_' + limb_key + '_' + mode
                    for parameter in limb_value.parameters:
                        if parameter.has_key(mode):
                            p = x[parameter['label']]
                            A = copy(limb_value.A_domains[mode])
                            b = copy(limb_value.b_domains[mode])
                            for i, index in enumerate(parameter[mode]['index']):
                                b[index, 0] += p * parameter[mode]['coefficient'][i]
                            wall = Polytope(A, b)
                            wall.add_bounds(vp['min'], vp['max'])
                            wall.assemble()
                            self._visualize_2d_polytope(wall, name)
        return

class Mesh(vc.BaseGeometry):
    __slots__ = ['vertices', 'triangular_faces']

    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.triangular_faces = []
        for face in faces:
            for i in range(1, len(face)-1):
                self.triangular_faces.append([face[0], face[i], face[i+1]])

    def serialize(self):
        return {
            'type': 'mesh_data',
            'vertices': self.vertices,
            'faces': self.triangular_faces
        }
