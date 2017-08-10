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


class BoxAtlasVisualizer():

    def __init__(self, x_nominal, walls, limbs):
        self.x_nominal = x_nominal
        self.walls = walls
        self.limbs = limbs
        self.vis = vc.Visualizer()['box_altas']
        vertical_translation = - (x_nominal['lf'][1,0] + x_nominal['rf'][1,0]) / 2.
        self.vis.settransform(vc.transformations.translation_matrix([0.,0.,vertical_translation]))
        self._visualize_environment()
        self._initialize_body()
        return

    def _visualize_environment(self, depth=[-.3,.3]):
        for i, wall in enumerate(self.walls):
            self._visualize_2d_polytope(wall, depth, 'wall_'+str(i))
        return

    def _visualize_2d_polytope(self, p2d, depth, name):
        p3d = self._extrude_2d_polytope(p2d, depth)
        self._visualize_3d_polytope(p3d, name)
        return

    @staticmethod
    def _extrude_2d_polytope(p2d, depth):
        A = np.vstack((
            np.hstack((p2d.lhs_min, np.zeros((p2d.lhs_min.shape[0], 1)))),
            np.hstack((np.zeros((2, p2d.lhs_min.shape[1])), np.array([[1.],[-1.]])))
            ))
        b = np.vstack((p2d.rhs_min, np.array([[depth[1]],[-depth[0]]])))
        p3d = Polytope(A, b)
        p3d.assemble()
        return p3d

    def _visualize_3d_polytope(self, p, name):
        p = self._reorder_coordinates_visualizer(p)
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
    def _reorder_coordinates_visualizer(p):
        T = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
        A = p.lhs_min.dot(T)
        p_reordered = Polytope(A, p.rhs_min)
        p_reordered.assemble()
        return p_reordered

    def _initialize_body(self):
        self.vis['b'].setgeometry(
            vc.GeometryData(
                vc.Box(
                    lengths = [.2]*3),
                    color = np.hstack((np.array([0.,0.,1.]), 1.))
                    )
            )

        for limb in self.limbs:
            self.vis[limb].setgeometry(
                vc.GeometryData(
                    vc.Sphere(radius = .05),
                    color = np.hstack((np.array([1.,0.,0.]), 1.))
                    )
                )
        return

    def visualize(self, x):
        translation = [
        0.,
        x['b'][0,0] + self.x_nominal['b'][0,0],
        x['b'][1,0] + self.x_nominal['b'][1,0]
        ]
        self.vis['b'].settransform(transformations.translation_matrix(translation))
        for limb in self.limbs:
            translation = [
            0.,
            x[limb][0,0] + self.x_nominal[limb][0,0],
            x[limb][1,0] + self.x_nominal[limb][1,0]
            ]
            self.vis[limb].settransform(transformations.translation_matrix(translation))
        return

    # def visualize(self, x):
    #     translation = [
    #     0.,
    #     x['qbx'] + self.x_nominal['qbx'],
    #     x['qby'] + self.x_nominal['qby']
    #     ]
    #     self.vis['b'].settransform(transformations.translation_matrix(translation))
    #     for limb in self.limbs:
    #         translation = [
    #         0.,
    #         x['q' + limb +'x'] + self.x_nominal['q' + limb +'x'],
    #         x['q' + limb +'y'] + self.x_nominal['q' + limb +'y']
    #         ]
    #         self.vis[limb].settransform(transformations.translation_matrix(translation))
    #     return

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
