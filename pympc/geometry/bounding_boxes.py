import numpy as np
from pympc.geometry.polytope import Polytope
from pympc.optimization.pnnls import linear_program
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from itertools import product
from matplotlib.path import Path
from matplotlib import patches

class Box:
    
    def __init__(self, x_min, x_max):
        if x_min.shape != x_max.shape:
            raise ValueError('Incoherent sizes of x_min and x_max.')
        self.x_min = x_min
        self.x_max = x_max
        self.volume = np.prod(x_max - x_min)
        self.d_split = np.argmax(self.x_max - self.x_min)
        self.x_d_split = self.x_min[self.d_split, 0] + (self.x_max[self.d_split, 0] - self.x_min[self.d_split, 0])/2.
        
    def is_inside(self, x):
        return np.min(x - self.x_min) > 0 and np.min(self.x_max - x) > 0
    
    def which_side(self, x):
        if not self.is_inside(x):
            return None
        else:
            if x[self.d_split, 0] < self.x_d_split:
                return 0
            else:
                return 1
            
    def plot(self, **kwargs):
        if max(self.x_min.shape) > 2:
            raise ValueError('Can plot only 2d boxes.')
        vertices = list(product([self.x_min[0,0],self.x_max[0,0]], [self.x_min[1,0],self.x_max[1,0]]))
        vertices = np.vstack(vertices)
        hull = ConvexHull(vertices)
        vertices = [hull.points[i].tolist() for i in hull.vertices]
        vertices += [vertices[0]]
        codes = [Path.MOVETO] + [Path.LINETO]*(len(vertices)-2) + [Path.CLOSEPOLY]
        path = Path(vertices, codes)
        ax = plt.gca()
        patch = patches.PathPatch(path, **kwargs)
        ax.add_patch(patch)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        return
        
    @staticmethod
    def from_point_set(point_set):
        point_set = np.hstack(point_set)
        n = point_set.shape[0]
        x_min = point_set.min(axis=1).reshape(n,1)
        x_max = point_set.max(axis=1).reshape(n,1)
        return Box(x_min, x_max)
            
class Polyhedron:
    
    def __init__(self, A, b):
        self.A = A
        self.b = b
        
    def split(self, d, x_d):
        a = np.zeros((1, self.A.shape[1]))
        a[0,d] = 1.
        A_0 = np.vstack((self.A, a))
        b_0 = np.vstack((self.b, np.array([[x_d]])))
        A_1 = np.vstack((self.A, - a))
        b_1 = np.vstack((self.b, - np.array([[x_d]])))
        return Polyhedron(A_0, b_0), Polyhedron(A_1, b_1)
        
class BoundingBox():
    
    def __init__(self, poly, project_to, path, min_vertices=None, max_vertices=None):
        self.poly = poly
        self.project_to = project_to
        self.path = path
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices
        self._complete_vertices()
        self.box = Box.from_point_set(self.min_vertices + self.max_vertices)
        self.children = None
        self.internal = self._internal()
        
    def _complete_vertices(self):
        if self.min_vertices is None:
            self.min_vertices = [None] * len(self.project_to)
        if self.max_vertices is None:
            self.max_vertices = [None] * len(self.project_to)
        for i, d in enumerate(self.project_to):
            f = np.zeros((self.poly.A.shape[1], 1))
            f[d, 0] = 1.
            if self.min_vertices[i] is None:
                self.min_vertices[i] = linear_program(f, self.poly.A, self.poly.b).argmin[self.project_to, :]
            if self.max_vertices[i] is None:
                self.max_vertices[i] = linear_program(-f, self.poly.A, self.poly.b).argmin[self.project_to, :]
        
    def get_children(self):
        poly_0, poly_1 = self.poly.split(self.box.d_split, self.box.x_d_split)
        min_vertices_0 = [None] * len(self.project_to)
        max_vertices_0 = [None] * len(self.project_to)
        min_vertices_1 = [None] * len(self.project_to)
        max_vertices_1 = [None] * len(self.project_to)
        for i in range(len(self.project_to)):
            if self.min_vertices[i][self.box.d_split, 0] < self.box.x_d_split:
                min_vertices_0[i] = self.min_vertices[i]
            else:
                min_vertices_1[i] = self.min_vertices[i]
            if self.max_vertices[i][self.box.d_split, 0] < self.box.x_d_split:
                max_vertices_0[i] = self.max_vertices[i]
            else:
                max_vertices_1[i] = self.max_vertices[i]
        path_0 = self.path + [(self.box.d_split, 0)]
        path_1 = self.path + [(self.box.d_split, 1)]
        box_0 = BoundingBox(poly_0, self.project_to, path_0, min_vertices_0, max_vertices_0)
        box_1 = BoundingBox(poly_1, self.project_to, path_1, min_vertices_1, max_vertices_1)
        self.children = [box_0, box_1]
        
    def _internal(self):
    	all_directions = list(product(range(len(self.project_to)), [0, 1]))
    	for direction in all_directions:
    		if self.path.count(direction) < 1:
    			return False
    	return True

        

class BoxTree:
    
    def __init__(self, A, b, project_to):
        self.poly = Polyhedron(A, b)
        self.project_to = project_to
        root_path = []
        self.root = BoundingBox(self.poly, project_to, root_path)
        
    def expand(self, min_volume):
        self._expand(min_volume, self.root)
                
    def _expand(self, min_volume, parent):
    	if parent.internal:
    		return
        parent.get_children()
        for child in parent.children:
            if child.box.volume > min_volume:
                self._expand(min_volume, child)
                
    def is_inside(self, x):
        if not self.root.box.is_inside(x):
            return False
        else:
            return self._is_inside(x, self.root)
        
    def _is_inside(self, x, parent):
        if parent.children is None:
            if parent.box.is_inside(x):
                return True
        else:
            child = parent.children[parent.box.which_side(x)]
            if not child.box.is_inside(x):
                return False
            else:
                return self._is_inside(x, child)
            
    def plot(self, **kwargs):
        self._plot(self.root, **kwargs)
        
    def _plot(self, parent, **kwargs):
        if parent.children is None:
        	if parent.internal:
        		parent.box.plot(facecolor='g', **kwargs)
        	else:
        		parent.box.plot(facecolor='b', **kwargs)
        else:
            for child in parent.children:
                self._plot(child, **kwargs)