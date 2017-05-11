import unittest
import numpy as np
from pympc.geometry import Polytope, chebyshev_center
import matplotlib.pyplot as plt


class TestGeometry(unittest.TestCase):

    def test_chebyshev_center(self):

        # without equility
        A = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        b = np.array([[.5],[.5],[1.],[1.]])
        true_radius = .5
        [center, radius] = chebyshev_center(A,b)
        self.assertTrue(np.isclose(0.,center[0,0]))
        self.assertTrue(np.absolute(center[1,0]) <= .5)
        self.assertTrue(np.isclose(true_radius, radius))
        
        # with equility
        A = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        b = np.array([[.5],[.5],[1.],[1.]])
        C = np.array([[1.,0.]])
        d = np.array([[.1]])
        true_center = np.array([[.1],[0.]])
        true_radius = 1.
        [center, radius] = chebyshev_center(A,b,C,d)
        self.assertTrue(all(np.isclose(true_center, center)))
        self.assertTrue(np.isclose(true_radius, radius))
        
        # unbounded domain, but finite radius
        A = np.array([[1.,0.],[-1.,0.],[0.,1.]])
        b = np.array([[1.],[1.],[0.]])
        [center, radius] = chebyshev_center(A,b)
        true_radius = 1.
        self.assertTrue(np.isclose(true_radius, radius))
        self.assertTrue(np.isclose(center[0], 0.))

    def test_Polytope(self):

        # empty
        A = np.array([[1.,0.],[-1.,0.],[0.,1.]])
        b = np.array([[1.],[-2.],[0.]])
        p = Polytope(A,b)
        p.assemble()
        self.assertTrue(p.empty)

        # unbounded (easy)
        A = np.array([[1.,1.],[1.,-1.]])
        b = np.array([[0.],[0.]])
        p = Polytope(A, b)
        with self.assertRaises(ValueError):
            p.assemble()

        # unbounded (difficult)
        A = np.array([[1.,0.],[-1.,0.],[0.,1.]])
        b = np.array([[1.],[1.],[0.]])
        p = Polytope(A, b)
        with self.assertRaises(ValueError):
            p.assemble()

        # bounded and not empty
        A = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        b = np.array([[1.],[1.],[1.],[1.]])
        p = Polytope(A,b)
        p.assemble()
        self.assertFalse(p.empty)
        self.assertTrue(p.bounded)

        # coincident facets
        A = np.array([[1.,0.],[1.-1.e-10,1.e-10],[-1.,0.],[0.,1.],[0.,-1.]])
        b = np.ones((5,1))
        p = Polytope(A,b)
        p.assemble()
        true_coincident_facets = [[0,1], [0,1], [2], [3], [4]]
        self.assertEqual(p.coincident_facets, true_coincident_facets)
        self.assertTrue(p.minimal_facets[0] in [0,1])
        self.assertEqual(p.minimal_facets[1:], [2,3,4])

        # coincident facets, minimal facets, points on facets
        A = np.array([[1.,1.],[-1.,1.],[0.,-1.],[0.,1.],[0.,1.],[1.-1.e-10,1.-1.e-10]])
        b = np.array([   [1.],    [1.],    [1.],   [1.],   [2.],   [1.]])
        p = Polytope(A,b)
        p.assemble()
        true_coincident_facets = [[0,5], [1], [2], [3], [4], [0,5]]
        true_minimal_facets = [1, 2, 5]
        true_facet_centers = [[[-1.],[0.]], [[0.],[-1.]], [[1.],[0.]]]
        self.assertEqual(p.coincident_facets, true_coincident_facets)
        self.assertEqual(true_minimal_facets, p.minimal_facets)
        for i in range(0, len(true_facet_centers)):
            self.assertTrue(all(np.isclose(true_facet_centers[i], p.facet_centers(i))))

        # from_ and add_ methods
        x_max = np.ones((2,1))
        x_min = -x_max
        p = Polytope.from_bounds(x_min, x_max)
        p.add_bounds(x_min*2.,x_max/2.)
        A = np.array([[-1.,0.],[0.,-1.]])
        b = np.array([[.2],[2.]])
        p.add_facets(A, b)
        p.assemble()
        self.assertFalse(p.empty)
        self.assertTrue(p.bounded)
        true_vertices = [[.5,.5],[-.2,.5],[-.2,-1.],[.5,-1.]]
        self.assertTrue(all([any(np.all(np.isclose(vertex, true_vertices),axis=1)) for vertex in p.vertices]))

        # intersection and inclusion
        x_max = np.ones((2,1))
        x_min = -x_max
        p1 = Polytope.from_bounds(x_min, x_max)
        p1.assemble()
        x_max = np.ones((2,1))*2.
        x_min = -x_max
        p2 = Polytope.from_bounds(x_min, x_max)
        p2.assemble()
        x_min = np.zeros((2,1))
        p3 = Polytope.from_bounds(x_min, x_max)
        p3.assemble()
        x_max = np.ones((2,1))*5.
        x_min = np.ones((2,1))*4.
        p4 = Polytope.from_bounds(x_min, x_max)
        p4.assemble()

        # intersection
        self.assertTrue(p1.intersect_with(p2))
        self.assertTrue(p1.intersect_with(p3))
        self.assertFalse(p1.intersect_with(p4))

        # inclusion
        self.assertTrue(p1.included_in(p2))
        self.assertFalse(p1.included_in(p3))
        self.assertFalse(p1.included_in(p4))

        # # test fourier_motzkin_elimination
        # lhs = np.array([[1.,1.],[-1.,1.],[-1.,-9.]])
        # rhs = np.array([[1.],[3.],[3.]])
        # p = Polytope(lhs,rhs)
        # p.assemble()
        # row_0 = [1., max(p.vertices[:,1])]
        # row_1 = [-1., min(p.vertices[:,1])]
        # p_proj = p.fourier_motzkin_elimination(0)
        # rows = np.hstack((p_proj.lhs_min, p_proj.rhs_min))
        # self.assertTrue(row_0 in rows)
        # self.assertTrue(row_1 in rows)
        
        return

if __name__ == '__main__':
    unittest.main()