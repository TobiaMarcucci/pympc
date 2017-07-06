import unittest
import numpy as np
from pympc.geometry.polytope import Polytope
from pympc.geometry.chebyshev_center import chebyshev_center
from pympc.geometry.convex_hull import plane_through_points
from pympc.geometry.convex_hull import ConvexHull

class TestGeometry(unittest.TestCase):

    def test_chebyshev_center(self):

        # without equility
        A = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        b = np.array([[.5],[.5],[1.],[1.]])
        true_radius = .5
        [center, radius] = chebyshev_center(A, b)
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
        true_vertices = [np.array([[.5],[.5]]),np.array([[-.2],[.5]]),np.array([[-.2],[-1.]]),np.array([[.5],[-1.]])]
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

    def test_orthogonal_projection(self):

        # test plane generator
        points = [np.array([[1.],[0.],[0.]]), np.array([[-1.],[0.],[0.]]), np.array([[0.],[1.],[0.]])]
        a, b = plane_through_points(points)
        self.assertTrue(np.allclose(a, np.array([[0.],[0.],[1.]])))
        self.assertTrue(np.allclose(b, np.array([[0.]])))
        points = [np.array([[1.],[0.],[0.]]), np.array([[0.],[1.],[0.]]), np.array([[0.],[0.],[1.]])]
        a, b = plane_through_points(points)
        real_a = np.ones((3,1))/np.sqrt(3)
        self.assertTrue(np.allclose(a, real_a))
        self.assertTrue(np.isclose(b[0,0], real_a[0,0]))

        # test CHM
        n_test = 100
        n_ineq = 20
        n_var = 5
        res = [1,3,4]
        for i in range(n_test):
            everything_ok = False
            while not everything_ok:
                A = np.random.randn(n_ineq, n_var)
                b = np.random.rand(n_ineq, 1)
                x_offeset = np.random.rand(n_var, 1)
                b += A.dot(x_offeset)
                p = Polytope(A, b)
                try:
                    p.assemble()
                    everything_ok = True
                except ValueError:
                    pass
            p_proj_ve = p.orthogonal_projection(res, 'vertex_enumeration')
            for method in ['convex_hull']:#, 'block_elimination']:
                p_proj = p.orthogonal_projection(res, method)
                self.assertTrue(p_proj.lhs_min.shape[0] == p_proj_ve.lhs_min.shape[0])
                # note that sometimes qhull gives the same vertex twice!
                for v in p_proj.vertices:
                    self.assertTrue(any([np.allclose(v, v_ve) for v_ve in p_proj_ve.vertices]))
                for v_ve in p_proj_ve.vertices:
                    self.assertTrue(any([np.allclose(v, v_ve) for v in p_proj.vertices]))


    def test_convex_hull(self):

        # first hull with 4 points
        points = [np.array([[1.],[0.],[0.]]), np.array([[-1.],[0.],[0.]]), np.array([[0.],[1.],[0.]]), np.array([[0.],[0.],[1.]])]
        A_real = np.array([
            [1., 1., 1.]/np.sqrt(3.),
            [-1., 1., 1.]/np.sqrt(3.),
            [0., 0., -1.],[0., -1., 0.]
            ])
        b_real = np.array([
            [1./np.sqrt(3.)],
            [1./np.sqrt(3.)],
            [0.],
            [0.]
            ])

        # test Hrep
        hull = ConvexHull(points)
        Ab_real = np.hstack((A_real, b_real))
        Ab = np.hstack((hull.A, hull.b))
        self.assertEqual(Ab_real.shape[0], Ab.shape[0])
        for ab_real in Ab_real:
            self.assertTrue(any([np.allclose(ab_real, ab) for ab in Ab]))
        for ab in Ab:
            self.assertTrue(any([np.allclose(ab_real, ab) for ab_real in Ab_real]))

        # add point on surface and inside (nothing is supposed happen...)
        for point in [np.array([[0.],[0.],[.5]]), np.array([[0.],[.1],[.5]])]:
            hull.add_point(point)
            Ab = np.hstack((hull.A, hull.b))
            for ab_real in Ab_real:
                self.assertTrue(any([np.allclose(ab_real, ab) for ab in Ab]))
            for ab in Ab:
                self.assertTrue(any([np.allclose(ab_real, ab) for ab_real in Ab_real]))

        # add coplanar point outside
        point = np.array([[0.],[0.],[-1.]])
        hull.add_point(point)
        Ab = np.hstack((hull.A, hull.b))
        A_real = np.array([
            [1., 1., 1.]/np.sqrt(3.),
            [-1., 1., 1.]/np.sqrt(3.),
            [1., 1., -1.]/np.sqrt(3.),
            [-1., 1., -1.]/np.sqrt(3.),
            [0., -1., 0.]
            ])
        b_real = np.array([
            [1./np.sqrt(3.)],
            [1./np.sqrt(3.)],
            [1./np.sqrt(3.)],
            [1./np.sqrt(3.)],
            [0.]])
        Ab_real = np.hstack((A_real, b_real))
        Ab = np.hstack((hull.A, hull.b))
        points.append(point)
        for ab_real in Ab_real:
            self.assertTrue(any([np.allclose(ab_real, ab) for ab in Ab]))

        # test minimal H-rep
        Ab = np.hstack(hull.get_minimal_H_rep())
        self.assertEqual(Ab_real.shape[0], Ab.shape[0])
        for ab_real in Ab_real:
            self.assertTrue(any([np.allclose(ab_real, ab) for ab in Ab]))
        for ab in Ab:
            self.assertTrue(any([np.allclose(ab_real, ab) for ab_real in Ab_real]))

        # add point outside
        point = np.array([[0.],[-1.],[0.]])
        hull.add_point(point)
        Ab = np.hstack((hull.A, hull.b))
        A_real = np.array([
            [1., 1., 1.],
            [-1., 1., 1.],
            [1., -1., 1.],
            [1., 1., -1.],
            [-1., -1., 1.],
            [1., -1., -1.],
            [-1., 1., -1.],
            [-1., -1., -1.]
            ])/np.sqrt(3.)
        b_real = np.ones((8,1))/np.sqrt(3.)
        Ab_real = np.hstack((A_real, b_real))
        Ab = np.hstack((hull.A, hull.b))
        points.append(point)
        for ab_real in Ab_real:
            self.assertTrue(any([np.allclose(ab_real, ab) for ab in Ab]))
        for ab in Ab:
            self.assertTrue(any([np.allclose(ab_real, ab) for ab_real in Ab_real]))

if __name__ == '__main__':
    unittest.main()