import unittest
import numpy as np

import mpc_tools as mpc

# https://docs.python.org/2/library/unittest.html

class TestMPCTools(unittest.TestCase):

    def test_polytope(self):

        # unbounded 1d
        lhs = np.array([[1.]])
        rhs = np.array([[1.]])
        p = mpc.Polytope(lhs,rhs)
        with self.assertRaises(ValueError):
            p.assemble()

        # unbounded 2d
        lhs = np.array([[1.,0.],[-1.,0.],[0.,1.]])
        rhs = np.array([[1.],[1.],[0.]])
        p = mpc.Polytope(lhs,rhs)
        with self.assertRaises(ValueError):
            p.assemble()

        # empty 1d
        lhs = np.array([[1.],[-1.]])
        rhs = np.array([[1.],[-2.]])
        p = mpc.Polytope(lhs,rhs)
        p.assemble()
        self.assertTrue(p.empty)

        # empty 2d
        lhs = np.array([[1.,0.],[-1.,0.],[0.,1.]])
        rhs = np.array([[1.],[-2.],[0.]])
        p = mpc.Polytope(lhs,rhs)
        p.assemble()
        self.assertTrue(p.empty)

        # bounded and not empty 1d
        lhs = np.array([[1.],[-1.],[-3.]])
        rhs = np.array([[1.],[1.],[10.]])
        p = mpc.Polytope(lhs,rhs)
        p.assemble()
        self.assertFalse(p.empty)
        self.assertTrue(p.bounded)

        # bounded and not empty 2d
        lhs = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        rhs = np.array([[1.],[1.],[1.],[1.]])
        p = mpc.Polytope(lhs,rhs)
        p.assemble()
        self.assertFalse(p.empty)
        self.assertTrue(p.bounded)

        # coincident and minimal facets 1d
        lhs = np.array([[1.],[1.+1.e-9],[-1.]])
        rhs = np.ones((3,1))
        p = mpc.Polytope(lhs,rhs)
        p.assemble()
        self.assertEqual(set(p.coincident_facets[0]), set([0,1]))
        self.assertEqual(set(p.coincident_facets[1]), set([0,1]))
        self.assertEqual(p.coincident_facets[2], [2])
        self.assertTrue(p.minimal_facets[0] in [0,1])
        self.assertEqual(p.minimal_facets[1], 2)

        # coincident and minimal facets 2d
        lhs = np.array([[1.,0.],[1.+1.e-9,0.+1.e-9],[-1.,0.],[0.,1.],[0.,-1.]])
        rhs = np.ones((5,1))
        p = mpc.Polytope(lhs,rhs)
        p.assemble()
        self.assertEqual(set(p.coincident_facets[0]), set([0,1]))
        self.assertEqual(set(p.coincident_facets[1]), set([0,1]))
        self.assertEqual(p.coincident_facets[2:], [[2],[3],[4]])
        self.assertTrue(p.minimal_facets[0] in [0,1])
        self.assertEqual(p.minimal_facets[1:], [2,3,4])

        # add_ functions 1d
        x_max = np.ones((1,1))
        x_min = -x_max
        p = mpc.Polytope.from_bounds(x_max, x_min)
        p.add_bounds(x_max/2., x_min/5.)
        lhs = np.array([[1.],[-1.]])
        rhs = np.array([[.1],[1.]])
        p.add_facets(lhs, rhs)
        p.assemble()
        self.assertFalse(p.empty)
        self.assertTrue(p.bounded)

        # vertices 1d
        true_vertices = [[.1],[-.2]]
        self.assertTrue(all([any(np.all(np.isclose(vertex, true_vertices),axis=1)) for vertex in p.vertices]))

        # facet centers 1d
        true_facet_centers = true_vertices
        self.assertTrue(all([any(np.all(np.isclose(facet_center, true_facet_centers),axis=1)) for facet_center in p.facet_centers]))

        # minimal facets 1d
        true_minimal_facets = [3, 4]
        self.assertEqual(p.minimal_facets, true_minimal_facets)
        e = [(p.lhs_min[i,:]*p.facet_centers[i] - p.rhs_min[i])[0] for i in range(0,len(true_minimal_facets))]
        self.assertTrue(np.isclose(np.linalg.norm(e),0.))


        # add_ functions 2d
        x_max = np.ones((2,1))
        x_min = -x_max
        p = mpc.Polytope.from_bounds(x_max, x_min)
        p.add_bounds(x_max/2.,x_min/5.)
        lhs = np.array([[1.,0.],[-1.,0.]])
        rhs = np.array([[.1],[1.]])
        p.add_facets(lhs, rhs)
        p.assemble()
        self.assertFalse(p.empty)
        self.assertTrue(p.bounded)

        # vertices 2d
        true_vertices = [[.1,.5],[-.2,.5],[.1,-.2],[-.2,-.2]]
        self.assertTrue(all([any(np.all(np.isclose(vertex, true_vertices),axis=1)) for vertex in p.vertices]))

        # facet centers 2d
        true_facet_centers = [[-.05,.5],[-.2,.15],[-.05,-.2],[.1,.15]]
        self.assertTrue(all([any(np.all(np.isclose(facet_center, true_facet_centers),axis=1)) for facet_center in p.facet_centers]))

        # minimal facets 2d
        true_minimal_facets = [5,6,7,8]
        self.assertEqual(p.minimal_facets, true_minimal_facets)
        e = [(p.lhs_min[i,:].dot(p.facet_centers[i]) - p.rhs_min[i])[0] for i in range(0,len(true_minimal_facets))]
        self.assertTrue(np.isclose(np.linalg.norm(e),0.))

if __name__ == '__main__':
    unittest.main()