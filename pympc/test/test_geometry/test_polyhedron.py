import unittest
import numpy as np
from pympc.geometry.polyhedron import Polyhedron

class TestPolyhedron(unittest.TestCase):

    def test_initialization(self):

    	# 1 or 2d right hand sides
    	A = np.eye(3)
    	C = np.eye(2)
    	b = np.ones((3,1))
    	b_1d = np.ones(3)
    	d = np.ones((2,1))
    	d_1d = np.ones(2)
    	p = Polyhedron(A, b, C, d)
    	p_1d = Polyhedron(A, b_1d, C, d_1d)
    	np.testing.assert_array_equal(p.b, p_1d.b)
    	np.testing.assert_array_equal(p.d, p_1d.d)

        # wrong initializations
        self.assertRaises(ValueError, Polyhedron, A, b, C)
        self.assertRaises(ValueError, Polyhedron, A, d)
        self.assertRaises(ValueError, Polyhedron, A, b, C, b)

    def test_add_functions(self):

    	# add inequalities
        A = np.eye(2)
        b = np.ones(2)
        p = Polyhedron(A, b)
        A = np.ones((1, 2))
        b = np.ones((1, 1))
        p.add_inequality(A, b)
        np.testing.assert_array_equal(
        	p.A,
        	np.array([[1., 0.], [0., 1.], [1., 1.]])
        	)
        np.testing.assert_array_equal(
        	p.b,
        	np.ones((3, 1))
        	)
        c = np.ones(2)
        self.assertRaises(ValueError, p.add_inequality, A, c)

    	# add equalities
        p.add_equality(A, b)
        np.testing.assert_array_equal(p.C, A)
        np.testing.assert_array_equal(p.d, b)
        b = np.ones(2)
        self.assertRaises(ValueError, p.add_equality, A, b)

        # add lower bounds
        A = np.zeros((0, 2))
        b = np.zeros((0, 1))
        p = Polyhedron(A, b)
        p.add_lower_bound(-np.ones((2,1)))
        p.add_upper_bound(2*np.ones((2,1)))
        p.add_bounds(-3*np.ones((2,1)), 3*np.ones((2,1)))
        p.add_lower_bound(-4*np.ones((1,1)), [0])
        p.add_upper_bound(5*np.ones((1,1)), [0])
        p.add_bounds(-6*np.ones((1,1)), 6*np.ones((1,1)), [1])
        A =np.array([[-1., 0.], [0., -1.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 0.], [0., 1.], [-1., 0.], [1., 0.], [0., -1.], [0., 1.] ])
        b = np.array([[1.], [1.], [2.], [2.], [3.], [3.], [3.], [3.], [4.], [5.], [6.], [6.] ])
        np.testing.assert_array_equal(p.A, A)
        np.testing.assert_array_equal(p.b, b)

        # wrong size bounds
        A = np.eye(3)
        b = np.ones(3)
        p = Polyhedron(A, b)
        x = np.zeros((1,1))
        indices = [0, 1]
        self.assertRaises(ValueError, p.add_lower_bound, x, indices)
        self.assertRaises(ValueError, p.add_upper_bound, x, indices)
        self.assertRaises(ValueError, p.add_bounds, x, x, indices)

    def test_from_functions(self):

        # row vector input
        x = np.ones((1,5))
        self.assertRaises(ValueError, Polyhedron.from_lower_bound, x)

        # from lower bound
        x = np.ones((2,1))
        p = Polyhedron.from_lower_bound(x)
        A_lb = np.array([[-1., 0.], [0., -1.]])
        b_lb = np.array([[-1.], [-1.]])
        np.testing.assert_array_equal(p.A, A_lb)
        np.testing.assert_array_equal(p.b, b_lb)

        # from upper bound
        p = Polyhedron.from_upper_bound(x)
        A_ub = np.array([[1., 0.], [0., 1.]])
        b_ub = np.array([[1.], [1.]])
        np.testing.assert_array_equal(p.A, A_ub)
        np.testing.assert_array_equal(p.b, b_ub)

        # from upper and lower bounds
        p = Polyhedron.from_bounds(x, x)
        A = np.vstack((A_lb, A_ub))
        b = np.vstack((b_lb, b_ub))
        np.testing.assert_array_equal(p.A, A)
        np.testing.assert_array_equal(p.b, b)

        # different size lower and upper bound
        y = np.ones((3,1))
        self.assertRaises(ValueError, Polyhedron.from_bounds, x, y)

        # from lower bound of not all the variables
        indices = [1, 2]
        n = 3
        p = Polyhedron.from_lower_bound(x, indices, n)
        A_lb = np.array([[0., -1., 0.], [0., 0., -1.]])
        b_lb = np.array([[-1.], [-1.]])
        np.testing.assert_array_equal(p.A, A_lb)
        np.testing.assert_array_equal(p.b, b_lb)

        # from lower bound of not all the variables
        indices = [0, 2]
        n = 3
        p = Polyhedron.from_upper_bound(x, indices, n)
        A_ub = np.array([[1., 0., 0.], [0., 0., 1.]])
        b_ub = np.array([[1.], [1.]])
        np.testing.assert_array_equal(p.A, A_ub)
        np.testing.assert_array_equal(p.b, b_ub)

        # from upper and lower bounds of not all the variables
        indices = [0, 1]
        n = 3
        p = Polyhedron.from_bounds(x, x, indices, n)
        A = np.array([[-1., 0., 0.], [0., -1., 0.], [1., 0., 0.], [0., 1., 0.]])
        b = np.array([[-1.], [-1.], [1.], [1.]])
        np.testing.assert_array_equal(p.A, A)
        np.testing.assert_array_equal(p.b, b)

        # too many indices
        indices = [1, 2, 3]
        n = 5
        self.assertRaises(ValueError, Polyhedron.from_lower_bound, x, indices, n)
        self.assertRaises(ValueError, Polyhedron.from_upper_bound, x, indices, n)
        self.assertRaises(ValueError, Polyhedron.from_bounds, x, x, indices, n)

    def test_normalize(self):

        # construct polyhedron
        A = np.array([[0., 2.], [0., -3.]])
        b = np.array([[2.], [0.]])
        C = np.ones((1, 2))
        d = np.ones((1, 1))
        p = Polyhedron(A, b, C, d)

        # normalize
        p.normalize()
        A = np.array([[0., 1.], [0., -1.]])
        b = np.array([[1.], [0.]])
        C = np.ones((1, 2))/np.sqrt(2.)
        d = np.ones((1, 1))/np.sqrt(2.)
        np.testing.assert_array_almost_equal(p.A, A)
        np.testing.assert_array_almost_equal(p.b, b)
        np.testing.assert_array_almost_equal(p.C, C)
        np.testing.assert_array_almost_equal(p.d, d)
        
    def test_remove_equalities(self):

        # contruct polyhedron
        x_min = - np.ones((2, 1))
        x_max = - x_min
        p = Polyhedron.from_bounds(x_min, x_max)
        C = np.ones((1, 2))
        d = np.ones((1, 1))
        p.add_equality(C, d)
        E, f, Y, Z = p._remove_equalities()

        # check result
        self.assertAlmostEqual(Z[0,0]/Z[1,0], -1)
        self.assertAlmostEqual(Y[0,0]/Y[1,0], 1)
        intersections = [
            np.sqrt(2.)/2,
            3.*np.sqrt(2.)/2,
            - np.sqrt(2.)/2,
            - 3.*np.sqrt(2.)/2
            ]
        for z in intersections:
            r = E.dot(np.array([[z]])) - f
            self.assertAlmostEqual(np.min(np.abs(r)), 0.)

    def test_remove_redundant_inequalities(self):

        # only inequalities
        A = np.array([[1., 1.], [-1., 1.], [0., -1.], [0., 1.], [0., 1.], [2., 2.]])
        b = np.array([[1.], [1.], [1.], [1.], [2.], [2.]])
        p = Polyhedron(A,b)
        self.assertEqual(
            [1, 2, 5],
            sorted(p.get_minimal_facets())
            )
        p.remove_redundant_inequalities()
        A_min = np.array([[-1., 1.], [0., -1.], [2., 2.]])
        b_min = np.array([[1.], [1.], [2.]])
        np.testing.assert_array_almost_equal(p.A, A_min)
        np.testing.assert_array_almost_equal(p.b, b_min)

        # both inequalities and equalities
        x_min = - np.ones((2, 1))
        x_max = - x_min
        p = Polyhedron.from_bounds(x_min, x_max)
        C = np.ones((1, 2))
        d = np.ones((1, 1))
        p.add_equality(C, d)
        self.assertEqual(
            [2, 3],
            sorted(p.get_minimal_facets())
            )
        p.remove_redundant_inequalities()
        A_min = np.array([[1., 0.], [0., 1.]])
        b_min = np.array([[1.], [1.]])
        np.testing.assert_array_almost_equal(p.A, A_min)
        np.testing.assert_array_almost_equal(p.b, b_min)

        # add (redundant) inequality coincident with the equality
        p.add_inequality(C, d)
        p.remove_redundant_inequalities()
        np.testing.assert_array_almost_equal(p.A, A_min)
        np.testing.assert_array_almost_equal(p.b, b_min)

        # add (redundant) inequality
        p.add_inequality(
            np.array([[-1., 1.]]),
            np.array([[1.1]])
            )
        p.remove_redundant_inequalities()
        np.testing.assert_array_almost_equal(p.A, A_min)
        np.testing.assert_array_almost_equal(p.b, b_min)

if __name__ == '__main__':
    unittest.main()