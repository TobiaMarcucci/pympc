# external imports
import unittest
import numpy as np

# internal inputs
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

    def test_is_empty(self):

        # full dimensional
        x_min = 1.*np.ones((2,1))
        x_max = 2.*np.ones((2,1))
        p = Polyhedron.from_bounds(x_min, x_max)
        self.assertFalse(p.is_empty())

        # lower dimensional, but not empy
        C = np.ones((1, 2))
        d = np.array([[3.]])
        p.add_equality(C, d)
        self.assertFalse(p.is_empty())

        # lower dimensional and empty
        x_0_max = np.array([[.5]])
        p.add_upper_bound(x_0_max, [0])
        self.assertTrue(p.is_empty())

    def test_is_bounded(self):

        # bounded (empty), easy (ker(A) empty)
        x_min = np.ones((2, 1))
        x_max = - np.ones((2, 1))
        p = Polyhedron.from_bounds(x_min, x_max)
        self.assertTrue(p.is_bounded())

        # bounded (empty), tricky (ker(A) not empty)
        x0_min = np.array([[1.]])
        x0_max = np.array([[-1.]])
        p = Polyhedron.from_bounds(x0_min, x0_max, [0], 2)
        self.assertTrue(p.is_bounded())

        # bounded easy
        x_min = 1.*np.ones((2,1))
        x_max = 2.*np.ones((2,1))
        p = Polyhedron.from_bounds(x_min, x_max)
        self.assertTrue(p.is_bounded())

        # unbounded, halfspace
        x0_min = np.array([[0.]])
        p = Polyhedron.from_lower_bound(x0_min, [0], 2)
        self.assertFalse(p.is_bounded())

        # unbounded, positive orthant
        x1_min = x0_min
        p.add_lower_bound(x1_min, [1])
        self.assertFalse(p.is_bounded())

        # unbounded: parallel inequalities, slice of positive orthant
        x0_max = np.array([[1.]])
        p.add_upper_bound(x0_max, [0])
        self.assertFalse(p.is_bounded())

        # unbounded lower dimensional, line in the positive orthant
        x0_min = x0_max
        p.add_lower_bound(x0_min, [0])
        self.assertFalse(p.is_bounded())

        # bounded lower dimensional, segment in the positive orthant
        x1_max = np.array([[1000.]])
        p.add_upper_bound(x1_max, [1])
        self.assertTrue(p.is_bounded())

    def test_contains(self):

        # some points
        x1 = 2.*np.ones((2,1))
        x2 = 2.5*np.ones((2,1))
        x3 = 3.5*np.ones((2,1))

        # full dimensional
        x_min = 1.*np.ones((2,1))
        x_max = 3.*np.ones((2,1))
        p = Polyhedron.from_bounds(x_min, x_max)
        self.assertTrue(p.contains(x1))
        self.assertTrue(p.contains(x2))
        self.assertFalse(p.contains(x3))

        # lower dimensional, but not empty
        C = np.ones((1, 2))
        d = np.array([[4.]])
        p.add_equality(C, d)
        self.assertTrue(p.contains(x1))
        self.assertFalse(p.contains(x2))
        self.assertFalse(p.contains(x3))

    def test_is_included_in(self):

        # inner polyhdron
        x_min = 1.*np.ones((2,1))
        x_max = 2.*np.ones((2,1))
        p1 = Polyhedron.from_bounds(x_min, x_max)

        # outer polyhdron
        x_min = .5*np.ones((2,1))
        x_max = 2.5*np.ones((2,1))
        p2 = Polyhedron.from_bounds(x_min, x_max)

        # check inclusions
        self.assertTrue(p1.is_included_in(p1))
        self.assertTrue(p1.is_included_in(p2))
        self.assertFalse(p2.is_included_in(p1))

        # polytope that intesects p1
        x_min = .5*np.ones((2,1))
        x_max = 1.5*np.ones((2,1))
        p2 = Polyhedron.from_bounds(x_min, x_max)
        self.assertFalse(p1.is_included_in(p2))
        self.assertFalse(p2.is_included_in(p1))

        # polytope that includes p1, with two equal facets
        x_min = .5*np.ones((2,1))
        x_max = 2.*np.ones((2,1))
        p2 = Polyhedron.from_bounds(x_min, x_max)
        self.assertTrue(p1.is_included_in(p2))
        self.assertFalse(p2.is_included_in(p1))

        # polytope that does not include p1, with two equal facets
        x_min = 1.5*np.ones((2,1))
        p2.add_lower_bound(x_min)
        self.assertFalse(p1.is_included_in(p2))
        self.assertTrue(p2.is_included_in(p1))

        # with equalities
        C = np.ones((1, 2))
        d = np.array([[3.5]])
        p1.add_equality(C, d)
        self.assertTrue(p1.is_included_in(p2))
        self.assertFalse(p2.is_included_in(p1))
        p2.add_equality(C, d)
        self.assertTrue(p1.is_included_in(p2))
        self.assertTrue(p2.is_included_in(p1))
        x1_max = np.array([[1.75]])
        p2.add_upper_bound(x1_max, [1])
        self.assertFalse(p1.is_included_in(p2))
        self.assertTrue(p2.is_included_in(p1))

if __name__ == '__main__':
    unittest.main()