# external imports
import unittest
import numpy as np

# internal inputs
from pympc.optimization.linear_program import LinearProgram
from pympc.geometry.polyhedron import Polyhedron

class TestLinearProgram(unittest.TestCase):

    def test_solve(self):

    	# trivial LP with only inequalities
        A = -np.eye(2)
        b = np.zeros((2, 1))
        X = Polyhedron(A, b)
        f = np.ones((2, 1))
        lp = LinearProgram(X, f)
        sol = lp.solve()
        self.assertAlmostEqual(
            sol['min'],
            0.
            )
        np.testing.assert_array_almost_equal(
            sol['argmin'],
            np.zeros((2, 1))
            )
        self.assertEqual(
            sol['active_set'],
            [0,1]
            )
        np.testing.assert_array_almost_equal(
            sol['multiplier_inequality'],
            np.ones((2, 1))
            )
        self.assertTrue(
            sol['multiplier_equality'] is None
            )

        # add equality
        C = np.array([[2., 1.]])
        d = np.array([[2.]])
        lp.X.add_equality(C, d)
        sol = lp.solve()
        self.assertAlmostEqual(
            sol['min'],
            1.
            )
        np.testing.assert_array_almost_equal(
            sol['argmin'],
            np.array([[1.], [0.]])
            )
        self.assertEqual(
            sol['active_set'],
            [1]
            )
        np.testing.assert_array_almost_equal(
            sol['multiplier_inequality'],
            np.array([[0.], [.5]])
            )
        np.testing.assert_array_almost_equal(
            sol['multiplier_equality'],
            np.array([[-.5]])
            )

    def test_solve_min_norm(self):

    	# simple 2d problem
    	x0_min = np.zeros((1,1))
    	X = Polyhedron.from_lower_bound(x0_min, [0], 2)
    	C = - np.array(([[1., 10.]]))
    	d = - np.array(([[10.]]))
    	X.add_equality(C, d)
    	lp = LinearProgram(X)

    	# norm one, identity weight
    	sol_one = lp.solve_min_norm_one()
    	self.assertAlmostEqual(
            sol_one['min'],
            1.
            )
        np.testing.assert_array_almost_equal(
            sol_one['argmin'],
            np.array([[0.], [1.]])
            )
        self.assertEqual(
            sol_one['active_set'],
            [0]
            )
        np.testing.assert_array_almost_equal(
            sol_one['multiplier_inequality'],
            np.array([[0.]])
            )
        np.testing.assert_array_almost_equal(
            sol_one['multiplier_equality'],
            np.array([[.1]])
            )

        # norm inf, identity weight
    	sol_inf = lp.solve_min_norm_inf()
    	self.assertAlmostEqual(
            sol_inf['min'],
            10./11.
            )
        np.testing.assert_array_almost_equal(
            sol_inf['argmin'],
            np.array([[10./11.], [10./11.]])
            )
        self.assertEqual(
            sol_inf['active_set'],
            []
            )
        np.testing.assert_array_almost_equal(
            sol_inf['multiplier_inequality'],
            np.array([[0.]])
            )
        np.testing.assert_array_almost_equal(
            sol_inf['multiplier_equality'],
            np.array([[1./11.]])
            )

        # norm one, with weight matrix
        W = np.array([[1., 0.],[0., 11.]])
    	sol_one = lp.solve_min_norm_one(W)
    	self.assertAlmostEqual(
            sol_one['min'],
            10.
            )
        np.testing.assert_array_almost_equal(
            sol_one['argmin'],
            np.array([[10.], [0.]])
            )
        self.assertEqual(
            sol_one['active_set'],
            []
            )
        np.testing.assert_array_almost_equal(
            sol_one['multiplier_inequality'],
            np.array([[0.]])
            )
        np.testing.assert_array_almost_equal(
            sol_one['multiplier_equality'],
            np.array([[1.]])
            )

        # norm inf, with weight matrix
    	sol_inf = lp.solve_min_norm_inf(W)
    	self.assertAlmostEqual(
            sol_inf['min'],
            110./21.
            )
        np.testing.assert_array_almost_equal(
            sol_inf['argmin'],
            np.array([[110./21.], [10./21.]])
            )
        self.assertEqual(
            sol_inf['active_set'],
            []
            )
        np.testing.assert_array_almost_equal(
            sol_inf['multiplier_inequality'],
            np.array([[0.]])
            )
        np.testing.assert_array_almost_equal(
            sol_inf['multiplier_equality'],
            np.array([[11./21.]])
            )

if __name__ == '__main__':
    unittest.main()