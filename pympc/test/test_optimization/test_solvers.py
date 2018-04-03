# external imports
import unittest
import numpy as np

# internal inputs
from pympc.optimization.solvers.pnnls import linear_program as lp_pnnls, quadratic_program as qp_pnnls
from pympc.optimization.solvers.gurobi import linear_program as lp_gurobi, quadratic_program as qp_gurobi, mixed_integer_quadratic_program as miqp_gurobi
#from pympc.optimization.solvers.drake import linear_program as lp_drake, quadratic_program as qp_drake, mixed_integer_quadratic_program as miqp_drake

class TestSolvers(unittest.TestCase):

    def test_linear_program(self):

        # loop over solvers
        for linear_program in [lp_pnnls, lp_gurobi]:

            # trivial lp with only inequalities
            f = np.ones((2, 1))
            A = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
            b = np.array([[5.],[5.],[1.],[1.]])
            sol = linear_program(f, A, b)
            self.assertAlmostEqual(
                sol['min'],
                -6.
                )
            np.testing.assert_array_almost_equal(
                sol['argmin'],
                np.array([[-5.],[-1.]])
                )
            self.assertEqual(
                sol['active_set'],
                [1,3]
                )
            np.testing.assert_array_almost_equal(
                sol['multiplier_inequality'],
                np.array([[0.],[1.],[0.],[1.]])
                )
            self.assertTrue(
                sol['multiplier_equality'] is None
                )

            # another trivial LP with only inequalities
            A = -np.eye(2)
            b = np.zeros((2, 1))
            sol = linear_program(f, A, b)
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
            sol = linear_program(f, A, b, C, d)
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

            # unfeasible lp
            A = np.array([[0.,1.],[0.,-1.]])
            b = np.array([[0.],[-1.]])
            sol = linear_program(f, A, b)
            for value in sol.values():
                self.assertTrue(value is None)

            # unbounded lp
            A = np.array([[1.,0.],[0.,1.],[0.,-1.]])
            b = np.array([[0.],[0.],[1.]])
            sol = linear_program(f, A, b)
            for value in sol.values():
                self.assertTrue(value is None)

            # add equalities (still unbounded)
            C = np.array([[0., 1.]])
            d = np.array([[-.1]])
            sol = linear_program(f, A, b, C, d)
            for value in sol.values():
                self.assertTrue(value is None)

            # add equalities (now bounded)
            C = np.array([[1., -1.]])
            d = np.array([[0.]])
            sol = linear_program(f, A, b, C, d)
            self.assertAlmostEqual(
                sol['min'],
                -2.
                )
            np.testing.assert_array_almost_equal(
                sol['argmin'],
                - np.ones((2,1))
                )
            self.assertEqual(
                sol['active_set'],
                [2]
                )
            np.testing.assert_array_almost_equal(
                sol['multiplier_inequality'],
                np.array([[0.], [0.], [2.]])
                )
            np.testing.assert_array_almost_equal(
                sol['multiplier_equality'],
                np.array([[-1.]])
                )

            # bounded LP with unbounded domain
            f = np.array([[0.],[1.]])
            A = np.array([[0.,-1.]])
            b = np.zeros((1,1))
            sol = linear_program(f, A, b)
            self.assertAlmostEqual(
                sol['min'],
                0.
                )
            self.assertAlmostEqual(
                sol['argmin'][1,0],
                0.
                )
            self.assertEqual(
                sol['active_set'],
                [0]
                )
            np.testing.assert_array_almost_equal(
                sol['multiplier_inequality'],
                np.array([[1.]])
                )
            self.assertTrue(
                sol['multiplier_equality'] is None
                )

            # 3d LPs (constraint set: min_x -x_1 s.t. ||x||_1 <= 1, x_1 <= .5)
            A = np.array([
                  [1., 1., 1.],
                  [-1., 1., 1.],
                  [1., -1., 1.],
                  [1., 1., -1.],
                  [-1., -1., 1.],
                  [1., -1., -1.],
                  [-1., 1., -1.],
                  [-1., -1., -1.],
                  [1., 0., 0.]
                  ])
            b = np.vstack((np.ones((8,1)), np.array([[.5]])))
            f = np.array([[-1.],[0.],[0.]])
            sol = linear_program(f, A, b)
            self.assertAlmostEqual(
                sol['min'],
                -.5
                )
            self.assertAlmostEqual(
                sol['argmin'][0,0],
                .5
                )
            self.assertTrue(8 in sol['active_set'])
            np.testing.assert_array_almost_equal(
                sol['multiplier_inequality'],
                np.vstack((np.zeros((8,1)), np.ones((1,1))))
                )
            self.assertTrue(
                sol['multiplier_equality'] is None
                )

            # new cost
            f = np.array([[-1.],[-.1],[0.]])
            sol = linear_program(f, A, b)
            self.assertAlmostEqual(
                sol['min'],
                -.55
                )
            np.testing.assert_array_almost_equal(
                sol['argmin'],
                np.array([[.5],[.5],[0.]])
                )
            self.assertEqual(
                sol['active_set'],
                [0,3,8]
                )
            np.testing.assert_array_almost_equal(
                sol['multiplier_inequality'],
                np.array([[.05],[0.],[0.],[.05],[0.],[0.],[0.],[0.],[.9]])
                )
            self.assertTrue(
                sol['multiplier_equality'] is None
                )

            # new cost
            f = np.array([[1.],[1.],[0.]])
            sol = linear_program(f, A, b)
            self.assertAlmostEqual(
                sol['min'],
                -1.
                )
            self.assertAlmostEqual(
                sol['argmin'][0,0] + sol['argmin'][1,0],
                -1.
                )
            self.assertTrue(4 in sol['active_set'])
            self.assertTrue(7 in sol['active_set'])
            self.assertAlmostEqual(sol['multiplier_inequality'][4,0], .5)
            self.assertAlmostEqual(sol['multiplier_inequality'][7,0], .5)
            self.assertTrue(
                sol['multiplier_equality'] is None
                )

            # lower dimensional domain because of the inequalities (in this case one cannot get the active set just looking at the residuals of the inequalities)
            f = np.array([[1.],[0.]])
            A = np.array([[1., 0.],[-1., 0.]])
            b = np.zeros((2,1))
            sol = linear_program(f, A, b)
            self.assertAlmostEqual(
                sol['min'],
                0.
                )
            self.assertAlmostEqual(
                sol['argmin'][0,0],
                0.
                )
            self.assertEqual(
                sol['active_set'],
                [1]
                )
            np.testing.assert_array_almost_equal(
                sol['multiplier_inequality'],
                np.array([[0.],[1.]])
                )
            self.assertTrue(
                sol['multiplier_equality'] is None
                )

    def test_quadratic_program(self):

        # loop over solvers
        for quadratic_program in [qp_pnnls, qp_gurobi]:

            # trivial qp with only inequalities
            H = np.eye(2)
            f = np.ones((2, 1))
            A = -np.eye(2)
            b = -np.ones((2, 1))
            sol = quadratic_program(H, f, A, b)
            self.assertAlmostEqual(
                sol['min'],
                3.
                )
            np.testing.assert_array_almost_equal(
                sol['argmin'],
                np.array([[1.],[1.]])
                )
            self.assertEqual(
                sol['active_set'],
                [0,1]
                )
            np.testing.assert_array_almost_equal(
                sol['multiplier_inequality'],
                np.array([[2.],[2.]])
                )
            self.assertTrue(
                sol['multiplier_equality'] is None
                )

            # add equality constraints
            C = np.array([[0., 1.]])
            d = np.array([[2.]])
            sol = quadratic_program(H, f, A, b, C, d)
            self.assertAlmostEqual(
                sol['min'],
                5.5
                )
            np.testing.assert_array_almost_equal(
                sol['argmin'],
                np.array([[1.],[2.]])
                )
            self.assertEqual(
                sol['active_set'],
                [0]
                )
            np.testing.assert_array_almost_equal(
                sol['multiplier_inequality'],
                np.array([[2.],[0.]])
                )
            np.testing.assert_array_almost_equal(
                sol['multiplier_equality'],
                np.array([[-3.]])
                )

            # unfeasible
            sol = quadratic_program(H, f, A, b, C, -d)
            for value in sol.values():
                self.assertTrue(value is None)

            # lower dimensional domain because of the inequalities (in this case one cannot get the active set just looking at the residuals of the inequalities)
            H = np.eye(2)
            f = np.zeros((2,1))
            A = np.array([[1., 0.],[-1., 0.]])
            b = np.array([[1.],[-1.]])
            sol = quadratic_program(H, f, A, b)
            self.assertAlmostEqual(
                sol['min'],
                .5
                )
            self.assertAlmostEqual(
                sol['argmin'][0,0],
                1.
                )
            self.assertEqual(
                sol['active_set'],
                [1]
                )
            np.testing.assert_array_almost_equal(
                sol['multiplier_inequality'],
                np.array([[0.],[1.]])
                )
            self.assertTrue(
                sol['multiplier_equality'] is None
                )

    def test_mixed_integer_quadratic_program(self):

        # loop over solvers
        for mixed_integer_quadratic_program in [miqp_gurobi]:

            # simple miqp
            H = np.eye(2)
            f = np.array([[0.],[-.6]])
            nc = 1
            A = np.eye(2)
            b = 2.*np.ones((2,1))
            sol = mixed_integer_quadratic_program(nc, H, f, A, b)
            self.assertAlmostEqual(
                sol['min'],
                -.1
                )
            np.testing.assert_array_almost_equal(
                sol['argmin'],
                np.array([[0.],[1.]])
                )

            # add equalities
            C = np.array([[1., -1.]])
            d = np.zeros((1,1))
            sol = mixed_integer_quadratic_program(nc, H, f, A, b, C, d)
            self.assertAlmostEqual(
                sol['min'],
                0.
                )
            np.testing.assert_array_almost_equal(
                sol['argmin'],
                np.zeros((2,1))
                )

            # unfeasible miqp
            C = np.ones((1,2))
            d = 5.*np.ones((1,1))
            sol = mixed_integer_quadratic_program(nc, H, f, A, b, C, d)
            self.assertTrue(sol['min'] is None)
            self.assertTrue(sol['argmin'] is None)

if __name__ == '__main__':
    unittest.main()