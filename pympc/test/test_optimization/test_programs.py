# external imports
import unittest
import numpy as np

# internal inputs
from pympc.optimization.programs import linear_program, quadratic_program, mixed_integer_quadratic_program

class TestPrograms(unittest.TestCase):
    """
    This tests only if the solvers are called correctly,.
    More tests on the correctness of the results can be fuond in test_solvers.py.
    """

    def test_linear_program(self, solver='pnnls'):

        # trivial LP with only inequalities
        A = -np.eye(2)
        b = np.zeros(2)
        f = np.ones(2)
        sol = linear_program(f, A, b, solver=solver)
        self.assertAlmostEqual(
            sol['min'],
            0.
            )
        np.testing.assert_array_almost_equal(
            sol['argmin'],
            np.zeros(2)
            )
        self.assertEqual(
            sol['active_set'],
            [0,1]
            )
        np.testing.assert_array_almost_equal(
            sol['multiplier_inequality'],
            np.ones(2)
            )
        self.assertTrue(
            sol['multiplier_equality'] is None
            )

        # add equality
        C = np.array([[2., 1.]])
        d = np.array([2.])
        sol = linear_program(f, A, b, C, d, solver=solver)
        self.assertAlmostEqual(
            sol['min'],
            1.
            )
        np.testing.assert_array_almost_equal(
            sol['argmin'],
            np.array([1.,0.])
            )
        self.assertEqual(
            sol['active_set'],
            [1]
            )
        np.testing.assert_array_almost_equal(
            sol['multiplier_inequality'],
            np.array([0.,.5])
            )
        np.testing.assert_array_almost_equal(
            sol['multiplier_equality'],
            np.array([-.5])
            )

    def test_linear_program_gurobi(self):

        # use gurobi to solve the mathematical programs
        self.test_linear_program(solver='gurobi')

    def test_quadratic_program(self, solver='pnnls'):

        # trivial QP with only inequalities
        H = np.eye(2)
        f = np.ones(2)
        A = -np.eye(2)
        b = -np.ones(2)
        sol = quadratic_program(H, f, A, b, solver=solver)
        self.assertAlmostEqual(
            sol['min'],
            3.
            )
        np.testing.assert_array_almost_equal(
            sol['argmin'],
            np.array([1.,1.])
            )
        self.assertEqual(
            sol['active_set'],
            [0,1]
            )
        np.testing.assert_array_almost_equal(
            sol['multiplier_inequality'],
            np.array([2.,2.])
            )
        self.assertTrue(
            sol['multiplier_equality'] is None
            )

        # add equality constraints
        C = np.array([[0., 1.]])
        d = np.array([2.])
        sol = quadratic_program(H, f, A, b, C, d, solver=solver)
        self.assertAlmostEqual(
            sol['min'],
            5.5
            )
        np.testing.assert_array_almost_equal(
            sol['argmin'],
            np.array([1.,2.])
            )
        self.assertEqual(
            sol['active_set'],
            [0]
            )
        np.testing.assert_array_almost_equal(
            sol['multiplier_inequality'],
            np.array([2.,0.])
            )
        np.testing.assert_array_almost_equal(
            sol['multiplier_equality'],
            np.array([-3.])
            )

    def test_quadratic_program_gurobi(self):

        # use gurobi to solve the mathematical programs
        self.test_quadratic_program(solver='gurobi')

    def test_mixed_integer_quadratic_program_gurobi(self):

        # select solvers
        solver = 'gurobi'

        # simple miqp
        H = np.eye(2)
        f = np.array([[0.],[-.6]])
        nc = 1
        A = np.eye(2)
        b = 2.*np.ones((2,1))
        sol = mixed_integer_quadratic_program(nc, H, f, A, b, solver=solver)
        self.assertAlmostEqual(
            sol['min'],
            -.1
            )
        np.testing.assert_array_almost_equal(
            sol['argmin'],
            np.array([0.,1.])
            )

        # add equalities
        C = np.array([[1., -1.]])
        d = np.zeros((1,1))
        sol = mixed_integer_quadratic_program(nc, H, f, A, b, C, d, solver=solver)
        self.assertAlmostEqual(
            sol['min'],
            0.
            )
        np.testing.assert_array_almost_equal(
            sol['argmin'],
            np.zeros(2)
            )

        # unfeasible miqp
        C = np.ones((1,2))
        d = 5.*np.ones((1,1))
        sol = mixed_integer_quadratic_program(nc, H, f, A, b, C, d, solver=solver)
        self.assertTrue(sol['min'] is None)
        self.assertTrue(sol['argmin'] is None)

if __name__ == '__main__':
    unittest.main()