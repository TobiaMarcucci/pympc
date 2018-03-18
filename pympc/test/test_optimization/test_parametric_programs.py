# external imports
import unittest
import numpy as np

# internal inputs
from pympc.optimization.parametric_programs import MultiParametricQuadraticProgram
from pympc.geometry.utils import same_rows

class TestMultiParametricQuadraticProgram(unittest.TestCase):
    """
    These are just some very basic tests.
    The class MultiParametricQuadraticProgram is further tested with the ModelPredictiveController class.
    """

    def test_solve(self):

        # 1-dimensional mpqp
        H = dict()
        H['uu'] = np.array([[1.]])
        H['xx'] = np.zeros((1,1))
        H['ux'] = np.zeros((1,1))
        f = dict()
        f['u'] = np.zeros((1,1))
        f['x'] = np.zeros((1,1))
        g = np.zeros((1,1))
        A = dict()
        A['u'] = np.array([[1.],[-1.],[0.],[0.]])
        A['x'] = np.array([[-1.],[1.],[1.],[-1.]])
        b = np.array([[1.],[1.],[2.],[2.]])
        mpqp = MultiParametricQuadraticProgram(H, f, g, A, b)

        # explicit solve given active set
        cr = mpqp.explicit_solve_given_active_set([])
        self.assertAlmostEqual(cr._V['xx'], 0.)
        self.assertAlmostEqual(cr._V['x'], 0.)
        self.assertAlmostEqual(cr._V['0'], 0.)
        self.assertAlmostEqual(cr._u['x'], 0.)
        self.assertAlmostEqual(cr._u['0'], 0.)
        cr = mpqp.explicit_solve_given_active_set([0])
        self.assertAlmostEqual(cr._V['xx'], 1.)
        self.assertAlmostEqual(cr._V['x'], 1.)
        self.assertAlmostEqual(cr._V['0'], .5)
        self.assertAlmostEqual(cr._u['x'], 1.)
        self.assertAlmostEqual(cr._u['0'], 1.)

        # explicit solve given point
        cr = mpqp.explicit_solve_given_point(np.array([[1.5]]))
        self.assertAlmostEqual(cr._V['xx'], 1.)
        self.assertAlmostEqual(cr._V['x'], -1.)
        self.assertAlmostEqual(cr._V['0'], .5)
        self.assertAlmostEqual(cr._u['x'], 1.)
        self.assertAlmostEqual(cr._u['0'], -1.)

        # implicit solve given point
        sol = mpqp.solve(np.array([[1.5]]))
        self.assertAlmostEqual(sol['min'], .125)
        self.assertTrue(np.allclose(sol['argmin'], 0.5))
        self.assertEqual(sol['active_set'], [1])

        # solve
        exp_sol = mpqp.explicit_solve()
        for x in [np.array([[.5]]), np.array([[1.5]]), np.array([[-1.5]]), np.array([[2.5]]), np.array([[-2.5]])]:
            sol = mpqp.solve(x)
            if sol['min'] is not None:
                self.assertAlmostEqual(sol['min'], exp_sol.V(x))
                np.testing.assert_array_almost_equal(sol['argmin'], exp_sol.u(x))
                np.testing.assert_array_almost_equal(sol['multiplier_inequality'], exp_sol.p(x))
            else:
                self.assertTrue(exp_sol.V(x) is None)
                self.assertTrue(exp_sol.u(x) is None)
                self.assertTrue(exp_sol.p(x) is None)

        # feasible set
        fs = mpqp.get_feasible_set()
        A = np.array([[1.],[-1.]])
        b = np.array([[2.],[2.]])
        self.assertTrue(same_rows(
            np.hstack((A, b)),
            np.hstack((fs.A, fs.b))
            ))

if __name__ == '__main__':
    unittest.main()
