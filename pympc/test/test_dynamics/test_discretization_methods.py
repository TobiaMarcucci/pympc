# external imports
import unittest
import numpy as np

# internal inputs
from pympc.dynamics.discretization_methods import explicit_euler, zero_order_hold

class TestDiscretizationMethods(unittest.TestCase):

    def test_explicit_euler_and_zero_order_hold(self):
        np.random.seed(1)

        # double integrator
        A = np.array([[0., 1.],[0., 0.]])
        B = np.array([[0.],[1.]])
        c = np.zeros((2, 1))
        h = 1.

        # discrete time from continuous
        A_d, B_d, c_d = explicit_euler(A, B, c, h)
        np.testing.assert_array_almost_equal(A_d, np.eye(2) + A*h)
        np.testing.assert_array_almost_equal(B_d, B*h)
        np.testing.assert_array_almost_equal(c_d, c)

        # discrete time from continuous
        A_d, B_d, c_d = zero_order_hold(A, B, c, h)
        np.testing.assert_array_almost_equal(A_d, np.eye(2) + A*h)
        np.testing.assert_array_almost_equal(
            B_d,
            B*h + np.array([[0.,h**2/2.],[0.,0.]]).dot(B)
            )
        np.testing.assert_array_almost_equal(c_d, c)

        # test with random systems
        for i in range(100):
            n = np.random.randint(1, 10)
            m = np.random.randint(1, 10)
            A = np.random.rand(n,n)
            B = np.random.rand(n,m)
            c = np.random.rand(n,1)

            # reduce discretization step until the two method are almost equivalent
            h = .01
            convergence = False
            while not convergence:
                A_ee, B_ee, c_ee = explicit_euler(A, B, c, h)
                A_zoh, B_zoh, c_zoh = zero_order_hold(A, B, c, h)
                convergence = np.allclose(A_ee, A_zoh) and np.allclose(B_ee, B_zoh) and np.allclose(c_ee, c_zoh)
                if not convergence:
                    h /= 10.
            self.assertTrue(convergence)

if __name__ == '__main__':
    unittest.main()