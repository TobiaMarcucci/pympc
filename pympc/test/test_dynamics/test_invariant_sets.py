# external imports
import unittest
import numpy as np

# internal inputs
from pympc.dynamics.invariant_sets import mcais
from pympc.geometry.polyhedron import Polyhedron

class TestMCAIS(unittest.TestCase):

    def test_mcais(self):
        np.random.seed(1)

        # domain
        nx = 2
        x_min = - np.ones((nx, 1))
        x_max = - x_min
        X = Polyhedron.from_bounds(x_min, x_max)

        # stable dynamics
        for i in range(10):
            stable = False
            while not stable:
                A = np.random.rand(nx, nx)
                stable = np.max(np.absolute(np.linalg.eig(A)[0])) < 1.

            # get mcais
            O_inf, _ = mcais(A, X)

            # generate random initial conditions
            for j in range(100):
                x = 3.*np.random.rand(nx, 1) - 1.5

                # if inside stays inside X, if outside sooner or later will leave X
                if O_inf.contains(x):
                    while np.linalg.norm(x) > 0.001:
                        x = A.dot(x)
                        self.assertTrue(X.contains(x))
                else:
                    while X.contains(x):
                        x = A.dot(x)
                        if np.linalg.norm(x) < 0.0001:
                            self.assertTrue(False)
                            pass

if __name__ == '__main__':
    unittest.main()