# external imports
import unittest
import numpy as np

# internal inputs
from pympc.dynamics.utils import check_affine_system

class TestUtils(unittest.TestCase):

    def test_check_affine_system(self):

        # non-square A
        A = np.ones((2,1))
        B = np.ones((2,1))
        self.assertRaises(ValueError, check_affine_system, A, B)

        # uncoherent B and A
        A = np.ones((3,3))
        self.assertRaises(ValueError, check_affine_system, A, B)

        # uncoherent c and A
        B = np.ones((3,1))
        c = np.ones((1, 1))
        self.assertRaises(ValueError, check_affine_system, A, B, c)

        # negative h
        c = np.ones((3, 1))
        h = -.1
        self.assertRaises(ValueError, check_affine_system, A, B, c, h)

if __name__ == '__main__':
    unittest.main()