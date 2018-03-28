# external imports
from six.moves import range  # behaves like xrange on python2, range on python3
import unittest
import numpy as np
from random import shuffle

# internal inputs
from pympc.geometry.utils import nullspace_basis, linearly_independent_rows, plane_through_points, same_rows, same_vectors

class TestUtils(unittest.TestCase):

    def test_nullspace_basis(self):

        # empty nullspace
        A = np.eye(3)
        N = nullspace_basis(A)
        self.assertTrue(N.shape == (3,0))

        # 1d nullspace
        A = np.array([[1.,0.,0.],[0.,1.,0.]])
        N = nullspace_basis(A)
        np.testing.assert_array_almost_equal(
            N,
            np.array([[0.],[0.],[1.]])
        )

        # 1d nullspace
        A = np.array([[1.,0.,0.],[0.,1.,1.]])
        N = nullspace_basis(A)
        self.assertTrue(N.shape == (3,1))
        self.assertAlmostEqual(N[0,0], 0.)
        self.assertAlmostEqual(N[1,0]/N[2,0], -1.)

    def test_linearly_independent_rows(self):

        # one linear dependency
        A = np.array([[1.,0.,0.],[0.,1.,0.],[1.,0.,0.]])
        li_rows = linearly_independent_rows(A)
        self.assertTrue(len(li_rows) == 2)
        self.assertTrue(1 in li_rows)
        li_rows.remove(1)
        self.assertTrue(li_rows[0] == 0 or li_rows[0] == 2)

        # all linearly independent
        A = np.eye(3)
        li_rows = linearly_independent_rows(A)
        self.assertTrue(li_rows == list(range(3)))

    def test_plane_through_points(self):

        # n-dimensional case
        for n in range(2,10):
            points = [p.reshape(n,1) for p in np.eye(n)]
            a, d = plane_through_points(points)
            np.testing.assert_array_almost_equal(
                a,
                np.ones((n,1))/np.sqrt(n)
                )
            self.assertAlmostEqual(d, 1./np.sqrt(n))

        # 2d case through origin
        points = [np.array([[1.],[-1.]]), np.array([[-1.],[1.]])]
        a, d = plane_through_points(points)
        self.assertAlmostEqual(a[0,0]/a[1,0], 1.)
        self.assertAlmostEqual(d, 0.)

    def test_same_rows(self):
        np.random.seed(1)

        # test dimensions
        n = 10
        m = 5
        for i in range(10):

            # check with scaling factors
            A = np.random.rand(n, m)
            scaling = np.random.rand(n)
            B = np.multiply(A.T, scaling).T
            self.assertTrue(same_rows(A, B))

            # check without scaling factors
            B_order = list(range(n))
            shuffle(B_order)
            B = A[B_order, :]
            self.assertTrue(same_rows(A, B, normalize=False))

            # wrong check (same columns)
            scaling = np.random.rand(m)
            B = np.multiply(A, scaling)
            self.assertFalse(same_rows(A, B))


    def test_same_vectors(self):
        np.random.seed(1)

        # test dimensions
        n = 10
        N = 100
        for i in range(10):

            # check equal lists
            v_list = [np.random.rand(n, 1) for j in range(N)]
            u_order = list(range(N))
            shuffle(u_order)
            u_list = [v_list[j] for j in u_order]
            self.assertTrue(same_vectors(v_list, u_list))

        # wrong size (only 2d arrays allowed)
        v_list = [np.random.rand(n) for j in range(N)]
        u_order = list(range(N))
        shuffle(u_order)
        u_list = [v_list[j] for j in u_order]
        self.assertRaises(ValueError, same_vectors, v_list, u_list)

        # wrong size (matrices not allowed)
        v_list = [np.random.rand(n,3) for j in range(N)]
        u_order = list(range(N))
        shuffle(u_order)
        u_list = [v_list[j] for j in u_order]
        self.assertRaises(ValueError, same_vectors, v_list, u_list)

if __name__ == '__main__':
    unittest.main()