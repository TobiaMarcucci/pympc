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
        A = np.array([[1., 0.], [0., 1.], [1., 1.]])
        b = np.ones((3, 1))
        self._test_matrix_unordered_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            )
        c = np.ones(2)
        self.assertRaises(ValueError, p.add_inequality, A, c)

    	# add equalities
        p.add_equality(A, b)
        self._test_matrix_unordered_rows(
            np.hstack((A, b)),
            np.hstack((p.C, p.d))
            )
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
        self._test_matrix_unordered_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            )

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
        self._test_matrix_unordered_rows(
            np.hstack((A_lb, b_lb)),
            np.hstack((p.A, p.b))
            )

        # from upper bound
        p = Polyhedron.from_upper_bound(x)
        A_ub = np.array([[1., 0.], [0., 1.]])
        b_ub = np.array([[1.], [1.]])
        self._test_matrix_unordered_rows(
            np.hstack((A_ub, b_ub)),
            np.hstack((p.A, p.b))
            )

        # from upper and lower bounds
        p = Polyhedron.from_bounds(x, x)
        A = np.vstack((A_lb, A_ub))
        b = np.vstack((b_lb, b_ub))
        self._test_matrix_unordered_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            )

        # different size lower and upper bound
        y = np.ones((3,1))
        self.assertRaises(ValueError, Polyhedron.from_bounds, x, y)

        # from lower bound of not all the variables
        indices = [1, 2]
        n = 3
        p = Polyhedron.from_lower_bound(x, indices, n)
        A_lb = np.array([[0., -1., 0.], [0., 0., -1.]])
        b_lb = np.array([[-1.], [-1.]])
        self._test_matrix_unordered_rows(
            np.hstack((A_lb, b_lb)),
            np.hstack((p.A, p.b))
            )

        # from lower bound of not all the variables
        indices = [0, 2]
        n = 3
        p = Polyhedron.from_upper_bound(x, indices, n)
        A_ub = np.array([[1., 0., 0.], [0., 0., 1.]])
        b_ub = np.array([[1.], [1.]])
        self._test_matrix_unordered_rows(
            np.hstack((A_ub, b_ub)),
            np.hstack((p.A, p.b))
            )

        # from upper and lower bounds of not all the variables
        indices = [0, 1]
        n = 3
        p = Polyhedron.from_bounds(x, x, indices, n)
        A = np.array([[-1., 0., 0.], [0., -1., 0.], [1., 0., 0.], [0., 1., 0.]])
        b = np.array([[-1.], [-1.], [1.], [1.]])
        self._test_matrix_unordered_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            )

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
        self._test_matrix_unordered_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b)),
            normalize=False
            )
        self._test_matrix_unordered_rows(
            np.hstack((C, d)),
            np.hstack((p.C, p.d)),
            normalize=False
            )
        
    def test_remove_equalities(self):

        # contruct polyhedron
        x_min = - np.ones((2, 1))
        x_max = - x_min
        p = Polyhedron.from_bounds(x_min, x_max)
        C = np.ones((1, 2))
        d = np.ones((1, 1))
        p.add_equality(C, d)
        E, f, N, R = p._remove_equalities()

        # check result
        self.assertAlmostEqual(N[0,0]/N[1,0], -1)
        self.assertAlmostEqual(R[0,0]/R[1,0], 1)
        intersections = [
            np.sqrt(2.)/2,
            3.*np.sqrt(2.)/2,
            - np.sqrt(2.)/2,
            - 3.*np.sqrt(2.)/2
            ]
        for n in intersections:
            residual = E.dot(np.array([[n]])) - f
            self.assertAlmostEqual(np.min(np.abs(residual)), 0.)

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
        self._test_matrix_unordered_rows(
            np.hstack((A_min, b_min)),
            np.hstack((p.A, p.b))
            )

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
        self._test_matrix_unordered_rows(
            np.hstack((A_min, b_min)),
            np.hstack((p.A, p.b))
            )

        # add (redundant) inequality coincident with the equality
        p.add_inequality(C, d)
        p.remove_redundant_inequalities()
        self._test_matrix_unordered_rows(
            np.hstack((A_min, b_min)),
            np.hstack((p.A, p.b))
            )

        # add (redundant) inequality
        p.add_inequality(
            np.array([[-1., 1.]]),
            np.array([[1.1]])
            )
        p.remove_redundant_inequalities()
        self._test_matrix_unordered_rows(
            np.hstack((A_min, b_min)),
            np.hstack((p.A, p.b))
            )

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

        # with equalities, 3d case
        x_min = np.array([[-1.],[-2.]])
        x_max = - x_min
        p = Polyhedron.from_bounds(x_min, x_max, [0,1], 3)
        C = np.array([[1., 0., -1.]])
        d = np.zeros((1,1))
        p.add_equality(C, d)
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

    def test_chebyshev(self):

        # simple 2d problem
        x_min = np.zeros((2,1))
        x_max = 2. * np.ones((2,1))
        p = Polyhedron.from_bounds(x_min, x_max)
        r, c = p.chebyshev()
        self.assertAlmostEqual(r, 1.)
        np.testing.assert_array_almost_equal(c, np.ones((2, 1)))

        # add nasty inequality
        A = np.zeros((1,2))
        b = np.ones((1,1))
        p.add_inequality(A,b)
        r, c = p.chebyshev()
        self.assertAlmostEqual(r, 1.)
        np.testing.assert_array_almost_equal(c, np.ones((2, 1)))

        # add equality
        C = np.ones((1,2))
        d = np.array([[3.]])
        p.add_equality(C,d)
        r, c = p.chebyshev()
        self.assertAlmostEqual(r, np.sqrt(2.)/2.)
        np.testing.assert_array_almost_equal(c, 1.5*np.ones((2, 1)))

        # negative radius
        x_min = np.ones((2,1))
        x_max = - np.ones((2,1))
        p = Polyhedron.from_bounds(x_min, x_max)
        r, c = p.chebyshev()
        self.assertAlmostEqual(r, -1.)
        np.testing.assert_array_almost_equal(c, np.zeros((2, 1)))

        # unbounded
        p = Polyhedron.from_lower_bound(x_min)
        r, c = p.chebyshev()
        self.assertTrue(r is None)
        self.assertTrue(c is None)

        # bounded very difficult
        x0_max = np.array([[2.]])
        p.add_upper_bound(x0_max, [1])
        r, c = p.chebyshev()
        self.assertAlmostEqual(r, .5)
        self.assertAlmostEqual(c[0,0], 1.5)

        # 3d case
        x_min = np.array([[-1.],[-2.]])
        x_max = - x_min
        p = Polyhedron.from_bounds(x_min, x_max, [0,1], 3)
        C = np.array([[1., 0., -1.]])
        d = np.zeros((1,1))
        p.add_equality(C, d)
        r, c = p.chebyshev()
        self.assertAlmostEqual(r, np.sqrt(2.))
        self.assertAlmostEqual(c[0,0], 0.)
        self.assertAlmostEqual(c[2,0], 0.)
        self.assertTrue(np.abs(c[1,0]) < 2. - r + 1.e-6)

    def test_from_convex_hull(self):

        # simple 2d
        points = [
            np.array([[0.],[0.]]),
            np.array([[1.],[0.]]),
            np.array([[0.],[1.]])
        ]
        p = Polyhedron.from_convex_hull(points)
        A = np.array([
            [-1., 0.],
            [0., -1.],
            [1., 1.],
            ])
        b = np.array([[0.],[0.],[1.]])
        self._test_matrix_unordered_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            )

        # simple 3d
        points = [
            np.array([[0.],[0.],[0.]]),
            np.array([[1.],[0.],[0.]]),
            np.array([[0.],[1.],[0.]]),
            np.array([[0.],[0.],[1.]])
        ]
        p = Polyhedron.from_convex_hull(points)
        A = np.array([
            [-1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.],
            [1., 1., 1.],
            ])
        b = np.array([[0.],[0.],[0.],[1.]])
        self._test_matrix_unordered_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            )

        # another 2d with internal point
        points = [
            np.array([[0.],[0.]]),
            np.array([[1.],[0.]]),
            np.array([[0.],[1.]]),
            np.array([[1.],[1.]]),
            np.array([[.5],[.5]]),
        ]
        p = Polyhedron.from_convex_hull(points)
        A = np.array([
            [-1., 0.],
            [0., -1.],
            [1., 0.],
            [0., 1.],
            ])
        b = np.array([[0.],[0.],[1.],[1.]])
        self._test_matrix_unordered_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            )

    def test_get_vertices(self):

        # basic eample
        A = np.array([[-1, 0.],[0., -1.],[2., 1.],[-0.5, 1.]])
        b = np.array([[0.],[0.],[4.],[2.]])
        p = Polyhedron(A,b)
        u_list = [
            np.array([[0.],[0.]]),
            np.array([[2.],[0.]]),
            np.array([[0.],[2.]]),
            np.array([[.8],[2.4]])
        ]
        v_list = p.get_vertices()
        self._test_list_of_arrays(v_list, u_list)

        # 1d example
        A = np.array([[-1],[1.]])
        b = np.array([[1.],[1.]])
        p = Polyhedron(A,b)
        u_list = [
            np.array([[-1.]]),
            np.array([[1.]])
        ]
        v_list = p.get_vertices()
        self._test_list_of_arrays(v_list, u_list)

        # unbounded
        x_min = np.zeros((2,1))
        p = Polyhedron.from_lower_bound(x_min)
        v_list = p.get_vertices()
        self.assertTrue(v_list is None)

        # lower dimensional (because of the inequalities)
        x_max = np.array([[1.],[0.]])
        p.add_upper_bound(x_max)
        v_list = p.get_vertices()
        self.assertTrue(v_list is None)

        # empty
        x_max = - np.ones((2,1))
        p.add_upper_bound(x_max)
        v_list = p.get_vertices()
        self.assertTrue(v_list is None)

        # 3d case
        x_min = - np.ones((3,1))
        x_max = - x_min
        p = Polyhedron.from_bounds(x_min, x_max)
        u_list = [
            np.array([[1.],[1.],[1.]]),
            np.array([[1.],[1.],[-1.]]),
            np.array([[1.],[-1.],[1.]]),
            np.array([[-1.],[1.],[1.]]),
            np.array([[1.],[-1.],[-1.]]),
            np.array([[-1.],[-1.],[1.]]),
            np.array([[-1.],[1.],[-1.]]),
            np.array([[-1.],[-1.],[-1.]]),
        ]
        v_list = p.get_vertices()
        self._test_list_of_arrays(v_list, u_list)

        # 3d case with equalities
        x_min = np.array([[-1.],[-2.]])
        x_max = - x_min
        p = Polyhedron.from_bounds(x_min, x_max, [0,1], 3)
        C = np.array([[1., 0., -1.]])
        d = np.zeros((1,1))
        p.add_equality(C, d)
        u_list = [
            np.array([[1.],[2.],[1.]]),
            np.array([[-1.],[2.],[-1.]]),
            np.array([[1.],[-2.],[1.]]),
            np.array([[-1.],[-2.],[-1.]])
        ]
        v_list = p.get_vertices()
        self._test_list_of_arrays(v_list, u_list)

    def _test_matrix_unordered_rows(self, A, B, normalize=True):
        """
        Tests that two matrices contain the same rows.
        The order of the rows can be different.
        The option normalize, normalizes the rows of A and B. 
        """
        self.assertTrue(A.shape[0] == B.shape[0])
        if normalize:
            for i in range(A.shape[0]):
                A[i,:] = A[i,:]/np.linalg.norm(A[i,:])
                B[i,:] = B[i,:]/np.linalg.norm(B[i,:])
        for a in A:
            i = np.where([np.allclose(a, b) for b in B])[0]
            self.assertTrue(len(i) == 1)
            B = np.delete(B, i, 0)

    def _test_list_of_arrays(self, v_list, u_list):
        """
        Tests that two lists of array contain the same elements.
        The order of the elements in the lists can be different.
        """
        self.assertTrue(len(u_list) == len(v_list))
        for v in v_list:
            i = np.where([np.allclose(v, u) for u in u_list])[0]
            self.assertTrue(len(i) == 1)
            del u_list[i[0]]

if __name__ == '__main__':
    unittest.main()