# external imports
import unittest
import numpy as np
from itertools import product
from scipy.linalg import block_diag
from copy import copy

# internal inputs
from pympc.geometry.polyhedron import Polyhedron, convex_hull_method
from pympc.geometry.utils import same_rows, same_vectors

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
        self.assertTrue(same_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            ))
        c = np.ones(2)
        self.assertRaises(ValueError, p.add_inequality, A, c)

        # add equalities
        p.add_equality(A, b)
        self.assertTrue(same_rows(
            np.hstack((A, b)),
            np.hstack((p.C, p.d))
            ))
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
        self.assertTrue(same_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            ))

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
        self.assertTrue(same_rows(
            np.hstack((A_lb, b_lb)),
            np.hstack((p.A, p.b))
            ))

        # from upper bound
        p = Polyhedron.from_upper_bound(x)
        A_ub = np.array([[1., 0.], [0., 1.]])
        b_ub = np.array([[1.], [1.]])
        self.assertTrue(same_rows(
            np.hstack((A_ub, b_ub)),
            np.hstack((p.A, p.b))
            ))

        # from upper and lower bounds
        p = Polyhedron.from_bounds(x, x)
        A = np.vstack((A_lb, A_ub))
        b = np.vstack((b_lb, b_ub))
        self.assertTrue(same_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            ))

        # different size lower and upper bound
        y = np.ones((3,1))
        self.assertRaises(ValueError, Polyhedron.from_bounds, x, y)

        # from lower bound of not all the variables
        indices = [1, 2]
        n = 3
        p = Polyhedron.from_lower_bound(x, indices, n)
        A_lb = np.array([[0., -1., 0.], [0., 0., -1.]])
        b_lb = np.array([[-1.], [-1.]])
        self.assertTrue(same_rows(
            np.hstack((A_lb, b_lb)),
            np.hstack((p.A, p.b))
            ))

        # from lower bound of not all the variables
        indices = [0, 2]
        n = 3
        p = Polyhedron.from_upper_bound(x, indices, n)
        A_ub = np.array([[1., 0., 0.], [0., 0., 1.]])
        b_ub = np.array([[1.], [1.]])
        self.assertTrue(same_rows(
            np.hstack((A_ub, b_ub)),
            np.hstack((p.A, p.b))
            ))

        # from upper and lower bounds of not all the variables
        indices = [0, 1]
        n = 3
        p = Polyhedron.from_bounds(x, x, indices, n)
        A = np.array([[-1., 0., 0.], [0., -1., 0.], [1., 0., 0.], [0., 1., 0.]])
        b = np.array([[-1.], [-1.], [1.], [1.]])
        self.assertTrue(same_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            ))

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
        self.assertTrue(same_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b)),
            normalize=False
            ))
        self.assertTrue(same_rows(
            np.hstack((C, d)),
            np.hstack((p.C, p.d)),
            normalize=False
            ))

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

        # add another feasible equality (raises 'equality constraints C x = d do not have a nullspace.')
        C = np.array([[1., -1.]])
        d = np.zeros((1, 1))
        p1 = copy(p)
        p1.add_equality(C, d)
        self.assertRaises(ValueError, p1._remove_equalities)

        # add another unfeasible equality (raises 'equality constraints C x = d do not have a nullspace.')
        C = np.array([[0., 1.]])
        d = np.array([[5.]])
        p1.add_equality(C, d)
        self.assertRaises(ValueError, p1._remove_equalities)

        # add a linearly dependent equality (raises 'equality constraints C x = d are linearly dependent.')
        C = 2*np.ones((1, 2))
        d = np.ones((1, 1))
        p2 = copy(p)
        p2.add_equality(C, d)
        self.assertRaises(ValueError, p2._remove_equalities)

    def test_remove_redundant_inequalities(self):

        # minimal facets only inequalities
        A = np.array([[1., 1.], [-1., 1.], [0., -1.], [0., 1.], [0., 1.], [2., 2.]])
        b = np.array([[1.],          [1.],      [1.],     [1.],     [2.],     [2.]])
        p = Polyhedron(A,b)
        mf = set(p.minimal_facets())
        self.assertTrue(mf == set([1,2,0]) or mf == set([1,2,5]))

        # add nasty redundant inequality
        A = np.zeros((1,2))
        b = np.ones((1,1))
        p.add_inequality(A, b)
        mf = set(p.minimal_facets())
        self.assertTrue(mf == set([1,2,0]) or mf == set([1,2,5]))

        # remove redundant facets
        p.remove_redundant_inequalities()
        A_min = np.array([[-1., 1.], [0., -1.], [1., 1.]])
        b_min = np.array([[1.], [1.], [1.]])
        self.assertTrue(same_rows(
            np.hstack((A_min, b_min)),
            np.hstack((p.A, p.b))
            ))

        # both inequalities and equalities
        x_min = - np.ones((2, 1))
        x_max = - x_min
        p = Polyhedron.from_bounds(x_min, x_max)
        C = np.ones((1, 2))
        d = np.ones((1, 1))
        p.add_equality(C, d)
        self.assertEqual(
            [2, 3],
            sorted(p.minimal_facets())
            )
        p.remove_redundant_inequalities()
        A_min = np.array([[1., 0.], [0., 1.]])
        b_min = np.array([[1.], [1.]])
        self.assertTrue(same_rows(
            np.hstack((A_min, b_min)),
            np.hstack((p.A, p.b))
            ))

        # add (redundant) inequality coincident with the equality
        p.add_inequality(C, d)
        p.remove_redundant_inequalities()
        self.assertTrue(same_rows(
            np.hstack((A_min, b_min)),
            np.hstack((p.A, p.b))
            ))

        # add (redundant) inequality
        p.add_inequality(
            np.array([[-1., 1.]]),
            np.array([[1.1]])
            )
        p.remove_redundant_inequalities()
        self.assertTrue(same_rows(
            np.hstack((A_min, b_min)),
            np.hstack((p.A, p.b))
            ))

        # add unfeasible equality (raises: 'empty polyhedron, cannot remove redundant inequalities.')
        C = np.ones((1, 2))
        d = -np.ones((1, 1))
        p.add_equality(C, d)
        self.assertRaises(ValueError, p.remove_redundant_inequalities)

        # empty polyhderon (raises: 'empty polyhedron, cannot remove redundant inequalities.')
        x_min = np.ones((2,1))
        x_max = np.zeros((2,1))
        p = Polyhedron.from_bounds(x_min, x_max)
        self.assertRaises(ValueError, p.remove_redundant_inequalities)

    def test_empty(self):

        # full dimensional
        x_min = 1.*np.ones((2,1))
        x_max = 2.*np.ones((2,1))
        p = Polyhedron.from_bounds(x_min, x_max)
        self.assertFalse(p.empty)

        # lower dimensional, but not empy
        C = np.ones((1, 2))
        d = np.array([[3.]])
        p.add_equality(C, d)
        self.assertFalse(p.empty)

        # lower dimensional and empty
        x_0_max = np.array([[.5]])
        p.add_upper_bound(x_0_max, [0])
        self.assertTrue(p.empty)

    def test_bounded(self):

        # bounded (empty), easy (ker(A) empty)
        x_min = np.ones((2, 1))
        x_max = - np.ones((2, 1))
        p = Polyhedron.from_bounds(x_min, x_max)
        self.assertTrue(p.bounded)

        # bounded (empty), tricky (ker(A) not empty)
        x0_min = np.array([[1.]])
        x0_max = np.array([[-1.]])
        p = Polyhedron.from_bounds(x0_min, x0_max, [0], 2)
        self.assertTrue(p.bounded)

        # bounded easy
        x_min = 1.*np.ones((2,1))
        x_max = 2.*np.ones((2,1))
        p = Polyhedron.from_bounds(x_min, x_max)
        self.assertTrue(p.bounded)

        # unbounded, halfspace
        x0_min = np.array([[0.]])
        p = Polyhedron.from_lower_bound(x0_min, [0], 2)
        self.assertFalse(p.bounded)

        # unbounded, positive orthant
        x1_min = x0_min
        p.add_lower_bound(x1_min, [1])
        self.assertFalse(p.bounded)

        # unbounded: parallel inequalities, slice of positive orthant
        x0_max = np.array([[1.]])
        p.add_upper_bound(x0_max, [0])
        self.assertFalse(p.bounded)

        # unbounded lower dimensional, line in the positive orthant
        x0_min = x0_max
        p.add_lower_bound(x0_min, [0])
        self.assertFalse(p.bounded)

        # bounded lower dimensional, segment in the positive orthant
        x1_max = np.array([[1000.]])
        p.add_upper_bound(x1_max, [1])
        self.assertTrue(p.bounded)

        # with equalities, 3d case
        x_min = np.array([[-1.],[-2.]])
        x_max = - x_min
        p = Polyhedron.from_bounds(x_min, x_max, [0,1], 3)
        C = np.array([[1., 0., -1.]])
        d = np.zeros((1,1))
        p.add_equality(C, d)
        self.assertTrue(p.bounded)

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

    def test_intersection(self):

        # first polyhedron
        x1_min = - np.ones((2,1))
        x1_max = - x1_min
        p1 = Polyhedron.from_bounds(x1_min, x1_max)

        # second polyhedron
        x2_min = np.zeros((2,1))
        x2_max = 2. * np.ones((2,1))
        p2 = Polyhedron.from_bounds(x2_min, x2_max)

        # intersection
        p3 = p1.intersection(p2)
        p3.remove_redundant_inequalities()
        p4 = Polyhedron.from_bounds(x2_min, x1_max)
        self.assertTrue(same_rows(
            np.hstack((p3.A, p3.b)),
            np.hstack((p4.A, p4.b))
            ))

        # add equalities
        C1 = np.array([[1., 0.]])
        d1 = np.array([-.5])
        p1.add_equality(C1, d1)
        p3 = p1.intersection(p2)
        self.assertTrue(p3.empty)

    def test_cartesian_product(self):

        # simple case
        n = 2
        x_min = -np.ones((n,1))
        x_max = -x_min
        p1 = Polyhedron.from_bounds(x_min, x_max)
        C = np.ones((1,n))
        d = np.zeros((1,1))
        p1.add_equality(C, d)
        p2 = p1.cartesian_product(p1)

        # derive the cartesian product
        x_min = -np.ones((2*n,1))
        x_max = -x_min
        p3 = Polyhedron.from_bounds(x_min, x_max)
        C = block_diag(*[np.ones((1,n))]*2)
        d = np.zeros((2,1))
        p3.add_equality(C, d)

        # compare results
        self.assertTrue(same_rows(
            np.hstack((p2.A, p2.b)),
            np.hstack((p3.A, p3.b))
            ))
        self.assertTrue(same_rows(
            np.hstack((p2.C, p2.d)),
            np.hstack((p3.C, p3.d))
            ))

    def test_chebyshev(self):

        # simple 2d problem
        x_min = np.zeros((2,1))
        x_max = 2. * np.ones((2,1))
        p = Polyhedron.from_bounds(x_min, x_max)
        self.assertAlmostEqual(p.radius, 1.)
        np.testing.assert_array_almost_equal(p.center, np.ones((2, 1)))

        # add nasty inequality
        A = np.zeros((1,2))
        b = np.ones((1,1))
        p.add_inequality(A,b)
        self.assertAlmostEqual(p.radius, 1.)
        np.testing.assert_array_almost_equal(p.center, np.ones((2, 1)))

        # add equality
        C = np.ones((1,2))
        d = np.array([[3.]])
        p.add_equality(C,d)
        self.assertAlmostEqual(p.radius, np.sqrt(2.)/2.)
        np.testing.assert_array_almost_equal(p.center, 1.5*np.ones((2, 1)))

        # negative radius
        x_min = np.ones((2,1))
        x_max = - np.ones((2,1))
        p = Polyhedron.from_bounds(x_min, x_max)
        self.assertAlmostEqual(p.radius, -1.)
        np.testing.assert_array_almost_equal(p.center, np.zeros((2, 1)))

        # unbounded
        p = Polyhedron.from_lower_bound(x_min)
        self.assertTrue(p.radius is None)
        self.assertTrue(p.center is None)

        # bounded very difficult
        x0_max = np.array([[2.]])
        p.add_upper_bound(x0_max, [1])
        self.assertAlmostEqual(p.radius, .5)
        self.assertAlmostEqual(p.center[0,0], 1.5)

        # 3d case
        x_min = np.array([[-1.],[-2.]])
        x_max = - x_min
        p = Polyhedron.from_bounds(x_min, x_max, [0,1], 3)
        C = np.array([[1., 0., -1.]])
        d = np.zeros((1,1))
        p.add_equality(C, d)
        self.assertAlmostEqual(p.radius, np.sqrt(2.))
        self.assertAlmostEqual(p.center[0,0], 0.)
        self.assertAlmostEqual(p.center[2,0], 0.)
        self.assertTrue(np.abs(p.center[1,0]) < 2. - p.radius + 1.e-6)

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
        self.assertTrue(same_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            ))

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
        self.assertTrue(same_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            ))

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
        self.assertTrue(same_rows(
            np.hstack((A, b)),
            np.hstack((p.A, p.b))
            ))

    def test_vertices(self):

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
        self.assertTrue(same_vectors(p.vertices, u_list))

        # 1d example
        A = np.array([[-1],[1.]])
        b = np.array([[1.],[1.]])
        p = Polyhedron(A,b)
        u_list = [
            np.array([[-1.]]),
            np.array([[1.]])
        ]
        self.assertTrue(same_vectors(p.vertices, u_list))

        # unbounded
        x_min = np.zeros((2,1))
        p = Polyhedron.from_lower_bound(x_min)
        self.assertTrue(p.vertices is None)

        # lower dimensional (because of the inequalities)
        x_max = np.array([[1.],[0.]])
        p.add_upper_bound(x_max)
        self.assertTrue(p.vertices is None)

        # empty
        x_max = - np.ones((2,1))
        p.add_upper_bound(x_max)
        self.assertTrue(p.vertices is None)

        # 3d case
        x_min = - np.ones((3,1))
        x_max = - x_min
        p = Polyhedron.from_bounds(x_min, x_max)
        u_list = [np.array(v).reshape(3,1) for v in product([1., -1.], repeat=3)]
        self.assertTrue(same_vectors(p.vertices, u_list))

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
        self.assertTrue(same_vectors(p.vertices, u_list))

    def test_delete_attributes(self):

        # initialize polyhedron
        x_min = - np.ones((2,1))
        x_max = - x_min
        p = Polyhedron.from_bounds(x_min, x_max)

        # compute alle the property attributes
        self.assertFalse(p.empty)
        self.assertTrue(p.bounded)
        self.assertAlmostEqual(p.radius, 1.)
        np.testing.assert_array_almost_equal(
            p.center,
            np.zeros((2,1))
            )
        self.assertTrue(same_vectors(
            p.vertices,
            [np.array(v).reshape(2,1) for v in product([1., -1.], repeat=2)]
            ))

        # add inequality
        A = np.eye(2)
        b = .8*np.ones((2,1))
        p.add_inequality(A, b)
        self.assertFalse(p.empty)
        self.assertTrue(p.bounded)
        self.assertAlmostEqual(p.radius, .9)
        np.testing.assert_array_almost_equal(
            p.center,
            -.1*np.ones((2,1))
            )
        self.assertTrue(same_vectors(
            p.vertices,
            [np.array(v).reshape(2,1) for v in product([.8, -1.], repeat=2)]
            ))

        # add equality
        C = np.array([[1., 1.]])
        d = np.zeros((1, 1))
        p.add_equality(C, d)
        self.assertFalse(p.empty)
        self.assertTrue(p.bounded)
        self.assertAlmostEqual(p.radius, .8*np.sqrt(2.))
        np.testing.assert_array_almost_equal(
            p.center,
            np.zeros((2,1))
            )
        self.assertTrue(same_vectors(
            p.vertices,
            [np.array([[-.8],[.8]]), np.array([[.8],[-.8]])]
            ))

    def test_orthogonal_projection(self):
        # (more tests on projections are in geometry/test_orthogonal_projection.py)

        # unbounded polyhedron
        x_min = - np.ones((3, 1))
        p = Polyhedron.from_lower_bound(x_min)
        self.assertRaises(ValueError, p.project_to, range(2))

        # simple 3d onto 2d
        p.add_upper_bound(- x_min)
        E = np.vstack((
            np.eye(2),
            - np.eye(2)
            ))
        f = np.ones((4, 1))
        vertices = [np.array(v).reshape(2,1) for v in product([1., -1.], repeat=2)]
        proj = p.project_to(range(2))
        proj.remove_redundant_inequalities()
        self.assertTrue(same_rows(
            np.hstack((E, f)),
            np.hstack((proj.A, proj.b))
            ))
        self.assertTrue(same_vectors(vertices, proj.vertices))

        # lower dimensional
        C = np.array([[1., 1., 0.]])
        d = np.zeros((1, 1))
        p.add_equality(C, d)
        self.assertRaises(ValueError, p.project_to, range(2))

        # lower dimensional
        x_min = - np.ones((3, 1))
        x_max = 2. * x_min
        p = Polyhedron.from_bounds(x_min, x_max)
        self.assertRaises(ValueError, p.project_to, range(2))

class TestOrthogonalProjection(unittest.TestCase):

    def test_convex_hull_method(self):
        np.random.seed(3)

        # cube from n-dimensions to n-1-dimensions
        for n in range(2, 6):
            x_min = - np.ones((n, 1))
            x_max = - x_min
            p = Polyhedron.from_bounds(x_min, x_max)
            E = np.vstack((
                np.eye(n-1),
                - np.eye(n-1)
                ))
            f = np.ones((2*(n-1), 1))
            vertices = [np.array(v).reshape(n-1,1) for v in product([1., -1.], repeat=n-1)]
            E_chm, f_chm, vertices_chm = convex_hull_method(p.A, p.b, range(n-1))
            p_chm = Polyhedron(E_chm, f_chm)
            p_chm.remove_redundant_inequalities()
            self.assertTrue(same_rows(
                np.hstack((E, f)),
                np.hstack((p_chm.A, p_chm.b))
                ))
            self.assertTrue(same_vectors(vertices, vertices_chm))

        # test with random polytopes wiith m facets
        m = 10

        # dimension of the higher dimensional polytope
        for n in range(3, 6):

            # dimension of the lower dimensional polytope
            for n_proj in range(2, n):

                # higher dimensional polytope
                A = np.random.randn(m, n)
                b = np.random.rand(m, 1)
                p = Polyhedron(A, b)

                # if not empty or unbounded
                if p.vertices is not None:
                    points = [v[:n_proj, :] for v in p.vertices]
                    p = Polyhedron.from_convex_hull(points)

                    # check half spaces
                    p.remove_redundant_inequalities()
                    E_chm, f_chm, vertices_chm = convex_hull_method(A, b, range(n_proj))
                    p_chm = Polyhedron(E_chm, f_chm)
                    p_chm.remove_redundant_inequalities()
                    self.assertTrue(same_rows(
                        np.hstack((p.A, p.b)),
                        np.hstack((p_chm.A, p_chm.b))
                    ))

                    # check vertices
                    self.assertTrue(same_vectors(p.vertices, vertices_chm))

if __name__ == '__main__':
    unittest.main()