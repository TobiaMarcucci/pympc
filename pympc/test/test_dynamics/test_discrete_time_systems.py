# external imports
import unittest
import numpy as np

# internal inputs
from pympc.dynamics.discrete_time_systems import LinearSystem, AffineSystem,PieceWiseAffineSystem
from pympc.geometry.polyhedron import Polyhedron

class TestLinearSystem(unittest.TestCase):

    def test_intialization(self):
        np.random.seed(1)

        # wrong initializations
        A = np.ones((3,3))
        B = np.ones((4,1))
        self.assertRaises(ValueError, LinearSystem, A, B)
        A = np.ones((4,5))
        self.assertRaises(ValueError, LinearSystem, A, B)

    def test_condense_and_simulate(self):

        # random systems
        for i in range(10):
            n = np.random.randint(1, 10)
            m = np.random.randint(1, 10)
            N = np.random.randint(10, 50)
            x0 = np.random.rand(n, 1)
            u = [np.random.rand(m, 1)/10. for j in range(N)]
            A = np.random.rand(n,n)/10.
            B = np.random.rand(n,m)/10.
            S = LinearSystem(A, B)

            # simulate vs condense
            x = S.simulate(x0, u)
            A_bar, B_bar = S.condense(N)
            np.testing.assert_array_almost_equal(
                np.vstack(x),
                A_bar.dot(x0) + B_bar.dot(np.vstack(u))
                )

    def test_from_continuous(self):

        # test from continuous
        for i in range(10):
            n = np.random.randint(1, 10)
            m = np.random.randint(1, 10)
            A = np.random.rand(n,n)
            B = np.random.rand(n,m)

            # reduce discretization step until the two method are almost equivalent
            h = .01
            convergence = False
            while not convergence:
                S_ee = LinearSystem.from_continuous(A, B, h, 'explicit_euler')
                S_zoh = LinearSystem.from_continuous(A, B, h, 'zero_order_hold')
                convergence = np.allclose(S_ee.A, S_zoh.A) and np.allclose(S_ee.B, S_zoh.B)
                if not convergence:
                    h /= 10.
            self.assertTrue(convergence)
        self.assertRaises(ValueError, LinearSystem.from_continuous, A, B, h, 'gatto')

class TestAffineSystem(unittest.TestCase):

    def test_intialization(self):
        np.random.seed(1)

        # wrong initializations
        A = np.ones((3,3))
        B = np.ones((4,1))
        c = np.ones((4,1))
        self.assertRaises(ValueError, AffineSystem, A, B, c)
        A = np.ones((4,5))
        self.assertRaises(ValueError, AffineSystem, A, B, c)
        A = np.ones((4,4))
        c = np.ones((5,1))
        self.assertRaises(ValueError, AffineSystem, A, B, c)


    def test_condense_and_simulate(self):

        # random systems
        for i in range(10):
            n = np.random.randint(1, 10)
            m = np.random.randint(1, 10)
            N = np.random.randint(10, 50)
            x0 = np.random.rand(n, 1)
            u = [np.random.rand(m, 1)/10. for j in range(N)]
            A = np.random.rand(n,n)/10.
            B = np.random.rand(n,m)/10.
            c = np.random.rand(n,1)/10.
            S = AffineSystem(A, B, c)

            # simulate vs condense
            x = S.simulate(x0, u)
            A_bar, B_bar, c_bar = S.condense(N)
            np.testing.assert_array_almost_equal(
                np.vstack(x),
                A_bar.dot(x0) + B_bar.dot(np.vstack(u)) + c_bar
                )

    def test_from_continuous(self):

        # test from continuous
        for i in range(10):
            n = np.random.randint(1, 10)
            m = np.random.randint(1, 10)
            A = np.random.rand(n,n)
            B = np.random.rand(n,m)
            c = np.random.rand(n,1)

            # reduce discretization step until the two method are almost equivalent
            h = .01
            convergence = False
            while not convergence:
                S_ee = AffineSystem.from_continuous(A, B, c, h, 'explicit_euler')
                S_zoh = AffineSystem.from_continuous(A, B, c, h, 'zero_order_hold')
                convergence = np.allclose(S_ee.A, S_zoh.A) and np.allclose(S_ee.B, S_zoh.B) and np.allclose(S_ee.c, S_zoh.c)
                if not convergence:
                    h /= 10.
            self.assertTrue(convergence)
        self.assertRaises(ValueError, AffineSystem.from_continuous, A, B, c, h, 'gatto')

class TestPieceWiseAffineSystem(unittest.TestCase):

    def test_intialization(self):
        np.random.seed(1)

        # different number of systems and domains
        A = np.ones((3,3))
        B = np.ones((3,2))
        c = np.ones((3,1))
        S = AffineSystem(A, B, c)
        affine_systems = [S]*5
        F = np.ones((9,5))
        g = np.ones((9,1))
        D = Polyhedron(F, g)
        domains = [D]*4
        self.assertRaises(ValueError, PieceWiseAffineSystem, affine_systems, domains)

        # incopatible number of states in affine systems
        domains += [D, D]
        A = np.ones((2,2))
        B = np.ones((2,2))
        c = np.ones((2,1))
        affine_systems.append(AffineSystem(A, B, c))
        self.assertRaises(ValueError, PieceWiseAffineSystem, affine_systems, domains)

        # incopatible number of inputs in affine systems
        del affine_systems[-1]
        A = np.ones((3,3))
        B = np.ones((3,1))
        c = np.ones((3,1))
        affine_systems.append(AffineSystem(A, B, c))
        self.assertRaises(ValueError, PieceWiseAffineSystem, affine_systems, domains)

        # different dimensinality of the domains and the systems
        del affine_systems[-1]
        affine_systems += [S, S]
        F = np.ones((9,4))
        g = np.ones((9,1))
        domains.append(Polyhedron(F, g))
        self.assertRaises(ValueError, PieceWiseAffineSystem, affine_systems, domains)

        # different dimensinality of the domains and the systems
        F = np.ones((9,4))
        g = np.ones((9,1))
        D = Polyhedron(F, g)
        domains = [D]*7
        self.assertRaises(ValueError, PieceWiseAffineSystem, affine_systems, domains)

    def test_condense_and_simulate_and_get_mode(self):
        np.random.seed(1)

        # test with random systems
        for i in range(10):

            # test dimensions
            n = np.random.randint(1, 10)
            m = np.random.randint(1, 10)
            N = np.random.randint(10, 50)

            # initial state
            x0 = np.random.rand(n, 1)

            # initialize loop variables
            x_t = x0
            x = x0
            u = np.zeros((0,1))
            u_list = []
            affine_systems = []
            domains = []

            # simulate for N steps
            for t in range(N):

                # generate random dynamics
                A_t = np.random.rand(n,n)/10.
                B_t = np.random.rand(n,m)/10.
                c_t = np.random.rand(n,1)/10.

                # simulate with random input
                u_t = np.random.rand(m,1)/10.
                u = np.vstack((u, u_t))
                u_list.append(u_t)
                x_tp1 = A_t.dot(x_t) + B_t.dot(u_t) + c_t
                x = np.vstack((x, x_tp1))
                
                # create a domain that contains x and u (it has to be super-tight so that the pwa system is well posed!)
                D = Polyhedron.from_bounds(
                    np.vstack((x_t/1.01, u_t/1.01)),
                    np.vstack((x_t*1.01, u_t*1.01))
                    )
                domains.append(D)
                affine_systems.append(AffineSystem(A_t, B_t, c_t))

                # reset state
                x_t = x_tp1

            # construct pwa system
            S = PieceWiseAffineSystem(affine_systems, domains)

            # test condense
            mode_sequence = range(N)
            A_bar, B_bar, c_bar = S.condense(mode_sequence)
            np.testing.assert_array_almost_equal(
                x,
                A_bar.dot(x0) + B_bar.dot(u) + c_bar
                )

            # test simulate
            x_list, mode_sequence = S.simulate(x0, u_list)
            np.testing.assert_array_almost_equal(x, np.vstack(x_list))

            # test get mode
            for t in range(N):
                self.assertTrue(S.get_mode(x_list[t], u_list[t]) == t)

    def test_is_well_posed(self):

        # domains
        D0 = Polyhedron.from_bounds(-np.ones((3,1)), np.ones((3,1)))
        D1 = Polyhedron.from_bounds(np.zeros((3,1)), 2.*np.ones((3,1)))
        domains = [D0, D1]

        # pwa system
        A = np.ones((2,2))
        B = np.ones((2,1))
        c = np.ones((2,1))
        S = AffineSystem(A, B, c)
        affine_systems = [S]*2
        S_pwa = PieceWiseAffineSystem(affine_systems, domains)

        # check if well posed
        self.assertFalse(S_pwa.is_well_posed())

        # make it well posed
        D1.add_lower_bound(np.ones((3,1)))
        D2 = Polyhedron.from_bounds(2.*np.ones((3,1)), 3.*np.ones((3,1)))
        domains = [D0, D1, D2]
        affine_systems = [S]*3
        S_pwa = PieceWiseAffineSystem(affine_systems, domains)
        self.assertTrue(S_pwa.is_well_posed())

    def test_from_orthogonal_domains(self):
        np.random.seed(1)

        # different number of state and input domains
        A = np.ones((3,3))
        B = np.ones((3,2))
        c = np.ones((3,1))
        S = AffineSystem(A, B, c)
        affine_systems = [S]*5
        Fx = np.ones((6,3))
        gx = np.ones((6,1))
        X = Polyhedron(Fx, gx)
        state_domains = [X]*5
        Fu = np.ones((9,2))
        gu = np.ones((9,1))
        U = Polyhedron(Fu, gu)
        input_domains = [U]*4
        self.assertRaises(
            ValueError,
            PieceWiseAffineSystem.from_orthogonal_domains,
            affine_systems,
            state_domains,
            input_domains
            )

        # different number of states
        Fx = np.ones((6,4))
        gx = np.ones((6,1))
        del state_domains[-1]
        state_domains.append(Polyhedron(Fx, gx))
        input_domains.append(U)
        self.assertRaises(
            ValueError,
            PieceWiseAffineSystem.from_orthogonal_domains,
            affine_systems,
            state_domains,
            input_domains
            )

        # different number of states
        Fu = np.ones((9,3))
        gu = np.ones((9,1))
        del state_domains[-1]
        del input_domains[-1]
        state_domains.append(X)
        input_domains.append(Polyhedron(Fu, gu))
        self.assertRaises(
            ValueError,
            PieceWiseAffineSystem.from_orthogonal_domains,
            affine_systems,
            state_domains,
            input_domains
            )

if __name__ == '__main__':
    unittest.main()