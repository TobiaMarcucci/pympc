# external imports
import unittest
import numpy as np
from copy import copy

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.dynamics.discrete_time_systems import LinearSystem, AffineSystem, PieceWiseAffineSystem
from pympc.control.controllers import ModelPredictiveController, HybridModelPredictiveController

class testModelPredictiveController(unittest.TestCase):

    def test_implicit_solution(self):
        np.random.seed(1)

        # random linear system
        for i in range(100):

            # ensure controllability
            n = np.random.randint(2, 5)
            m = np.random.randint(1, n)
            controllable = False
            while not controllable:
                A = np.random.rand(n,n)/10.
                B = np.random.rand(n,m)/10.
                S = LinearSystem(A, B)
                controllable = S.controllable

            # stage constraints
            x_min = -np.random.rand(n,1)
            x_max = np.random.rand(n,1)
            u_min = -np.random.rand(m,1)
            u_max = np.random.rand(m,1)
            X = Polyhedron.from_bounds(x_min, x_max)
            U = Polyhedron.from_bounds(u_min, u_max)
            D = X.cartesian_product(U)

            # assemble controller
            N = np.random.randint(5, 10)
            Q = np.eye(n)
            R = np.eye(m)
            P, K = S.solve_dare(Q, R)
            X_N = S.mcais(K, D)
            controller = ModelPredictiveController(S, N, Q, R, P, D, X_N)

            # simulate
            for j in range(10):

                # intial state
                x = np.multiply(np.random.rand(n,1), x_max - x_min) + x_min

                # mpc
                u_mpc, V_mpc = controller.feedforward(x)

                # lqr
                V_lqr = (.5*x.T.dot(P).dot(x))[0,0]
                u_lqr = []
                x_next = x
                for t in range(N):
                    u_lqr.append(K.dot(x_next))
                    x_next = (A + B.dot(K)).dot(x_next)

                # constraint set (not condensed)
                constraints = copy(D)
                C = np.hstack((np.eye(n), np.zeros((n,m))))
                constraints.add_equality(C, x)
                for t in range(N-1):
                    constraints = constraints.cartesian_product(D)
                    C = np.zeros((n, constraints.A.shape[1]))
                    C[:, -2*(n+m):] = np.hstack((A, B, -np.eye(n), np.zeros((n,m))))
                    d = np.zeros((n, 1))
                    constraints.add_equality(C, d)
                constraints = constraints.cartesian_product(X_N)

                # if x is inside mcais lqr = mpc
                if X_N.contains(x):
                    self.assertAlmostEqual(V_mpc, V_lqr)
                    for t in range(N):
                        np.testing.assert_array_almost_equal(u_mpc[t], u_lqr[t])

                # if x is outside mcais lqr > mpc
                elif V_mpc is not None:
                    self.assertTrue(V_mpc > V_lqr)
                    np.testing.assert_array_almost_equal(u_mpc[0], controller.feedback(x))

                    # simulate the open loop control
                    x_mpc = S.simulate(x, u_mpc)
                    for t in range(N):
                        self.assertTrue(X.contains(x_mpc[t]))
                        self.assertTrue(U.contains(u_mpc[t]))
                    self.assertTrue(X_N.contains(x_mpc[N]))

                    # check feasibility
                    xu_mpc = np.vstack([np.vstack((x_mpc[t], u_mpc[t])) for t in range(N)])
                    xu_mpc = np.vstack((xu_mpc, x_mpc[N]))
                    self.assertTrue(constraints.contains(xu_mpc))

                # else certify empyness of the constraint set
                else:
                    self.assertTrue(controller.feedback(x) is None)
                    self.assertTrue(constraints.empty)

    def test_explicit_solution(self):
        np.random.seed(1)

        # random linear system
        for i in range(10):

            # ensure controllability
            n = np.random.randint(2, 3)
            m = np.random.randint(1, 2)
            controllable = False
            while not controllable:
                A = np.random.rand(n,n)
                B = np.random.rand(n,m)
                S = LinearSystem(A, B)
                controllable = S.controllable

            # stage constraints
            x_min = -np.random.rand(n,1)
            x_max = np.random.rand(n,1)
            u_min = -np.random.rand(m,1)
            u_max = np.random.rand(m,1)
            X = Polyhedron.from_bounds(x_min, x_max)
            U = Polyhedron.from_bounds(u_min, u_max)
            D = X.cartesian_product(U)

            # assemble controller
            N = 10
            Q = np.eye(n)
            R = np.eye(m)
            P, K = S.solve_dare(Q, R)
            X_N = S.mcais(K, D)
            controller = ModelPredictiveController(S, N, Q, R, P, D, X_N)

            # store explicit solution
            controller.store_explicit_solution()

            # simulate
            for j in range(10):

                # intial state
                x = np.multiply(np.random.rand(n,1), x_max - x_min) + x_min

                # implicit vs explicit mpc
                u_implicit, V_implicit = controller.feedforward(x)
                u_explicit, V_explicit = controller.feedforward_explicit(x)

                # if feasible
                if V_implicit is not None:
                    self.assertAlmostEqual(V_implicit, V_explicit)
                    for t in range(N):
                        np.testing.assert_array_almost_equal(u_implicit[t], u_explicit[t])
                    np.testing.assert_array_almost_equal(
                        controller.feedback(x),
                        controller.feedback_explicit(x)
                        )

                # if unfeasible
                else:
                    self.assertTrue(V_explicit is None)
                    self.assertTrue(u_explicit is None)
                    self.assertTrue(controller.feedback_explicit(x) is None)

class testHybridModelPredictiveController(unittest.TestCase):

    def test_feedforward_feedback_and_get_mpqp(self):

        # numeric parameters
        m = 1.
        l = 1.
        g = 10.
        k = 100.
        d = .1
        h = .01

        # discretization method
        method = 'explicit_euler'

        # dynamics n.1
        A1 = np.array([[0., 1.],[g/l, 0.]])
        B1 = np.array([[0.],[1/(m*l**2.)]])
        S1 = LinearSystem.from_continuous(A1, B1, h, method)

        # dynamics n.2
        A2 = np.array([[0., 1.],[g/l-k/m, 0.]])
        B2 = B1
        c2 = np.array([[0.],[k*d/(m*l)]])
        S2 = AffineSystem.from_continuous(A2, B2, c2, h, method)

        # list of dynamics
        S_list = [S1, S2]

        # state domain n.1
        x1_min = np.array([[-2.*d/l],[-1.5]])
        x1_max = np.array([[d/l], [1.5]])
        X1 = Polyhedron.from_bounds(x1_min, x1_max)

        # state domain n.2
        x2_min = np.array([[d/l], [-1.5]])
        x2_max = np.array([[2.*d/l],[1.5]])
        X2 = Polyhedron.from_bounds(x2_min, x2_max)

        # input domain
        u_min = np.array([[-4.]])
        u_max = np.array([[4.]])
        U = Polyhedron.from_bounds(u_min, u_max)

        # domains
        D1 = X1.cartesian_product(U)
        D2 = X2.cartesian_product(U)
        D_list = [D1, D2]

        # pwa system
        S = PieceWiseAffineSystem(S_list, D_list)

        # controller parameters
        N = 20
        Q = np.eye(S.nx)
        R = np.eye(S.nu)

        # terminal set and cost
        P, K = S1.solve_dare(Q, R)
        X_N = S1.mcais(K, D1)

        # hybrid MPC controller
        controller = HybridModelPredictiveController(S, N, Q, R, P, X_N)

        # compare with lqr
        x0 = np.array([[.0],[.6]])
        self.assertTrue(X_N.contains(x0))
        V_lqr = .5*x0.T.dot(P).dot(x0)[0,0]
        x_lqr = [x0]
        u_lqr = []
        for t in range(N):
            u_lqr.append(K.dot(x_lqr[t]))
            x_lqr.append((S1.A + S1.B.dot(K)).dot(x_lqr[t]))
        u_hmpc, x_hmpc, ms_hmpc, V_hmpc = controller.feedforward(x0)
        np.testing.assert_array_almost_equal(
            np.vstack((u_lqr)),
            np.vstack((u_hmpc))
            )
        np.testing.assert_array_almost_equal(
            np.vstack((x_lqr)),
            np.vstack((x_hmpc))
            )
        self.assertAlmostEqual(V_lqr, V_hmpc)
        self.assertTrue(all([m == 0 for m in ms_hmpc]))
        np.testing.assert_array_almost_equal(u_hmpc[0], controller.feedback(x0))

        # compare with linear mpc
        x0 = np.array([[.0],[.8]])
        self.assertFalse(X_N.contains(x0))
        linear_controller = ModelPredictiveController(S1, N, Q, R, P, D1, X_N)
        u_lmpc, V_lmpc = linear_controller.feedforward(x0)
        x_lmpc = S1.simulate(x0, u_lmpc)
        u_hmpc, x_hmpc, ms_hmpc, V_hmpc = controller.feedforward(x0)
        np.testing.assert_array_almost_equal(
            np.vstack((u_lmpc)),
            np.vstack((u_hmpc))
            )
        np.testing.assert_array_almost_equal(
            np.vstack((x_lmpc)),
            np.vstack((x_hmpc))
            )
        self.assertAlmostEqual(V_lmpc, V_hmpc)
        self.assertTrue(all([m == 0 for m in ms_hmpc]))
        np.testing.assert_array_almost_equal(u_hmpc[0], controller.feedback(x0))

        # test get mpqp
        mpqp = controller.get_mpqp(ms_hmpc)
        sol = mpqp.solve(x0)
        np.testing.assert_array_almost_equal(
            np.vstack((u_lmpc)),
            sol['argmin']
            )
        self.assertAlmostEqual(V_lmpc, sol['min'])

        # with change of the mode sequence
        x0 = np.array([[.08],[.8]])
        u_hmpc, x_hmpc, ms_hmpc, V_hmpc = controller.feedforward(x0)
        self.assertTrue(sum(ms_hmpc) >= 1)
        mpqp = controller.get_mpqp(ms_hmpc)
        sol = mpqp.solve(x0)
        np.testing.assert_array_almost_equal(
            np.vstack((u_hmpc)),
            sol['argmin']
            )
        self.assertAlmostEqual(V_hmpc, sol['min'])

if __name__ == '__main__':
    unittest.main()