# external imports
import unittest
import numpy as np
from copy import copy

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.dynamics.discrete_time_systems import LinearSystem
from pympc.control.controllers import ModelPredictiveController

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
            D = X.cartesian_product_with(U)

            # assemble controller
            N = np.random.randint(5, 10)
            Q = np.eye(n)
            R = np.eye(m)
            P, K = S.solve_dare(Q, R)
            X_N = S.mcais(K, D)[0]
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
                    constraints = constraints.cartesian_product_with(D)
                    C = np.zeros((n, constraints.A.shape[1]))
                    C[:, -2*(n+m):] = np.hstack((A, B, -np.eye(n), np.zeros((n,m))))
                    d = np.zeros((n, 1))
                    constraints.add_equality(C, d)
                constraints = constraints.cartesian_product_with(X_N)

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
            D = X.cartesian_product_with(U)

            # assemble controller
            N = 10
            Q = np.eye(n)
            R = np.eye(m)
            P, K = S.solve_dare(Q, R)
            X_N = S.mcais(K, D)[0]
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

if __name__ == '__main__':
    unittest.main()