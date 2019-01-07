# external imports
import unittest
import numpy as np

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.dynamics.discrete_time_systems import LinearSystem, AffineSystem, PieceWiseAffineSystem
from pympc.control.hscc.controllers import HybridModelPredictiveController

class testHybridModelPredictiveController(unittest.TestCase):

    def test_feedforward_and_feedback(self):

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
        c2 = np.array([0., k*d/(m*l)])
        S2 = AffineSystem.from_continuous(A2, B2, c2, h, method)

        # list of dynamics
        S_list = [S1, S2]

        # state domain n.1
        x1_min = np.array([-2.*d/l, -1.5])
        x1_max = np.array([d/l, 1.5])
        X1 = Polyhedron.from_bounds(x1_min, x1_max)

        # state domain n.2
        x2_min = np.array([d/l, -1.5])
        x2_max = np.array([2.*d/l, 1.5])
        X2 = Polyhedron.from_bounds(x2_min, x2_max)

        # input domain
        u_min = np.array([-4.])
        u_max = np.array([4.])
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
        R = np.eye(S.nu)/100.

        # terminal set and cost
        P, K = S1.solve_dare(Q, R)
        X_N = S1.mcais(K, D1)

        # initial state
        x0 = np.array([.095,.2])

        # loop through all the norms
        for norm in ['inf', 'one', 'two']:
            inputs = []
            states = []
            mode_sequences = []
            costs = []

            # loop through all the mixed integer formulations
            for method in ['pf', 'ch', 'bm', 'mld']:

                # solve MICP
                controller = HybridModelPredictiveController(S, N, Q, R, P, X_N, method, norm)
                u, x, ms, V = controller.feedforward(x0, {'OutputFlag': 0})
                inputs.append(u)
                states.append(x)
                mode_sequences.append(ms)
                costs.append(V)

            # test inputs
            nu = 1
            for t in range(N):
                for i in range(nu):
                    u_t = [u[t][i] for u in inputs]
                    self.assertAlmostEqual(min(u_t), max(u_t), 3)

            # test states
            nx = 2
            for t in range(N+1):
                for i in range(nx):
                    x_t = [x[t][i] for x in states]
                    self.assertAlmostEqual(min(x_t), max(x_t), 3)

            # test mode sequences
            for t in range(N):
                ms_t = [ms[t] for ms in mode_sequences]
                self.assertEqual(min(ms_t), max(ms_t))

            # test costs
            self.assertAlmostEqual(min(costs), max(costs), 4)

if __name__ == '__main__':
    unittest.main()