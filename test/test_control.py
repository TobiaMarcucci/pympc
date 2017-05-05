import unittest
import numpy as np
import mpc_tools.dynamical_systems as ds
from mpc_tools.optimization.mpqpsolver import CriticalRegion
from mpc_tools.geometry import Polytope
from mpc_tools.control import MPCController, MPCExplicitController, MPCHybridController


class TestMPCTools(unittest.TestCase):

    def test_CriticalRegion(self):

        # test candidate_active_sets method
        active_set = [0,3,4]
        minimal_facets = [0,1,2]
        coincident_facets = [[0],[1],[2,3],[2,3],[4]]
        minimal_coincident_facets = [coincident_facets[i] for i in minimal_facets]
        candidate_active_sets = CriticalRegion.candidate_active_sets(active_set, minimal_coincident_facets)
        true_candidate_active_sets = [[[3,4]],[[0,1,3,4]],[[0,2,4]]]
        self.assertEqual(true_candidate_active_sets, candidate_active_sets)

        # test candidate_active_sets method
        weakly_active_constraints = [0,4]
        candidate_active_sets = CriticalRegion.expand_candidate_active_sets(candidate_active_sets, weakly_active_constraints)
        true_candidate_active_sets = [
        [[3, 4], [0, 3, 4], [3], [0, 3]],
        [[0, 1, 3, 4], [1, 3, 4], [0, 1, 3], [1, 3]],
        [[0, 2, 4], [2, 4], [0, 2], [2]]
        ]
        self.assertEqual(true_candidate_active_sets, candidate_active_sets)

    def test_MPCController(self):

        # double integrator
        A = np.array([[0., 1.],[0., 0.]])
        B = np.array([[0.],[1.]])
        t_s = 1.
        sys = ds.DTLinearSystem.from_continuous(A, B, t_s)

        # mpc controller
        N = 5
        Q = np.eye(A.shape[0])
        R = np.eye(B.shape[1])
        objective_norm = 'two'
        P, K = ds.dare(sys.A, sys.B, Q, R)
        u_max = np.array([[1.]])
        u_min = -u_max
        U = Polytope.from_bounds(u_max, u_min)
        U.assemble()
        x_max = np.array([[1.], [1.]])
        x_min = -x_max
        X = Polytope.from_bounds(x_max, x_min)
        X.assemble()
        X_N = ds.moas_closed_loop(sys.A, sys.B, K, X, U)
        controller = MPCController(sys, N, objective_norm, Q, R, P, X, U, X_N)

        # explicit vs implicit solution + feasibility region
        mpqp = controller.condensed_program
        explicit_controller = MPCExplicitController(mpqp)
        n_test = 100
        for i in range(n_test):
            x0 = np.random.rand(2,1)
            u_explicit, V_explicit = controller.feedforward(x0)
            u_implicit, V_implicit = controller.feedforward(x0)
            if any(np.isnan(u_explicit)) or any(np.isnan(u_implicit)):
                self.assertTrue(all(np.isnan(u_explicit)))
                self.assertTrue(all(np.isnan(u_implicit)))
                self.assertTrue(np.isnan(V_explicit))
                self.assertTrue(np.isnan(V_implicit))
                self.assertFalse(controller.condensed_program.feasible_set.applies_to(x0))
            else:
                self.assertTrue(all(np.isclose(u_explicit, u_implicit).flatten()))
                self.assertTrue(np.isclose(V_explicit, V_implicit))
                self.assertTrue(controller.condensed_program.feasible_set.applies_to(x0))

    def test_MPCHybridController(self):

        # PWA dynamics
        t_s = .1
        A_1 = np.array([[0., 1.],[1., 0.]])
        B_1 = np.array([[0.],[1.]])
        c_1 = np.array([[0.],[0.]])
        sys_1 = ds.DTAffineSystem.from_continuous(A_1, B_1, c_1, t_s)
        A_2 = np.array([[0., 1.],[-1., 0.]])
        B_2 = B_1
        c_2 = np.array([[0.],[1.]])
        sys_2 = ds.DTAffineSystem.from_continuous(A_2, B_2, c_2, t_s)
        sys = [sys_1, sys_2]

        # PWA state domains
        x_max_1 = np.array([[1.], [1.5]])
        x_max_2 = np.array([[2.],[x_max_1[1,0]]])
        x_min_1 = -x_max_2
        x_min_2 = np.array([[x_max_1[0,0]], [x_min_1[1,0]]])
        X_1 = Polytope.from_bounds(x_max_1, x_min_1)
        X_1.assemble()
        X_2 = Polytope.from_bounds(x_max_2, x_min_2)
        X_2.assemble()
        X = [X_1, X_2]

        # PWA input domains
        u_max = np.array([[1.]])
        u_min = -u_max
        U_1 = Polytope.from_bounds(u_max, u_min)
        U_1.assemble()
        U_2 = U_1
        U = [U_1, U_2]

        # PWA system
        pwa_sys = ds.DTPWASystem(sys, X, U)

        # hybrid controller
        N = 10
        Q = np.eye(A_1.shape[0])
        R = np.eye(B_1.shape[1])
        P = Q
        X_N = X_1
        for objective_norm in ['one', 'two']:
            controller = MPCHybridController(pwa_sys, N, objective_norm, Q, R, P, X_N)

            # compare the cost of the MIP and the condensed program
            n_test = 100
            for i in range(n_test):
                x0 = np.random.rand(2,1)
                u_mip, V_mip, ss = controller.feedforward(x0)
                if not any(np.isnan(ss)):
                    prog = controller.condense_program(ss)
                    u_condensed, V_condensed = prog.solve(x0)
                    #print np.linalg.norm(u_mip-u_condensed)
                    #self.assertTrue(all(np.isclose(u_mip, u_condensed).flatten()))
                    self.assertTrue(np.isclose(V_mip, V_condensed))

if __name__ == '__main__':
    unittest.main()