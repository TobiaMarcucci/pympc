import unittest
import numpy as np
import pympc.dynamical_systems as ds
from pympc.optimization.mpqpsolver import CriticalRegion
from pympc.geometry.polytope import Polytope
from pympc.control import MPCController, MPCHybridController


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
        U = Polytope.from_bounds(u_min, u_max)
        U.assemble()
        x_max = np.array([[1.], [1.]])
        x_min = -x_max
        X = Polytope.from_bounds(x_min, x_max)
        X.assemble()
        X_N = ds.moas_closed_loop_from_orthogonal_domains(sys.A, sys.B, K, X, U)
        controller = MPCController(sys, N, objective_norm, Q, R, P, X, U, X_N)

        # explicit vs implicit solution
        controller.get_explicit_solution()
        n_test = 100
        for i in range(n_test):
            x0 = np.random.rand(2,1)
            u_explicit, V_explicit = controller.feedforward_explicit(x0)
            u_implicit, V_implicit = controller.feedforward(x0)
            u_explicit = np.vstack(u_explicit)
            u_implicit = np.vstack(u_implicit)
            if any(np.isnan(u_explicit)) or any(np.isnan(u_implicit)):
                self.assertTrue(all(np.isnan(u_explicit)))
                self.assertTrue(all(np.isnan(u_implicit)))
                self.assertTrue(np.isnan(V_explicit))
                self.assertTrue(np.isnan(V_implicit))
                # self.assertFalse(controller.condensed_program.feasible_set.applies_to(x0))
            else:
                self.assertTrue(all(np.isclose(u_explicit, u_implicit, rtol=1.e-4).flatten()))
                self.assertTrue(np.isclose(V_explicit, V_implicit, rtol=1.e-4))
                # self.assertTrue(controller.condensed_program.feasible_set.applies_to(x0))

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
        X_1 = Polytope.from_bounds(x_min_1, x_max_1)
        X_1.assemble()
        X_2 = Polytope.from_bounds(x_min_2, x_max_2)
        X_2.assemble()
        X = [X_1, X_2]

        # PWA input domains
        u_max = np.array([[1.]])
        u_min = -u_max
        U_1 = Polytope.from_bounds(u_min, u_max)
        U_1.assemble()
        U_2 = U_1
        U = [U_1, U_2]

        # PWA system
        pwa_sys = ds.DTPWASystem.from_orthogonal_domains(sys, X, U)

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
                x0 = np.random.rand(A_1.shape[0], 1)
                u_mip, _, ss, V_mip = controller.feedforward(x0)
                if not np.isnan(V_mip):
                    prog = controller.condense_program(ss)
                    u_condensed, V_condensed = prog.solve(x0, u_length=B_1.shape[1])
                    argmin_error = max([np.linalg.norm(u_mip[i] - u_condensed[i]) for i in range(len(u_mip))])
                    # if not argmin_error < 1.e-5:
                    #     print argmin_error, objective_norm
                    self.assertTrue(np.isclose(V_mip, V_condensed))
                    if objective_norm == 'two':
                        self.assertTrue(argmin_error < 1.e-4)

        # # backwards reachability analysis
        # n_test = 100
        # ss_list = []
        # for i in range(n_test):
        #     x0 = np.random.rand(A_1.shape[0], 1)
        #     ss = controller.feedforward(x0)[2]
        #     if not any(np.isnan(ss)) and ss not in ss_list:
        #         ss_list.append(ss)
        #         prog = controller.condense_program(ss)
        #         fs = controller.backwards_reachability_analysis(ss)
        #         fs_od = controller.backwards_reachability_analysis_from_orthogonal_domains(ss, X, U)
        #         self.assertTrue(len(fs.vertices), len(fs_od.vertices))
        #         for v in fs.vertices:
        #             self.assertTrue(any([np.allclose(v, v_od) for v_od in fs_od.vertices]))
        #         for j in range(n_test):
        #             x0 = np.random.rand(A_1.shape[0], 1)
        #             V_star = prog.solve(x0)[1]
        #             if np.isnan(V_star):
        #                 self.assertFalse(fs.applies_to(x0))
        #                 self.assertFalse(fs_od.applies_to(x0))
        #             else:
        #                 self.assertTrue(fs.applies_to(x0))
        #                 self.assertTrue(fs_od.applies_to(x0))

if __name__ == '__main__':
    unittest.main()