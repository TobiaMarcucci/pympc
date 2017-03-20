import unittest
import numpy as np
#import sys
import mpc_tools as mpc
import mpc_tools.mpcqp as mqp

class TestMPCTools(unittest.TestCase):

    def test_DTLinearSystem(self):

        # constinuous time double integrator
        A = np.array([[0., 1.],[0., 0.]])
        B = np.array([[0.],[1.]])
        t_s = 1.

        # discrete time from continuous
        sys = mpc.DTLinearSystem.from_continuous(t_s, A, B)
        A_discrete = np.eye(2) + A*t_s
        B_discrete = B*t_s + np.array([[0.,t_s**2/2.],[0.,0.]]).dot(B)
        self.assertTrue(all(np.isclose(sys.A.flatten(), A_discrete.flatten())))
        self.assertTrue(all(np.isclose(sys.B.flatten(), B_discrete.flatten())))

        # simulation free dynamics
        x0 = np.array([[0.],[1.]])
        N = 10
        x_trajectory = sys.simulate(x0, N)
        real_x_trajectory = [[x0[0] + x0[1]*i*t_s, x0[1]] for i in range(0,N+1)]
        self.assertTrue(all(np.isclose(x_trajectory, real_x_trajectory).flatten()))

        # simulation forced dynamics
        u = np.array([[1.]])
        u_sequence = [u] * N
        x_trajectory = sys.simulate(x0, N, u_sequence)
        real_x_trajectory = [[x0[0] + x0[1]*i*t_s + u[0,0]*(i*t_s)**2/2., x0[1] + u*i*t_s] for i in range(0,N+1)]
        self.assertTrue(all(np.isclose(x_trajectory, real_x_trajectory).flatten()))



    def test_CriticalRegion(self):

        # test candidate_active_sets method
        active_set = [0,3,4]
        minimal_facets = [0,1,2]
        coincident_facets = [[0],[1],[2,3],[2,3],[4]]
        minimal_coincident_facets = [coincident_facets[i] for i in minimal_facets]
        candidate_active_sets = mpc.CriticalRegion.candidate_active_sets(active_set, minimal_coincident_facets)
        true_candidate_active_sets = [[[3,4]],[[0,1,3,4]],[[0,2,4]]]
        self.assertEqual(true_candidate_active_sets, candidate_active_sets)

        # test candidate_active_sets method
        weakly_active_constraints = [0,4]
        candidate_active_sets = mpc.CriticalRegion.expand_candidate_active_sets(candidate_active_sets, weakly_active_constraints)
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
        sys = mpc.DTLinearSystem.from_continuous(t_s, A, B)

        # mpc controller
        N = 5
        Q = np.eye(A.shape[0])
        R = np.eye(B.shape[1])
        terminal_cost = 'dare'
        factory = mqp.MPCQPFactory(sys, N, Q, R, terminal_cost)
        x_max = np.array([[1.], [1.]])
        x_min = -x_max
        factory.add_state_bound(x_max, x_min)
        u_max = np.array([[1.]])
        u_min = -u_max
        factory.add_input_bound(u_max, u_min)
        terminal_constraint = 'moas'
        factory.set_terminal_constraint(terminal_constraint)
        qp = factory.assemble()
        controller = mpc.MPCController(qp, 1)

        # explicit vs implicit solution
        controller.compute_explicit_solution()
        print controller.critical_regions
        n_test = 100
        for i in range(0, n_test):
            x0 = np.random.rand(2,1)
            u_explicit = controller.feedforward_explicit(x0)
            u_implicit = controller.feedforward(x0)
            if any(np.isnan(u_explicit)) or any(np.isnan(u_implicit)):
                self.assertTrue(all(np.isnan(u_explicit)))
                self.assertTrue(all(np.isnan(u_implicit)))
            else:
                rel_toll = 5.e-2
                self.assertTrue(all(np.isclose(u_explicit, u_implicit, rel_toll).flatten()))

if __name__ == '__main__':
    unittest.main()