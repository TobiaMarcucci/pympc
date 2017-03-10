import unittest
import numpy as np
import sys
import mpc_tools as mpc

# https://docs.python.org/2/library/unittest.html


class TestMPCTools(unittest.TestCase):

    def test_linear_program(self):

        # trivial lp
        f = np.ones((2,1))
        A = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        b = np.array([[5.],[5.],[1.],[1.]])
        true_x_min = np.array([[-5.],[-1.]])
        true_cost_min = -6.
        [x_min, cost_min, status] = mpc.linear_program(f, A, b)
        self.assertTrue(np.isclose(cost_min, true_cost_min))
        self.assertTrue(all(np.isclose(x_min, true_x_min)))
        self.assertEqual(status, 0)

        # unfeasible lp
        f = np.ones(2)
        A = np.array([[0.,1.],[0.,-1.]])
        b = np.array([[0.],[-1.]])
        [x_min, cost_min, status] = mpc.linear_program(f, A, b)
        self.assertTrue(np.isnan(cost_min))
        self.assertTrue(all(np.isnan(x_min)))
        self.assertEqual(status, 1)

        # unbounded lp
        f = np.ones((2,1))
        A = np.array([[1.,0.],[0.,1.],[0.,-1.]])
        b = np.array([[0.],[1.],[1.]])
        true_x_min = np.array([[-np.inf], [-1.]])
        true_cost_min = -np.inf
        [x_min, cost_min, status] = mpc.linear_program(f, A, b)
        self.assertTrue(np.isclose(cost_min, true_cost_min))
        self.assertTrue(all(np.isclose(x_min, true_x_min)))
        self.assertEqual(status, 2)

        # bounded lp with unbounded domain
        f = np.array([[0.],[1.]])
        A = np.array([[0.,-1.]])
        b = np.zeros((1,1))
        true_cost_min = 0.
        true_x1_min = np.zeros((1,1))
        [x_min, cost_min, status] = mpc.linear_program(f, A, b)
        self.assertTrue(np.isclose(cost_min, true_cost_min))
        self.assertTrue(np.isclose(x_min[1], true_x1_min))
        self.assertEqual(status, 0)



    def test_linear_program(self):

        # trivial qp
        H = np.eye(2)
        f = np.zeros((2,1))
        A = np.array([[1.,0.],[0.,1.]])
        b = np.array([[-1.],[-1.]])
        true_x_min = np.array([[-1.],[-1.]])
        true_cost_min = 1.
        [x_min, cost_min, status] = mpc.quadratic_program(H, f, A, b)
        self.assertTrue(np.isclose(cost_min, true_cost_min))
        self.assertTrue(all(np.isclose(x_min, true_x_min)))
        self.assertEqual(status, 0)

        # unfeasible qp
        H = np.eye(2)
        f = np.zeros((2,1))
        A = np.array([[0.,1.],[0.,-1.]])
        b = np.array([[0.],[-1.]])
        [x_min, cost_min, status] = mpc.quadratic_program(H, f, A, b)
        self.assertTrue(np.isnan(cost_min))
        self.assertTrue(all(np.isnan(x_min)))
        self.assertEqual(status, 1)



    def test_Polytope(self):

        # unbounded 1d
        lhs = np.array([[1.]])
        rhs = np.array([[1.]])
        p = mpc.Polytope(lhs,rhs)
        with self.assertRaises(ValueError):
            p.assemble()

        # unbounded 2d
        lhs = np.array([[1.,0.],[-1.,0.],[0.,1.]])
        rhs = np.array([[1.],[1.],[0.]])
        p = mpc.Polytope(lhs, rhs)
        with self.assertRaises(ValueError):
            p.assemble()

        # empty 1d
        lhs = np.array([[1.],[-1.]])
        rhs = np.array([[1.],[-2.]])
        p = mpc.Polytope(lhs,rhs)
        p.assemble()
        self.assertTrue(p.empty)

        # empty 2d
        lhs = np.array([[1.,0.],[-1.,0.],[0.,1.]])
        rhs = np.array([[1.],[-2.],[0.]])
        p = mpc.Polytope(lhs,rhs)
        p.assemble()
        self.assertTrue(p.empty)

        # bounded and not empty 1d
        lhs = np.array([[1.],[-1.],[-3.]])
        rhs = np.array([[1.],[1.],[10.]])
        p = mpc.Polytope(lhs,rhs)
        p.assemble()
        self.assertFalse(p.empty)
        self.assertTrue(p.bounded)

        # bounded and not empty 2d
        lhs = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        rhs = np.array([[1.],[1.],[1.],[1.]])
        p = mpc.Polytope(lhs,rhs)
        p.assemble()
        self.assertFalse(p.empty)
        self.assertTrue(p.bounded)

        # coincident and minimal facets 1d
        lhs = np.array([[1.],[1.+1.e-9],[-1.]])
        rhs = np.ones((3,1))
        p = mpc.Polytope(lhs,rhs)
        p.assemble()
        self.assertEqual(set(p.coincident_facets[0]), set([0,1]))
        self.assertEqual(set(p.coincident_facets[1]), set([0,1]))
        self.assertEqual(p.coincident_facets[2], [2])
        self.assertTrue(p.minimal_facets[0] in [0,1])
        self.assertEqual(p.minimal_facets[1], 2)

        # coincident and minimal facets 2d
        lhs = np.array([[1.,0.],[1.+1.e-9,0.+1.e-9],[-1.,0.],[0.,1.],[0.,-1.]])
        rhs = np.ones((5,1))
        p = mpc.Polytope(lhs,rhs)
        p.assemble()
        self.assertEqual(set(p.coincident_facets[0]), set([0,1]))
        self.assertEqual(set(p.coincident_facets[1]), set([0,1]))
        self.assertEqual(p.coincident_facets[2:], [[2],[3],[4]])
        self.assertTrue(p.minimal_facets[0] in [0,1])
        self.assertEqual(p.minimal_facets[1:], [2,3,4])

        # add_ functions 1d
        x_max = np.ones((1,1))
        x_min = -x_max
        p = mpc.Polytope.from_bounds(x_max, x_min)
        p.add_bounds(x_max/2., x_min/5.)
        lhs = np.array([[1.],[-1.]])
        rhs = np.array([[.1],[1.]])
        p.add_facets(lhs, rhs)
        p.assemble()
        self.assertFalse(p.empty)
        self.assertTrue(p.bounded)

        # vertices 1d
        true_vertices = [[.1],[-.2]]
        self.assertTrue(all([any(np.all(np.isclose(vertex, true_vertices),axis=1)) for vertex in p.vertices]))

        # facet centers 1d
        true_facet_centers = true_vertices
        self.assertTrue(all([any(np.all(np.isclose(facet_center, true_facet_centers),axis=1)) for facet_center in p.facet_centers]))

        # minimal facets 1d
        true_minimal_facets = [3, 4]
        self.assertEqual(p.minimal_facets, true_minimal_facets)
        e = [(p.lhs_min[i,:]*p.facet_centers[i] - p.rhs_min[i])[0] for i in range(0,len(true_minimal_facets))]
        self.assertTrue(np.isclose(np.linalg.norm(e),0.))

        # add_ functions 2d
        x_max = np.ones((2,1))
        x_min = -x_max
        p = mpc.Polytope.from_bounds(x_max, x_min)
        p.add_bounds(x_max/2.,x_min/5.)
        lhs = np.array([[1.,0.],[-1.,0.]])
        rhs = np.array([[.1],[1.]])
        p.add_facets(lhs, rhs)
        p.assemble()
        self.assertFalse(p.empty)
        self.assertTrue(p.bounded)

        # vertices 2d
        true_vertices = [[.1,.5],[-.2,.5],[.1,-.2],[-.2,-.2]]
        self.assertTrue(all([any(np.all(np.isclose(vertex, true_vertices),axis=1)) for vertex in p.vertices]))

        # facet centers 2d
        true_facet_centers = [[-.05,.5],[-.2,.15],[-.05,-.2],[.1,.15]]
        self.assertTrue(all([any(np.all(np.isclose(facet_center, true_facet_centers),axis=1)) for facet_center in p.facet_centers]))

        # minimal facets 2d
        true_minimal_facets = [5,6,7,8]
        self.assertEqual(p.minimal_facets, true_minimal_facets)
        e = [(p.lhs_min[i,:].dot(p.facet_centers[i]) - p.rhs_min[i])[0] for i in range(0,len(true_minimal_facets))]
        self.assertTrue(np.isclose(np.linalg.norm(e),0.))

        return



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
        controller = mpc.MPCController(sys, N, Q, R, terminal_cost)
        x_max = np.array([[1.], [1.]])
        x_min = -x_max
        controller.add_state_bound(x_max, x_min)
        u_max = np.array([[1.]])
        u_min = -u_max
        controller.add_input_bound(u_max, u_min)
        terminal_constraint = 'moas'
        controller.set_terminal_constraint(terminal_constraint)
        controller.assemble()

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