import unittest
import numpy as np
from mpc_tools.dynamical_systems import DTLinearSystem, DTPWASystem, DTAffineSystem, plot_state_space_trajectory
import matplotlib.pyplot as plt
import mpc_tools.mpcqp as mqp
from mpc_tools.optimization.mpqpsolver import CriticalRegion
from mpc_tools.geometry import Polytope
from mpc_tools.control import MPCController


class TestDynamicalSystems(unittest.TestCase):

    def test_DTLinearSystem(self):

        # constinuous time double integrator
        A = np.array([[0., 1.],[0., 0.]])
        B = np.array([[0.],[1.]])
        t_s = 1.

        # discrete time from continuous
        sys = DTLinearSystem.from_continuous(t_s, A, B)
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

    def test_DTAffineSystem(self):

        # constinuous time double integrator
        A = np.array([[0., 1.],[0., 0.]])
        B = np.array([[0.],[1.]])
        c = np.array([[1.],[1.]])
        t_s = 1.

        # discrete time from continuous
        sys = DTAffineSystem.from_continuous(t_s, A, B, c)
        A_discrete = np.eye(2) + A*t_s
        B_discrete = B*t_s + np.array([[0.,t_s**2/2.],[0.,0.]]).dot(B)
        c_discrete = c*t_s + np.array([[0.,t_s**2/2.],[0.,0.]]).dot(c)
        self.assertTrue(all(np.isclose(sys.A.flatten(), A_discrete.flatten())))
        self.assertTrue(all(np.isclose(sys.B.flatten(), B_discrete.flatten())))
        self.assertTrue(all(np.isclose(sys.c.flatten(), c_discrete.flatten())))
        return

    def test_DTPWASystem(self):

        # tests the QP condensing thorugh the simulate method

        t_s = .1

        A_1 = np.array([[0., 1.],[10., 0.]])
        B_1 = np.array([[0.],[1.]])
        c_1 = np.array([[0.],[0.]])
        sys_1 = DTAffineSystem.from_continuous(t_s, A_1, B_1, c_1)

        A_2 = np.array([[0., 1.],[-100., 0.]])
        B_2 = B_1
        c_2 = np.array([[0.],[10.]])
        sys_2 = DTAffineSystem.from_continuous(t_s, A_2, B_2, c_2)

        sys = [sys_1, sys_2]


        x_max_1 = np.array([[.1], [1.5]])
        x_max_2 = np.array([[.2],[x_max_1[1,0]]])
        x_min_1 = -x_max_2
        x_min_2 = np.array([[x_max_1[0,0]], [x_min_1[1,0]]])

        X_1 = Polytope.from_bounds(x_max_1, x_min_1)
        X_1.assemble()
        X_2 = Polytope.from_bounds(x_max_2, x_min_2)
        X_2.assemble()

        X = [X_1, X_2]

        u_max = np.array([[4.]])
        u_min = -u_max

        U_1 = Polytope.from_bounds(u_max, u_min)
        U_1.assemble()
        U_2 = U_1
        U = [U_1, U_2]

        sys = DTPWASystem(sys, X, U)
        
        x_0 = np.array([[.0],[.5]])
        N = 10
        u = [np.ones((1,1)) for i in range(N)]
        x_sim, switching_sequence = sys.simulate(x_0, u)
        x_sim = np.vstack(x_sim)
        free_evolution, forced_evolution, offset_evolution = sys.evolution_matrices(switching_sequence)
        x_evo = free_evolution.dot(x_0) + forced_evolution.dot(np.vstack(u)) + offset_evolution
        x_evo = np.vstack((x_0, x_evo))
        self.assertTrue(all(np.isclose(x_sim.flatten(), x_evo.flatten())))

if __name__ == '__main__':
    unittest.main()