import unittest
import numpy as np
from pympc.dynamical_systems import DTLinearSystem, DTPWASystem, DTAffineSystem, dare, moas_closed_loop
from pympc.geometry import Polytope

class TestDynamicalSystems(unittest.TestCase):

    def test_DTLinearSystem(self):

        # constinuous time double integrator
        A = np.array([[0., 1.],[0., 0.]])
        B = np.array([[0.],[1.]])
        t_s = 1.

        # discrete time from continuous
        sys = DTLinearSystem.from_continuous(A, B, t_s)
        A_discrete = np.eye(2) + A*t_s
        B_discrete = B*t_s + np.array([[0.,t_s**2/2.],[0.,0.]]).dot(B)
        self.assertTrue(all(np.isclose(sys.A.flatten(), A_discrete.flatten())))
        self.assertTrue(all(np.isclose(sys.B.flatten(), B_discrete.flatten())))

        # simulation forced dynamics
        x0 = np.array([[0.],[1.]])
        N = 10
        u = np.array([[1.]])
        u_list = [u] * N
        x_list = sys.simulate(x0, u_list)
        real_x_list = [[x0[0] + x0[1]*i*t_s + u[0,0]*(i*t_s)**2/2., x0[1] + u*i*t_s] for i in range(0,N+1)]
        self.assertTrue(all(np.isclose(x_list, real_x_list).flatten()))

        # test maximum output admissible set
        Q = np.eye(2)
        R = np.eye(1)
        P, K = dare(sys.A, sys.B, Q, R)
        x_max = np.ones((2,1))
        x_min = - x_max
        X = Polytope.from_bounds(x_max, x_min)
        X.assemble()
        u_max = np.ones((1,1))
        u_min = - u_max
        U = Polytope.from_bounds(u_max, u_min)
        U.assemble()
        moas = moas_closed_loop(A, B, K, X, U)
        sys_cl = DTLinearSystem(A + B.dot(K), np.zeros((2,1)))
        v0_max = max([v[0] for v in moas.vertices])
        v1_max = max([v[1] for v in moas.vertices])
        v0_min = min([v[0] for v in moas.vertices])
        v1_min = min([v[1] for v in moas.vertices])
        for x0 in list(np.linspace(v0_min, v0_max, 100)):
            for x1 in list(np.linspace(v1_min, v1_max, 100)):
                x_init = np.array([[x0],[x1]])
                if moas.applies_to(x_init):
                    u = np.zeros((1,1))
                    u_list = [u] * N
                    x_list = sys_cl.simulate(x_init, u_list)
                    for x in x_list:
                        self.assertTrue(moas.applies_to(x))



    def test_DTAffineSystem(self):

        # constinuous time double integrator
        A = np.array([[0., 1.],[0., 0.]])
        B = np.array([[0.],[1.]])
        c = np.array([[1.],[1.]])
        t_s = 1.

        # discrete time from continuous
        sys = DTAffineSystem.from_continuous(A, B, c, t_s)
        A_discrete = np.eye(2) + A*t_s
        B_discrete = B*t_s + np.array([[0.,t_s**2/2.],[0.,0.]]).dot(B)
        c_discrete = c*t_s + np.array([[0.,t_s**2/2.],[0.,0.]]).dot(c)
        self.assertTrue(all(np.isclose(sys.A.flatten(), A_discrete.flatten())))
        self.assertTrue(all(np.isclose(sys.B.flatten(), B_discrete.flatten())))
        self.assertTrue(all(np.isclose(sys.c.flatten(), c_discrete.flatten())))
        return

    def test_DTPWASystem(self):

        # PWA dynamics
        t_s = .1
        A_1 = np.array([[0., 1.],[10., 0.]])
        B_1 = np.array([[0.],[1.]])
        c_1 = np.array([[0.],[0.]])
        sys_1 = DTAffineSystem.from_continuous(A_1, B_1, c_1, t_s)
        A_2 = np.array([[0., 1.],[-100., 0.]])
        B_2 = B_1
        c_2 = np.array([[0.],[10.]])
        sys_2 = DTAffineSystem.from_continuous(A_2, B_2, c_2, t_s)
        sys = [sys_1, sys_2]

        # PWA state domains
        x_max_1 = np.array([[.1], [1.5]])
        x_max_2 = np.array([[.2],[x_max_1[1,0]]])
        x_min_1 = -x_max_2
        x_min_2 = np.array([[x_max_1[0,0]], [x_min_1[1,0]]])
        X_1 = Polytope.from_bounds(x_max_1, x_min_1)
        X_1.assemble()
        X_2 = Polytope.from_bounds(x_max_2, x_min_2)
        X_2.assemble()
        X = [X_1, X_2]

        # PWA input domains
        u_max = np.array([[4.]])
        u_min = -u_max
        U_1 = Polytope.from_bounds(u_max, u_min)
        U_1.assemble()
        U_2 = U_1
        U = [U_1, U_2]
        sys = DTPWASystem(sys, X, U)

        # simualate
        N = 10
        x_0 = np.array([[.0],[.5]])
        u_list = [np.ones((1,1)) for i in range(N)]
        x_sim_1, switching_sequence = sys.simulate(x_0, u_list)
        x_sim_1 = np.vstack(x_sim_1)

        # condense for the simulated switching sequence
        A_bar, B_bar, c_bar = sys.condense(switching_sequence)
        x_sim_2 = A_bar.dot(x_0) + B_bar.dot(np.vstack(u_list)) + c_bar

        # compare the 2 simulations
        self.assertTrue(all(np.isclose(x_sim_1.flatten(), x_sim_2.flatten())))

if __name__ == '__main__':
    unittest.main()