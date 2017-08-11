import unittest
import numpy as np
from pympc.optimization.pnnls import linear_program as lp_pnnls
from pympc.optimization.pnnls import quadratic_program as qp_pnnls
from pympc.optimization.gurobi import linear_program as lp_gurobi
from pympc.optimization.gurobi import quadratic_program as qp_gurobi

class TestMPCTools(unittest.TestCase):
    def test_linear_program(self):
        for linear_program in [lp_pnnls, lp_gurobi]:

            # trivial lp
            f = np.ones((2,1))
            A = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
            b = np.array([[5.],[5.],[1.],[1.]])
            true_argmin = np.array([[-5.],[-1.]])
            sol = linear_program(f, A, b)
            self.assertTrue(np.allclose(sol.argmin, np.array([[-5.],[-1.]])))
            self.assertTrue(np.isclose(sol.min, -6.))
            self.assertEqual(sol.active_set, [1, 3])
            self.assertTrue(np.allclose(sol.inequality_multipliers, np.array([[0.],[1.],[0.],[1.]])))
            self.assertFalse(sol.primal_degenerate)
            self.assertFalse(sol.dual_degenerate)

            # unfeasible lp
            f = np.ones((2,1))
            A = np.array([[0.,1.],[0.,-1.]])
            b = np.array([[0.],[-1.]])
            sol = linear_program(f, A, b)
            self.assertTrue(np.isnan(sol.min))
            self.assertTrue(all(np.isnan(sol.argmin)))
            self.assertTrue(all(np.isnan(sol.inequality_multipliers)))
            self.assertTrue(sol.active_set is None)
            self.assertTrue(sol.primal_degenerate is None)
            self.assertTrue(sol.dual_degenerate is None)

            # unbounded lp
            f = np.ones((2,1))
            A = np.array([[1.,0.],[0.,1.],[0.,-1.]])
            b = np.array([[0.],[1.],[1.]])
            sol = linear_program(f, A, b)
            self.assertTrue(np.isnan(sol.min))
            self.assertTrue(all(np.isnan(sol.argmin)))
            self.assertTrue(all(np.isnan(sol.inequality_multipliers)))
            self.assertTrue(sol.active_set is None)
            self.assertTrue(sol.primal_degenerate is None)
            self.assertTrue(sol.dual_degenerate is None)

            # bounded lp with unbounded domain
            f = np.array([[0.],[1.]])
            A = np.array([[0.,-1.]])
            b = np.zeros((1,1))
            sol = linear_program(f, A, b)
            self.assertTrue(np.isclose(sol.argmin[1,0], 0.))
            self.assertTrue(np.isclose(sol.min, 0.))
            self.assertEqual(sol.active_set, [0])
            self.assertTrue(np.allclose(sol.inequality_multipliers, np.array([[1.]])))
            self.assertFalse(sol.primal_degenerate)
            self.assertTrue(sol.dual_degenerate)

            # lp with equalities
            f = np.ones((2,1))
            A = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
            b = np.array([[5.],[5.],[1.],[1.]])
            C = np.array([[1.,0.]])
            d = np.array([[0.]])
            sol = linear_program(f, A, b, C, d)
            self.assertTrue(np.allclose(sol.argmin, np.array([[0.],[-1.]])))
            self.assertTrue(np.isclose(sol.min, -1.))
            self.assertEqual(sol.active_set, [3])
            self.assertTrue(np.allclose(sol.inequality_multipliers, np.array([[0.],[0.],[0.],[1.]])))
            self.assertTrue(np.allclose(sol.equality_multipliers, np.array([[-1.]])))
            self.assertFalse(sol.primal_degenerate)
            self.assertFalse(sol.dual_degenerate)

            # 3d LPs
            A = np.array([
                  [1., 1., 1.],
                  [-1., 1., 1.],
                  [1., -1., 1.],
                  [1., 1., -1.],
                  [-1., -1., 1.],
                  [1., -1., -1.],
                  [-1., 1., -1.],
                  [-1., -1., -1.],
                  [1., 0., 0.]
                  ])
            b = np.vstack((np.ones((8,1)), np.array([[.5]])))
            f = np.array([[-1.],[0.],[0.]])
            sol = linear_program(f, A, b)
            self.assertTrue(np.isclose(sol.argmin[0,0], .5))
            self.assertTrue(np.isclose(sol.min, -.5))
            self.assertTrue(8 in sol.active_set)
            self.assertTrue(np.allclose(sol.inequality_multipliers, np.vstack((np.zeros((8,1)), np.array([[1.]])))))
            self.assertFalse(sol.primal_degenerate)
            self.assertTrue(sol.dual_degenerate)
            f = np.array([[-1.],[-.1],[0.]])
            sol = linear_program(f, A, b)
            self.assertTrue(np.allclose(sol.argmin, np.array([[.5],[.5],[0.]])))
            self.assertTrue(np.isclose(sol.min, -.55))
            self.assertTrue(sol.active_set, [0,3,8])
            self.assertTrue(np.allclose(sol.inequality_multipliers, np.array([[.05],[0.],[0.],[.05],[0.],[0.],[0.],[0.],[.9]])))
            self.assertFalse(sol.primal_degenerate)
            self.assertFalse(sol.dual_degenerate)
            f = np.array([[1.],[1.],[0.]])
            sol = linear_program(f, A, b)
            self.assertTrue(np.isclose(sol.argmin[0,0] + sol.argmin[1,0], -1.))
            self.assertTrue(np.isclose(sol.min, -1.))
            self.assertTrue(4 in sol.active_set)
            self.assertTrue(7 in sol.active_set)
            self.assertTrue(np.isclose(sol.inequality_multipliers[4,0], .5))
            self.assertTrue(np.isclose(sol.inequality_multipliers[7,0], .5))
            self.assertTrue(sol.dual_degenerate)

        # random LPs
        for i in range(100):
            n_variables = np.random.randint(10, 100)
            n_ineq = np.random.randint(10, 100)
            n_eq = np.random.randint(1, 9)
            A = np.random.randn(n_ineq, n_variables)
            b = np.random.rand(n_ineq, 1)
            C = np.random.randn(n_eq, n_variables)
            d = np.random.rand(n_eq, 1)
            f = np.random.randn(n_variables, 1)
            sol_pnnls = lp_pnnls(f, A, b, C, d)
            sol_gurobi = lp_gurobi(f, A, b, C, d)
            if np.isnan(sol_pnnls.min):
                self.assertTrue(np.isnan(sol_gurobi.min))
                self.assertTrue(all(np.isnan(sol_pnnls.argmin)))
                self.assertTrue(all(np.isnan(sol_gurobi.argmin)))
                self.assertTrue(all(np.isnan(sol_pnnls.inequality_multipliers)))
                self.assertTrue(all(np.isnan(sol_gurobi.inequality_multipliers)))
                self.assertTrue(all(np.isnan(sol_pnnls.equality_multipliers)))
                self.assertTrue(all(np.isnan(sol_gurobi.equality_multipliers)))
                self.assertTrue(sol_pnnls.active_set is None)
                self.assertTrue(sol_gurobi.active_set is None)
                self.assertTrue(sol_pnnls.primal_degenerate is None)
                self.assertTrue(sol_gurobi.primal_degenerate is None)
                self.assertTrue(sol_pnnls.dual_degenerate is None)
                self.assertTrue(sol_gurobi.dual_degenerate is None)
            else:
                self.assertTrue(np.allclose(sol_pnnls.argmin, sol_gurobi.argmin))
                self.assertTrue(np.isclose(sol_pnnls.min, sol_gurobi.min))
                self.assertTrue(sol_pnnls.active_set, sol_gurobi.active_set)
                self.assertTrue(np.allclose(sol_gurobi.inequality_multipliers, sol_gurobi.inequality_multipliers))
                self.assertTrue(np.allclose(sol_gurobi.equality_multipliers, sol_gurobi.equality_multipliers))
                self.assertTrue(sol_pnnls.primal_degenerate == sol_gurobi.primal_degenerate)
                self.assertTrue(sol_pnnls.dual_degenerate == sol_gurobi.dual_degenerate)

    def test_quadratic_program(self):

        for quadratic_program in [qp_gurobi, qp_pnnls]:
            # trivial qp
            H = np.eye(2)
            f = np.zeros((2,1))
            A = np.array([[1.,0.],[0.,1.]])
            b = np.array([[-1.],[-1.]])
            true_x_min = np.array([[-1.],[-1.]])
            true_cost_min = 1.
            sol = quadratic_program(H, f, A, b)
            self.assertTrue(np.isclose(sol.min, true_cost_min))
            self.assertTrue(all(np.isclose(sol.argmin, true_x_min)))

            # unfeasible qp
            H = np.eye(2)
            f = np.zeros((2,1))
            A = np.array([[0.,1.],[0.,-1.]])
            b = np.array([[0.],[-1.]])
            sol = quadratic_program(H, f, A, b)
            self.assertTrue(np.isnan(sol.min))
            self.assertTrue(all(np.isnan(sol.argmin)))

        # random qps
        for i in range(100):
            n_variables = np.random.randint(10, 100)
            n_ineq = np.random.randint(10, 100)
            n_eq = np.random.randint(1, 9)
            f = np.random.randn(n_variables, 1)
            H = np.random.random((n_variables,n_variables))
            H = H.T.dot(H)+np.eye(n_variables)*1.e-3
            A = np.random.randn(n_ineq, n_variables)
            b = np.random.rand(n_ineq, 1)
            #C = np.random.randn(n_eq, n_variables)
            #d = np.random.rand(n_eq, 1)
            sol_pnnls = qp_pnnls(H, f, A, b)
            sol_gurobi = qp_gurobi(H, f, A, b)
            if np.isnan(sol_pnnls.min):
                self.assertTrue(np.isnan(sol_gurobi.min))
                self.assertTrue(all(np.isnan(sol_pnnls.argmin)))
                self.assertTrue(all(np.isnan(sol_gurobi.argmin)))
            else:
                self.assertTrue(np.allclose(sol_pnnls.argmin, sol_gurobi.argmin,1.e-3,1.e-5))
                self.assertTrue(np.isclose(sol_pnnls.min, sol_gurobi.min,1.e-3,1.e-5))

if __name__ == '__main__':
    unittest.main()
