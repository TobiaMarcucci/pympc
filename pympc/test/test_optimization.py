import unittest
import numpy as np
from pympc.optimization.pnnls import linear_program as lp_pnnls
from pympc.optimization.gurobi import linear_program as lp_gurobi
from pympc.optimization.gurobi import quadratic_program as qp_gurobi

class TestMPCTools(unittest.TestCase):

    def test_linear_program(self):

        for linear_program in [lp_pnnls, lp_gurobi]:

            # trivial lp
            f = np.ones((2,1))
            A = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
            b = np.array([[5.],[5.],[1.],[1.]])
            true_x_min = np.array([[-5.],[-1.]])
            true_cost_min = -6.
            [x_min, cost_min] = linear_program(f, A, b)
            self.assertTrue(np.isclose(cost_min, true_cost_min))
            self.assertTrue(all(np.isclose(x_min, true_x_min)))

            # unfeasible lp
            f = np.ones((2,1))
            A = np.array([[0.,1.],[0.,-1.]])
            b = np.array([[0.],[-1.]])
            [x_min, cost_min] = linear_program(f, A, b)
            self.assertTrue(np.isnan(cost_min))
            self.assertTrue(all(np.isnan(x_min)))

            # unbounded lp
            f = np.ones((2,1))
            A = np.array([[1.,0.],[0.,1.],[0.,-1.]])
            b = np.array([[0.],[1.],[1.]])
            true_x_min = np.array([[-np.inf], [-1.]])
            true_cost_min = -np.inf
            [x_min, cost_min] = linear_program(f, A, b)
            self.assertTrue(np.isnan(cost_min))
            self.assertTrue(all(np.isnan(x_min)))

            # bounded lp with unbounded domain
            f = np.array([[0.],[1.]])
            A = np.array([[0.,-1.]])
            b = np.zeros((1,1))
            true_cost_min = 0.
            true_x1_min = np.zeros((1,1))
            [x_min, cost_min] = linear_program(f, A, b)
            self.assertTrue(np.isclose(cost_min, true_cost_min))
            self.assertTrue(np.isclose(x_min[1], true_x1_min))

            # lp with equalities
            f = np.ones((2,1))
            A = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
            b = np.array([[5.],[5.],[1.],[1.]])
            C = np.array([[1.,0.]])
            d = np.array([[0.]])
            true_x_min = np.array([[0.],[-1.]])
            true_cost_min = -1.
            [x_min, cost_min] = linear_program(f, A, b, C, d)
            self.assertTrue(np.isclose(cost_min, true_cost_min))

    def test_quadratic_program(self):

        for quadratic_program in [qp_gurobi]:

            # trivial qp
            H = np.eye(2)
            f = np.zeros((2,1))
            A = np.array([[1.,0.],[0.,1.]])
            b = np.array([[-1.],[-1.]])
            true_x_min = np.array([[-1.],[-1.]])
            true_cost_min = 1.
            [x_min, cost_min] = quadratic_program(H, f, A, b)
            self.assertTrue(np.isclose(cost_min, true_cost_min))
            self.assertTrue(all(np.isclose(x_min, true_x_min)))

            # unfeasible qp
            H = np.eye(2)
            f = np.zeros((2,1))
            A = np.array([[0.,1.],[0.,-1.]])
            b = np.array([[0.],[-1.]])
            [x_min, cost_min] = quadratic_program(H, f, A, b)
            self.assertTrue(np.isnan(cost_min))
            self.assertTrue(all(np.isnan(x_min)))

if __name__ == '__main__':
    unittest.main()

