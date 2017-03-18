import unittest
import numpy as np
from mpc_tools.optimization import linear_program, quadratic_program

class TestMPCTools(unittest.TestCase):

    def test_linear_program(self):

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
        f = np.ones(2)
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

        # # I don't know why this LP gives nan !!!
        # f = np.array([[1.e-7],
        #        [0.]])
        # A = np.array([[  9.73537770e-01,   2.28526169e-01],
        #        [ -6.95497931e-01,  -7.18528098e-01],
        #        [  9.37562537e-01,   3.47816748e-01],
        #        [  9.62584453e-01,   2.70981865e-01],
        #        [  9.66326722e-01,   2.57318219e-01],
        #        [ -9.62584453e-01,  -2.70981865e-01]])
        # b = np.array([[-0.30239749],
        #        [ 0.48801894],
        #        [-0.31104601],
        #        [-0.31763297],
        #        [-0.31338212],
        #        [ 0.32430017]])
        # print linear_program(f, A, b)

    def test_quadratic_program(self):

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

