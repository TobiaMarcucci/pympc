# external imports
import numpy as np

# internal inputs
from pympc.optimization.pnnls import linear_program, quadratic_program

class ContinuousProgram():

    def __init__(self):

        # initilize LP matrices
        self.H = np.eye(0)
        self.f = np.eye(0)
        self.g = np.eye(0)
        self.A = np.eye(0)
        self.b = np.eye(0)
        self.C = np.eye(0)
        self.d = np.eye(0)

        # intialize LP variables
        self.nx = 0
        self.get_indices = dict()

    def add_variable(self, n, label):

        # increase size of A and C
        self.A = np.hstack((
            self.A,
            np.zeros((self.A.shape[0], n))
            ))
        self.C = np.hstack((
            self.C,
            np.zeros((self.C.shape[0], n))
            ))

        # increase size H and f
        H = np.zeros
        self.H = np.hstack((
            self.A,
            np.zeros((self.A.shape[0], n))
            ))



        # update varible dictionary
        self.get_indices[label] = [self.nx, self.nx+n]
        self.nx += n

    def add_inequality(self, A, b, label):

        # update A
        ind = self.get_indices[label]
        A_block = np.hstack((
            np.zeros((A.shape[0], ind[0])),
            A,
            np.zeros((A.shape[0], self.nx - ind[1]))
            ))
        self.A = np.vstack((self.A, A_block))

        # update b
        self.b = np.vstack((self.b, b))




