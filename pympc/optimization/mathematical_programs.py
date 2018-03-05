# external imports
import numpy as np
from copy import copy

# internal inputs
from pympc.optimization.pnnls import linear_program, quadratic_program

class LinearProgram():
    """
    Defines a linear program in the form min_{x in X} f' x.
    """

    def __init__(self, X, f=None):
        """
        Instantiates the linear program.

        Arguments
        ----------
        X : instance of Polyhderon
            Constraint set of the LP.
        f : numpy.ndarray
            Gradient of the cost function.
        """

        # make the cost vector a 2d matrix
        if f is not None and len(f.shape) == 1:
            f = np.reshape(f, (f.shape[0], 1))

        # store inputs
        self.X = X
        self.f = f

    def solve(self):
        """
        Returns the solution of the linear program.

        Returns
        ----------
        sol : dict
            Dictionary with the solution of the LP (see the documentation of pympc.optimization.pnnls.linear_program for the details of the fields of sol).
        """

        # check that a cost function has been set
        if self.f is None:
            raise ValueError('set a cost before solving the linear program.')

        # solve the LP
        sol = linear_program(
            self.f,
            self.X.A,
            self.X.b,
            self.X.C,
            self.X.d
            )

        return sol

    def solve_min_norm_one(self, W=None):
        """
        Sets the cost function to be the weighted norm one ||W x||_1 and solves the LP.
        Adds a number of slack variables s equal to the size of x
        The new optimization vector is [x' s']'.

        Arguments
        ----------
        W : numpy.ndarray
            Weight matrix of the cost function (if None it is set to the identity matrix).

        Returns
        ----------
        sol : dict
            Dictionary with the solution of the LP (see the documentation of pympc.optimization.pnnls.linear_program for the details of the fields of sol).
            Slack variables are removed from the solution.
        """

        # problem size
        n_ineq, n_x = self.X.A.shape
        n_eq = self.X.C.shape[0]

        # set W to identity if W is not passed
        if W is None:
            W = np.eye(n_x)

        # new inequalities
        A = np.vstack((
            np.hstack((self.X.A, np.zeros((n_ineq, n_x)))),
            np.hstack((W, -np.eye(n_x))),
            np.hstack((-W, -np.eye(n_x)))
            ))
        b = np.vstack((
            self.X.b,
            np.zeros((n_x, 1)),
            np.zeros((n_x, 1))
            ))

        # new equalities
        C = np.hstack((
            self.X.C,
            np.zeros((n_eq, n_x))
            ))
        d = self.X.d

        # new f
        f = np.vstack((
            np.zeros((n_x, 1)),
            np.ones((n_x, 1))
            ))

        # solve linear program and remove slacks
        sol = linear_program(f, A, b, C, d)
        sol = self._remove_slacks(sol)

        return sol

    def solve_min_norm_inf(self, W=None):
        """
        Sets the cost function to be the weighted infinity norm ||W x||_inf and solves the LP. Adds one slack variable. The new optimization vector is [x' s]'.

        Arguments
        ----------
        W : numpy.ndarray
            Weight matrix of the cost function (if None it is set to the identity matrix).

        Returns
        ----------
        sol : dict
            Dictionary with the solution of the LP (see the documentation of pympc.optimization.pnnls.linear_program for the details of the fields of sol).
            Slack variables are removed from the solution.
        """

        # problem size
        n_ineq, n_x = self.X.A.shape
        n_eq = self.X.C.shape[0]

        # set W to identity if W is not passed
        if W is None:
            W = np.eye(n_x)

        # new inequalities
        A = np.vstack((
            np.hstack((self.X.A, np.zeros((n_ineq, 1)))),
            np.hstack((W, -np.ones((n_x, 1)))),
            np.hstack((-W, -np.ones((n_x, 1))))
            ))
        b = np.vstack((
            self.X.b,
            np.zeros((n_x, 1)),
            np.zeros((n_x, 1))
            ))

        # new equalities
        C = np.hstack((self.X.C, np.zeros((n_eq, 1))))
        d = self.X.d

        # new f
        f = np.vstack((
            np.zeros((n_x, 1)),
            np.ones((1, 1))
            ))

        # solve linear program and remove slacks
        sol = linear_program(f, A, b, C, d)
        sol = self._remove_slacks(sol)

        return sol

    def _remove_slacks(self, sol):
        """
        Removes the slack variables from the solution of the linear program.

        Arguments
        ----------
        sol : dict
            Dictionary with the solution of the LP with slack variables.

        Returns
        ----------
        sol : dict
            Dictionary with the solution of the LP without slack variables.
        """

        # problem size
        n_ineq, n_x = self.X.A.shape
        n_eq = self.X.C.shape[0]

        # remove slack variables
        if sol['min'] is not None:
            sol['argmin'] = sol['argmin'][:n_x, :]
            sol['active_set'] = [i for i in sol['active_set'] if i < n_ineq]
            sol['multiplier_inequality'] = sol['multiplier_inequality'][:n_ineq, :]
            if n_eq > 0:
                sol['multiplier_equality'] = sol['multiplier_equality'][:n_eq, :]

        return sol

class QuadraticProgram():
    """
    Defines a quadratic program in the form min_{x in X} .5 x' H x + f' x.
    """

    def __init__(self, X, H, f=None):
        """
        Instantiates the quadratic program.

        Arguments
        ----------
        X : instance of Polyhderon
            Constraint set of the QP.
        H : numpy.ndarray
            Hessian of the cost function.
        f : numpy.ndarray
            Gradient of the cost function.
        """

        # make the cost vector a 2d matrix
        if f is None:
            f = np.zeros((H.shape[0], 1))
        elif len(f.shape) == 1:
            f = np.reshape(f, (f.shape[0], 1))

        # store inputs
        self.X = X
        self.H = H
        self.f = f

    def solve(self):
        """
        Returns the solution of the quadratic program.

        Returns
        ----------
        sol : dict
            Dictionary with the solution of the QP (see the documentation of pympc.optimization.pnnls.quadratic_program for the details of the fields of sol).
        """

        # solve the LP
        sol = quadratic_program(
            self.H,
            self.f,
            self.X.A,
            self.X.b,
            self.X.C,
            self.X.d
            )

        return sol