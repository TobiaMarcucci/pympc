
# external imports
import numpy as np
from copy import copy

# internal inputs
from pympc.optimization.pnnls import linear_program as lp_pnnls

class LinearProgram():

    def __init__(self, constraint, cost=None):
        """
        Defines a linear program in the form min_{x \in constraint} cost' x, where constraint is an instance of the Polyhedron class.
        """

        # make the cost vector a 2d matrix
        if cost is not None and len(cost.shape) == 1:
            cost = np.reshape(cost, (cost.shape[0], 1))

        # store inputs
        self.constraint = constraint
        self.cost = cost

        # keep track of slack variables
        self._constraint_with_slack = copy(constraint)
        self._cost_with_slack = copy(cost)


    def set_cost(self, cost):

        # restore the LP removing all the slacks etc.
        if self._cost_with_slack is not None:
            self._restore()

        # store inputs
        self.cost = cost

        # keep track of slack variables
        self._cost_with_slack = copy(cost)


    def set_norm_one_cost(self, W=None):
        """
        Sets the cost function to be the weighted norm one ||W x||_1. Adds a number of slack variables s equal to the size of x. The new optimization vector is [x' s']'.
        """

        ## restore the LP removing all the slacks etc.
        if self._cost_with_slack is not None:
            self._restore()

        # set W to identity if W is not passed
        n_ineq, n_x = self.constraint.A.shape
        n_eq, _ = self.constraint.C.shape
        if W is None:
            W = np.eye(n_x)

        # new inequalities
        self._constraint_with_slack.A = np.vstack((
            np.hstack((self.constraint.A, np.zeros((n_ineq, n_x)))),
            np.hstack((W, -np.eye(n_x))),
            np.hstack((-W, -np.eye(n_x)))
            ))
        self._constraint_with_slack.b = np.vstack((
            self.constraint.b,
            np.zeros((n_x, 1)),
            np.zeros((n_x, 1))
            ))

        # new equalities
        self._constraint_with_slack.C = np.hstack((self.constraint.C, np.zeros((n_eq, n_x))))
        self._constraint_with_slack.d = self.constraint.d

        # new cost
        self._cost_with_slack = np.vstack((
            np.zeros((n_x, 1)),
            np.ones((n_x, 1))
            ))

    def set_norm_inf_cost(self, W=None):
        """
        Sets the cost function to be the weighted infinity norm ||W x||_inf. Adds one slack variable. The new optimization vector is [x' s]'.
        """

        # restore the LP removing all the slacks etc.
        if self._cost_with_slack is not None:
            self._restore()

        # set W to identity if W is not passed
        n_ineq, n_x = self.constraint.A.shape
        n_eq = self.constraint.C.shape[0]
        if W is None:
            W = np.eye(n_x)

        # new inequalities
        self._constraint_with_slack.A = np.vstack((
            np.hstack((self.constraint.A, np.zeros((n_ineq, 1)))),
            np.hstack((W, -np.ones((n_x, 1)))),
            np.hstack((-W, -np.ones((n_x, 1))))
            ))
        self._constraint_with_slack.b = np.vstack((
            self.constraint.b,
            np.zeros((n_x, 1)),
            np.zeros((n_x, 1))
            ))

        # new equalities
        self._constraint_with_slack.C = np.hstack((self.constraint.C, np.zeros((n_eq, 1))))
        self._constraint_with_slack.d = self.constraint.d

        # new cost
        self._cost_with_slack = np.vstack((
            np.zeros((n_x, 1)),
            np.ones((1, 1))
            ))

    def _restore(self):

        # keep track of slack variables
        self.cost = None
        self._cost_with_slack = None
        self._constraint_with_slack = copy(self.constraint)

    def solve(self, solver='pnnls'):
        """
        Solves the linear program using the specified solver.
        """

        # solve with the home-mad partially-non-negative-least-squares solver
        if solver == 'pnnls':
            sol = lp_pnnls(
                self._cost_with_slack,
                self._constraint_with_slack.A,
                self._constraint_with_slack.b,
                self._constraint_with_slack.C,
                self._constraint_with_slack.d
                )

        # solve with gurobi
        elif solver == 'gurobi':
            sol = lp_gurobi(
                self._cost_with_slack,
                self._constraint_with_slack.A,
                self._constraint_with_slack.b,
                self._constraint_with_slack.C,
                self._constraint_with_slack.d
                )

        # remove slack variables
        n_ineq, n_x = self.constraint.A.shape
        n_eq = self.constraint.C.shape[0]
        sol.argmin = sol.argmin[:n_x, :]
        sol.inequality_multipliers = sol.inequality_multipliers[:n_ineq, :]
        sol.equality_multipliers = sol.equality_multipliers[:n_eq, :]
        sol.active_set = sol.active_set[:n_ineq]

        return sol