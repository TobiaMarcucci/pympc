# external imports
import numpy as np

# internal inputs
from pympc.optimization.pnnls import linear_program as lp_pnnls

class LinearProgram():

    def __init__(self, cost, constraint):
        """
        Defines a linear program in the form min_{x \in constraint} cost' x, where constraint is an instance of the Polyhedron class.
        """

        # make the cost vector a 2d matrix
        if len(cost.shape) == 1:
            cost = np.reshape(cost, (cost.shape[0], 1))

        # store inputs
        self.cost = cost
        self.constraint = constraint

    def solve(self, solver='pnnls'):
        """
        Solves the linear program using the specified solver.
        """

        # solve with the home-mad partially-non-negative-least-squares solver
        if solver == 'pnnls':
            sol = lp_pnnls(
                self.cost,
                self.constraint.A,
                self.constraint.b,
                self.constraint.C,
                self.constraint.d
                )

        # solve with gurobi
        elif solver == 'gurobi':
            sol = lp_gurobi(
                self.cost,
                self.constraint.A,
                self.constraint.b,
                self.constraint.C,
                self.constraint.d
                )

        return sol





