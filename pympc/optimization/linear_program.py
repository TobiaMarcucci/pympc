import numpy as numpy

class LinearProgram():

	def __init__(self, f, D):
		"""
		Defines a linear program in the form min_{x \in D} f' x, where D is a Polyhedron intance.
		"""

		# make f a 2d matrix
        if len(f.shape) == 1:
            b = np.reshape(f, (f.shape[0], 1))

        # store inputs
        self.f = f
        self.D = D

    def solve(self):



