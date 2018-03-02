# external imports
import numpy as np

# pympc imports
from pympc.optimization.linear_program import LinearProgram
from pympc.algebra import nullspace_basis

class Polyhedron:

    def __init__(self, A, b, C=None, d=None):
    	"""
        Defines a polyhedron in the form {x \in R^n | A x <= b, C x = d}.
        """

        # check and store inequalities
        self.A, self.b = self._check_input_matices(A, b)

        # check and store equalities
        if (C is None) != (d is None):
            raise ValueError("missing C or d.")
        if C is None:
            self.C = np.zeros((0, A.shape[1]))
            self.d = np.zeros((0, 1))
        else:
            self.C, self.d = self._check_input_matices(C, d)

    def add_inequality(self, A, b):
    	"""
    	Adds the inequality A x <= b to the existing polyhedron.
    	"""

        # check inequalities
        A, b = self._check_input_matices(A, b)

        # add inequalities
        self.A = np.vstack((self.A, A))
        self.b = np.vstack((self.b, b))

    def add_equality(self, C, d):
    	"""
    	Adds the equality C x <= d to the existing polyhedron.
    	"""

        # check equalities
        C, d = self._check_input_matices(C, d)

        # add equalities
        self.C = np.vstack((self.C, C))
        self.d = np.vstack((self.d, d))

    def add_lower_bound(self, x_min, indices=None):
    	"""
    	Adds the inequality x[indices] >= x_min to the existing polyhedron. If indices is None, the inequality is applied to all the elements of x.
    	"""

    	# if x_min is a float make it a 2d array
        if isinstance(x_min, float):
            x_min = np.array([[x_min]])

        # add the constraint - S x <= - x_min, with S selection matrix
        S = self._selection_matrix(indices)
        self.add_inequality(-S, -x_min)

    def add_upper_bound(self, x_max, indices=None):
    	"""
    	Adds the inequality x[indices] <= x_max to the existing polyhedron. If indices is None, the inequality is applied to all the elements of x.
    	"""

    	# if x_max is a float make it a 2d array
        if isinstance(x_max, float):
            x_max = np.array([[x_max]])

        # add the constraint S x <= x_max, with S selection matrix
        S = self._selection_matrix(indices)
        self.add_inequality(S, x_max)

    def add_bounds(self, x_min, x_max, indices=None):
    	"""
    	Adds the inequalities x_min <= x[indices] <= x_max to the existing polyhedron. If indices is None, the inequality is applied to all the elements of x.
    	"""

        self.add_lower_bound(x_min, indices)
        self.add_upper_bound(x_max, indices)

    def _selection_matrix(self, indices):
    	"""
    	Returns a selection matrix S such that S x = x[indices].
    	"""
        
        # if indices is None select all the rows
        n = self.A.shape[1]
        if indices is None:
            indices = range(n)

        # delete from the identity matrix all the rows that are not in indices
        complement = [i for i in range(n) if i not in indices]
        S = np.delete(np.eye(n), complement, 0)

        return S

    @staticmethod
    def from_lower_bound(x_min, indices=None, n=None):
    	"""
    	Instantiate a Polyhedron in the form {x | x[indices] >= x_min}. If indices is None, the inequality is applied to all the elements of x. If indices is not None, n must be provided to determine the length of x.
    	"""

    	# check if n is provided
    	if indices is not None and n is None:
            raise ValueError("specify the length of x to instantiate the polyhedron.")

        # construct the polyhderon
        if n is None:
        	n = x_min.shape[0]
        A = np.zeros((0, n))
        b = np.zeros((0, 1))
        p = Polyhedron(A, b)
        p.add_lower_bound(x_min, indices)

        return p

    @staticmethod
    def from_upper_bound(x_max, indices=None, n=None):
    	"""
    	Instantiate a Polyhedron in the form {x | x[indices] <= x_max}. If indices is None, the inequality is applied to all the elements of x. If indices is not None, n must be provided to determine the length of x.
    	"""

    	# check if n is provided
    	if indices is not None and n is None:
            raise ValueError("specify the length of x to instantiate the polyhedron.")

        # construct the polyhderon
        if n is None:
        	n = x_max.shape[0]
        A = np.zeros((0, n))
        b = np.zeros((0, 1))
        p = Polyhedron(A, b)
        p.add_upper_bound(x_max, indices)

        return p

    @staticmethod
    def from_bounds(x_min, x_max, indices=None, n=None):
    	"""
    	Instantiate a Polyhedron in the form {x | x_min <= x[indices] <= x_max}. If indices is None, the inequality is applied to all the elements of x. If indices is not None, n must be provided to determine the length of x.
    	"""

    	# check if n is provided
    	if indices is not None and n is None:
            raise ValueError("specify the length of x to instantiate the polyhedron.")

        # check size of the bounds
        if x_min.shape[0] != x_max.shape[0]:
        	raise ValueError("bounds must have the same size.")

        # construct the polyhderon
        if n is None:
        	n = x_min.shape[0]
        A = np.zeros((0, n))
        b = np.zeros((0, 1))
        p = Polyhedron(A, b)
        p.add_bounds(x_min, x_max, indices)

        return p

    @staticmethod
    def _check_input_matices(A, b):
    	"""
    	Reshapes the right hand side b in a 2d vector and checks that A and b have the same number of rows.
    	"""

        # make b a 2d matrix
        if len(b.shape) == 1:
            b = np.reshape(b, (b.shape[0], 1))

        # check nomber of rows
        if A.shape[0] != b.shape[0]:
            raise ValueError("incoherent size of the inputs.")

        return A, b

    def normalize(self, tol=1e-7):
        """
        Normalizes the polyhedron dividing each row of A by its norm and each entry of b by the norm of the corresponding row of A.
        """

        # inequalities
        for i in range(self.A.shape[0]):
            r = np.linalg.norm(self.A[i,:])
            if r > tol:
                self.A[i,:] = self.A[i,:]/r
                self.b[i,:] = self.b[i,:]/r

        # equalities
        for i in range(self.C.shape[0]):
            r = np.linalg.norm(self.C[i,:])
            if r > tol:
                self.C[i,:] = self.C[i,:]/r
                self.d[i,:] = self.d[i,:]/r

    def get_minimal_facets(self, tol=1.e-7):
    	"""
        Computrs the indices of the facets that generate a minimal representation of the polyhedron solving an LP for each facet of the redundant representation. (See "Fukuda - Frequently asked questions in polyhedral computation" Sec.2.21.) In case of equalities, first the problem is projected in the nullspace of the equalities.
        """

        # if there are equalities, project
        if self.C.shape[0] != 0:
        	E, f, _, _ = self._remove_equalities()
        else:
        	E = self.A
        	f = self.b

        # initialize list of non-redundant facets
        minimal_facets = range(E.shape[0])

        # check each facet
        for i in range(E.shape[0]):

        	# remove redundant facets and relax ith inequality
            E_minimal = E[minimal_facets,:]
            f_relaxation = np.zeros(np.shape(f))
            f_relaxation[i] += 1.
            f_relaxed = (f + f_relaxation)[minimal_facets];

            # solve linear program
            constraint = Polyhedron(E_minimal, f_relaxed)
            lp = LinearProgram(-E[i,:].T, constraint)
            sol = lp.solve()
            cost_i = - sol.min

            # remove redundant facets from the list
            if cost_i - f[i] < tol or np.isnan(cost_i):
                minimal_facets.remove(i)

        return minimal_facets

    def remove_redundant_inequalities(self):
    	"""
        Removes the redundant facets of the polyhedron, it modifies the attributes A and b.
        """

        minimal_facets = self.get_minimal_facets()
        self.A = self.A[minimal_facets,:]
        self.b = self.b[minimal_facets]

    def _remove_equalities(self):
    	"""
    	Given the polyhedron in the form P := {x | A x <= b, C x = d}, returns the change of variables x = [Y Z] [y' z']' such that P can be expressed only with inequalities, i.e. := {z | E z <= f}.

    	Math:
    	- derive the matrices Z = null(C) and Y = null(Z'),
    	- consider the change of variables x = [Y Z] [y' z']',
    	- solve the equalities for y, i.e. C x = C Y y = d => y = (C Y)^-1 d,
    	- change variables to get P := {z | E z <= f}, with E := A Z and f := b - A Y (C Y)^-1 d.
    	"""

    	# change of variables
    	Z = nullspace_basis(self.C)
        Y = nullspace_basis(Z.T)

        # new representation
        E = self.A.dot(Z)
        y = np.linalg.inv(self.C.dot(Y)).dot(self.d)
        f = self.b - self.A.dot(Y.dot(y))

        return E, f, Y, Z

    def is_empty(self):
    	pass

    def is_bounded(self):
    	pass

    def is_included_in(self, p):
    	pass

    def contains(self, x):
    	pass

    @property # needed to call the method without () ?
    def vertices(self):
    	pass

    def project_to(self, dimensions):
    	pass

    def intersect_with(self, p):
    	pass

    def plot(self):
    	# if more than 2d raise value error
    	pass