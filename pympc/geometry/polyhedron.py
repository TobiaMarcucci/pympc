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
            lp = LinearProgram(constraint, -E[i,:].T)
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
    	We consider the change of variables x = [Y Z] [y' z']', with Z = null(C) and Y = null(Z'). Substituting x in the equalities C x = C Y y = d, we get y = (C Y)^-1 d. Then, substituting x and y in the inequalities, we get P := {z | E z <= f}, where E := A Z and f := b - A Y (C Y)^-1 d.
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
    	"""
    	Checks if the polyhedron P is empty solving an LP for the x with minimum infinity norm contained in P.
    	"""

    	# if a sultion is found, return False
    	lp = LinearProgram(self)
    	lp.set_norm_inf_cost()
    	sol = lp.solve()
    	empty = np.isnan(sol.min)

    	return empty

    def is_bounded(self):
    	"""
    	Checks if the polyhedron is bounded (returns True or False).

    	Math:
    	Consider the non-empty polyhedron P := {x | A x <= b}.
    	We have that necessary and sufficient condition for P to be unbounded is the existence of a nonzero x | A x <= 0. (Proof: given x_1 that verifies the latter condition and x_2 in P, consider x_3 := a x_1 + x_2, with a in R. We have x_3 in P for all a >= 0, in fact A x_3 = a A x_1 + A x_2 <= b. Considering a -> inf the unboundedness of P follows.) It follows that sufficient condition for P to be unbounded is that ker(A) is not empty; hence in the following we consider only the case ker(A) = 0. Stiemke's Theorem of alternatives (see, e.g., Mangasarian, Nonlinear Programming, pag. 32) states that either there exists an x | A x <= 0, A x != 0, or there exists a y > 0 | A' y = 0. Note that: i) being ker(A) = 0, the condition A x != 0 is equivalent to x != 0; ii) in this case, y > 0 is equilvaent e.g. to y >= 1. In conclusion we have that: under the assumptions non-empty P and ker(A) = 0, necessary and sufficient conditions for the boundedness of P is the existence of y >= 1 | A' y = 0. Here we search for the y with minimum norm 1 that satisfies the latter condition (note that y >= 1 implies ||y||_1 = 1' y).
    	"""

    	# check emptyness
    	if self.is_empty():
    		return True

    	# include equalities
    	A = np.vstack((self.A, self.C, -self.C))

    	# check kernel of A
    	if nullspace_basis(A).shape[1] > 0:
    		return False

    	# check Stiemke's theorem of alternatives
    	n, m = A.shape
    	constraint = Polyhedron.from_lower_bound(np.ones((n, 1)))
    	constraint.add_equality(self.A.T, np.zeros((m, 1)))
    	cost = np.ones((n, 1))
    	lp = LinearProgram(constraint, cost)
    	sol = lp.solve()
    	bounded = not(np.isnan(sol.min))

    	return bounded

    def contains(self, x, tol=1.e-7):
        """
        Determines if the given point belongs to the polytope (returns True or False).
        """

        # check inequalities
        in_ineq = np.max(self.A.dot(x) - self.b) <= tol

        # check equalities
        in_eq = True
        if self.C.shape[0] > 0:
            in_eq = np.abs(np.max(self.C.dot(x) - self.d)) <= tol

        return in_ineq and in_eq

    def is_included_in(self, P2, tol=1.e-7):
        """
        Checks if the polyhedron P is a subset of the polyhedron P2 (returns True or False). For each halfspace H descibed a facet of P2, it solves an LP to check if the intersection of H with P1 is euqual to P1; if this is the case for all H, then P1 is in P2. 
        """

        # augment inequalities with equalities
        A1 = np.vstack((self.A, self.C, -self.C))
        b1 = np.vstack((self.b, self.d, -self.d))
        P1 = Polyhedron(A1, b1)
        A2 = np.vstack((P2.A, P2.C, -P2.C))
        b2 = np.vstack((P2.b, P2.d, -P2.d))
        
        # check inclusion, one facet per time
        included = True
        lp = LinearProgram(P1)
        for i in range(A2.shape[0]):
            lp.set_cost(-A2[i:i+1,:].T)
            sol = lp.solve()
            penetration = - sol.min - b2[i]
            if penetration > tol:
                included = False
                break

        return included

    @property # needed to call the method without () ?
    def vertices(self):
    	pass

    def project_to(self, dimensions):
    	pass

    def plot(self):
    	# if more than 2d raise value error
    	pass