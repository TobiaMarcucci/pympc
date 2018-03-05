# external imports
import numpy as np

# pympc imports
from pympc.optimization.linear_program import LinearProgram
from pympc.algebra import nullspace_basis

class Polyhedron:
    """
    Polyhedron in the form {x in R^n | A x <= b, C x = d}.
    """

    def __init__(self, A, b, C=None, d=None):
        """
        Instantiates the polyhedron.

        Arguments
        ----------
        A : numpy.ndarray
            Left-hand side of the inequalities.
        b : numpy.ndarray
            Right-hand side of the inequalities.
        C : numpy.ndarray
            Left-hand side of the equalities.
        d : numpy.ndarray
            Right-hand side of the equalities.
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

        Arguments
        ----------
        A : numpy.ndarray
            Left-hand side of the inequalities.
        b : numpy.ndarray
            Right-hand side of the inequalities.
        """

        # check inequalities
        A, b = self._check_input_matices(A, b)

        # add inequalities
        self.A = np.vstack((self.A, A))
        self.b = np.vstack((self.b, b))

    def add_equality(self, C, d):
        """
        Adds the equality C x = d to the existing polyhedron.

        Arguments
        ----------
        C : numpy.ndarray
            Left-hand side of the equalities.
        d : numpy.ndarray
            Right-hand side of the equalities.
        """

        # check equalities
        C, d = self._check_input_matices(C, d)

        # add equalities
        self.C = np.vstack((self.C, C))
        self.d = np.vstack((self.d, d))

    def add_lower_bound(self, x_min, indices=None):
        """
        Adds the inequality x[indices] >= x_min to the existing polyhedron.
        If indices is None, the inequality is applied to all the elements of x.

        Arguments
        ----------
        x_min : numpy.ndarray
            Lower bound on the elements of x.
        indices : list of int
            Set of indices of elements of x to which the inequality applies.
        """

        # if x_min is a float make it a 2d array
        if isinstance(x_min, float):
            x_min = np.array([[x_min]])

        # add the constraint - S x <= - x_min, with S selection matrix
        S = self._selection_matrix(indices)
        self.add_inequality(-S, -x_min)

    def add_upper_bound(self, x_max, indices=None):
        """
        Adds the inequality x[indices] <= x_max to the existing polyhedron.
        If indices is None, the inequality is applied to all the elements of x.

        Arguments
        ----------
        x_max : numpy.ndarray
            Upper bound on the elements of x.
        indices : list of int
            Set of indices of elements of x to which the inequality applies.
        """

        # if x_max is a float make it a 2d array
        if isinstance(x_max, float):
            x_max = np.array([[x_max]])

        # add the constraint S x <= x_max, with S selection matrix
        S = self._selection_matrix(indices)
        self.add_inequality(S, x_max)

    def add_bounds(self, x_min, x_max, indices=None):
        """
        Adds the inequalities x_min <= x[indices] <= x_max to the existing polyhedron.
        If indices is None, the inequality is applied to all the elements of x.

        Arguments
        ----------
        x_min : numpy.ndarray
            Lower bound on the elements of x.
        x_max : numpy.ndarray
            Upper bound on the elements of x.
        indices : list of int
            Set of indices of elements of x to which the inequality applies.
        """

        self.add_lower_bound(x_min, indices)
        self.add_upper_bound(x_max, indices)

    def _selection_matrix(self, indices):
        """
        Returns a selection matrix S such that S x = x[indices].

        Arguments
        ----------
        indices : list of int
            Set of indices of elements of x that have to be selected by S.

        Returns
        ----------
        S : numpy.ndarray
            Selection matrix.
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
        Instantiate a Polyhedron in the form {x | x[indices] >= x_min}.
        If indices is None, the inequality is applied to all the elements of x.
        If indices is not None, n must be provided to determine the length of x.

        Arguments
        ----------
        x_min : numpy.ndarray
            Lower bound on the elements of x.
        indices : list of int
            Set of indices of elements of x to which the inequality applies.
        n : int
            Dimension of the vector x in R^n.
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
        Instantiate a Polyhedron in the form {x | x[indices] <= x_max}.
        If indices is None, the inequality is applied to all the elements of x.
        If indices is not None, n must be provided to determine the length of x.

        Arguments
        ----------
        x_max : numpy.ndarray
            Upper bound on the elements of x.
        indices : list of int
            Set of indices of elements of x to which the inequality applies.
        n : int
            Dimension of the vector x in R^n.
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
        Instantiate a Polyhedron in the form {x | x_min <= x[indices] <= x_max}.
        If indices is None, the inequality is applied to all the elements of x.
        If indices is not None, n must be provided to determine the length of x.

        Arguments
        ----------
        x_min : numpy.ndarray
            Lower bound on the elements of x.
        x_max : numpy.ndarray
            Upper bound on the elements of x.
        indices : list of int
            Set of indices of elements of x to which the inequality applies.
        n : int
            Dimension of the vector x in R^n.
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
    def _check_input_matices(E, f):
        """
        Reshapes the right hand side f in a 2d vector and checks that E and f have the same number of rows.

        Arguments
        ----------
        E : numpy.ndarray
            Left-hand side of the (in)equalities.
        f : numpy.ndarray
            Right-hand side of the (in)equalities.

        Returns
        ----------
        E : numpy.ndarray
            Left-hand side of the (in)equalities.
        f : numpy.ndarray
            Right-hand side of the (in)equalities.
        """

        # make f a 2d matrix
        if len(f.shape) == 1:
            f = np.reshape(f, (f.shape[0], 1))

        # check nomber of rows
        if E.shape[0] != f.shape[0]:
            raise ValueError("incoherent size of the inputs.")

        return E, f

    def normalize(self, tol=1.e-7):
        """
        Normalizes the polyhedron dividing each row of A by its norm and each entry of b by the norm of the corresponding row of A.

        Arguments
        ----------
        tol : float
            Threshold value for the norm of the rows of A and C under which the related inequality (or equality) is not normalized.
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
        Computes the indices of the facets that generate a minimal representation of the polyhedron solving an LP for each facet of the redundant representation.
        (See "Fukuda - Frequently asked questions in polyhedral computation" Sec.2.21.)
        In case of equalities, first the problem is projected in the nullspace of the equalities.

        Arguments
        ----------
        tol : float
            Minimum distance of a redundant facet from the interior of the polyhedron to be considered as such.

        Returns
        ----------
        minimal_facets : list of int
            List of indices of the non-redundant inequalities A x <= b.
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
            cost_i = - sol['min']

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
        For the polyhedron P := {x | A x <= b, C x = d}, returns the change of variables x = [N R] [n' r']' such that P can be expressed only by inequalities, i.e. := {n | E n <= f}.

        Math
        ----------
        We consider the change of variables x = [N R] [n' r']', with N = null(C) and R = null(Z').
        Substituting x in the equalities C x = C R r = d, we get r = (C R)^-1 d.
        Then, substituting x and r in the inequalities, we get P := {n | E n <= f}, where E := A N and f := b - A R (C R)^-1 d.

        Returns
        ----------
        E : numpy.ndarray
            Left-hand side of the inequalities describing the polyhedron in the new set of variables.
        f : numpy.ndarray
            Right-hand side of the inequalities describing the polyhedron in the new set of variables.
        N : numpy.ndarray
            Basis of the nullspace of C, and first block of columns for the change of coordinates.
        R : numpy.ndarray
            Orthogonal complement of N to form an orthogonal change of coordinates.
        """

        # change of variables
        N = nullspace_basis(self.C)
        R = nullspace_basis(N.T)

        # new representation
        E = self.A.dot(N)
        r = np.linalg.inv(self.C.dot(R)).dot(self.d)
        f = self.b - self.A.dot(R.dot(r))

        return E, f, N, R

    def is_empty(self):
        """
        Checks if the polyhedron P is empty solving an LP for the x with minimum infinity norm contained in P.

        Returns
        ----------
        empty : bool
            True if the polyhedron is empty, False otherwise.
        """

        # if a sultion is found, return False
        lp = LinearProgram(self)
        sol = lp.solve_min_norm_inf()
        empty = sol['min'] is None

        return empty

    def is_bounded(self):
        """
        Checks if the polyhedron is bounded (returns True or False).

        Math
        ----------
        Consider the non-empty polyhedron P := {x | A x <= b}.
        We have that necessary and sufficient condition for P to be unbounded is the existence of a nonzero x | A x <= 0.
        (Proof: Given x_1 that verifies the latter condition and x_2 in P, consider x_3 := a x_1 + x_2, with a in R. We have x_3 in P for all a >= 0, in fact A x_3 = a A x_1 + A x_2 <= b. Considering a -> inf the unboundedness of P follows.)
        It follows that sufficient condition for P to be unbounded is that ker(A) is not empty; hence in the following we consider only the case ker(A) = 0.
        Stiemke's Theorem of alternatives (see, e.g., Mangasarian, Nonlinear Programming, pag. 32) states that either there exists an x | A x <= 0, A x != 0, or there exists a y > 0 | A' y = 0.
        Note that: i) being ker(A) = 0, the condition A x != 0 is equivalent to x != 0; ii) in this case, y > 0 is equilvaent e.g. to y >= 1.
        In conclusion we have that: under the assumptions non-empty P and ker(A) = 0, necessary and sufficient conditions for the boundedness of P is the existence of y >= 1 | A' y = 0.
        Here we search for the y with minimum norm 1 that satisfies the latter condition (note that y >= 1 implies ||y||_1 = 1' y).

        Returns
        ----------
        bounded : bool
            True if the polyhedron is bounded, False otherwise.
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
        bounded = sol['min'] is not None

        return bounded

    def contains(self, x, tol=1.e-7):
        """
        Determines if the given point belongs to the polytope.

        Arguments
        ----------
        x : numpy.ndarray
            Point whose belonging to the polyhedron must be verified.
        tol : float
            Maximum distance of a point from the polyhedron to be considered an internal point.

        Returns
        ----------
        contains_x : bool
            True if the point x is inside the polyhedron, False otherwise.
        """

        # check inequalities
        in_ineq = np.max(self.A.dot(x) - self.b) <= tol

        # check equalities
        in_eq = True
        if self.C.shape[0] > 0:
            in_eq = np.abs(np.max(self.C.dot(x) - self.d)) <= tol
        contains_x = in_ineq and in_eq

        return contains_x

    def is_included_in(self, P2, tol=1.e-7):
        """
        Checks if the polyhedron P is a subset of the polyhedron P2 (returns True or False).
        For each halfspace H descibed a facet of P2, it solves an LP to check if the intersection of H with P1 is euqual to P1.
        If this is the case for all H, then P1 is in P2.

        Arguments
        ----------
        P2 : instance of Polyhderon
            Polyhderon within which we want to check if this polyhedron is contained.
        tol : float
            Maximum distance of a point from P2 to be considered an internal point.

        Returns
        ----------
        included : bool
            True if this polyhedron is contained in P2, False otherwise.
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
            lp.f = -A2[i:i+1,:].T
            sol = lp.solve()
            penetration = - sol['min'] - b2[i]
            if penetration > tol:
                included = False
                break

        return included

    def chebyshev(self):
        """
        Returns the Chebyshev center and radius of the polyhedron.
        This should return inf in case of infinite radius (to check the unboundedness of the primal it should verify unfeasibility of the dual.)
        """

        if self.C.shape[0] > 0:
            A, B, N, R = self._remove_equalities()
        else:
            A = self.A
            b = self.b

        pass

    #@property # needed to call the method without () ?
    def vertices(self):
        # need to check what qhull says in case of unbounded polyhedron...
        pass

    def project_to(self, dimensions):
        pass

    @staticmethod
    def from_vertices(self, v_list):
        """
        This would assume the polyhedron to be bounded...
        """
        pass

    def plot(self):
        # if more than 2d raise value error
        pass