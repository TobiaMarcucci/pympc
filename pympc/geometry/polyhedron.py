# external imports
from six.moves import range  # behaves like xrange on python2, range on python3
import numpy as np
from copy import copy
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

# pympc imports
from pympc.optimization.programs import linear_program, quadratic_program
from pympc.geometry.utils import nullspace_basis, plane_through_points

class Polyhedron(object):
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
        if len(b.shape) > 1:
            raise ValueError('b must be a one dimensional array.')
        self._same_number_rows(A, b)
        self.A = A
        self.b = b

        # check and store equalities
        if (C is None) != (d is None):
            raise ValueError('missing C or d.')
        if C is None:
            self.C = np.zeros((0, A.shape[1]))
            self.d = np.zeros(0)
        else:
            if len(d.shape) > 1:
                raise ValueError('b must be a one dimensional array.')
            self._same_number_rows(C, d)
            self.C = C
            self.d = d

        # initializes the attributes to None
        self._empty = None
        self._bounded = None
        self._radius = None
        self._center = None
        self._vertices = None

    def add_inequality(self, A, b, indices=None):
        """
        Adds the inequality A x[indices] <= b to the existing polyhedron.

        Arguments
        ----------
        A : numpy.ndarray
            Left-hand side of the inequalities.
        b : numpy.ndarray
            Right-hand side of the inequalities.
        indices : list of int
            Set of indices of elements of x to which the inequality applies.
        """

        # check inequalities
        self._same_number_rows(A, b)
    
        # reset attributes to None
        self._delete_attributes()

        # add inequalities
        S = self._selection_matrix(indices)
        self.A = np.vstack((self.A, A.dot(S)))
        self.b = np.concatenate((self.b, b))

    def add_symbolic_inequality(self, x, ineq):
        """
        Adds the inequality ineq(x) <= 0 to the existing polyhedron.

        Arguments
        ----------
        x : sympy matrix filled with sympy symbols
            Variables.
        ineq : sympy matrix filled with sympy symbolic affine expressions
            Left hand side of the inequality constraint.
        """

        self.add_inequality(*get_matrices_affine_expression(x, ineq))

    def add_equality(self, C, d, indices=None):
        """
        Adds the equality C x = d to the existing polyhedron.

        Arguments
        ----------
        C : numpy.ndarray
            Left-hand side of the equalities.
        d : numpy.ndarray
            Right-hand side of the equalities.
        indices : list of int
            Set of indices of elements of x to which the equality applies.
        """

        # check equalities
        self._same_number_rows(C, d)

        # reset attributes to None
        self._delete_attributes()

        # add equalities
        S = self._selection_matrix(indices)
        self.C = np.vstack((self.C, C.dot(S)))
        self.d = np.concatenate((self.d, d))

    def add_symbolic_equality(self, x, eq):
        """
        Adds the inequality eq(x) = 0 to the existing polyhedron.

        Arguments
        ----------
        x : sympy matrix filled with sympy symbols
            Variables.
        eq : sympy matrix filled with sympy symbolic affine expressions
            Left hand side of the equality constraint.
        """

        self.add_equality(*get_matrices_affine_expression(x, eq))

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

        # if x_min is a float make it an array
        if isinstance(x_min, float):
            x_min = np.array([x_min])

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
            x_max = np.array([x_max])

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

    def _delete_attributes(self):
        """
        Resets al the attibutes of the class to None.
        """

        # reser the attributes to None
        self._empty = None
        self._bounded = None
        self._radius = None
        self._center = None
        self._vertices = None

    def _selection_matrix(self, indices=None):
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
        Instantiates a Polyhedron in the form {x | x[indices] >= x_min}.
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

        # construct the polyhedron
        if n is None:
            n = x_min.size
        A = np.zeros((0, n))
        b = np.zeros(0)
        p = Polyhedron(A, b)
        p.add_lower_bound(x_min, indices)

        return p

    @staticmethod
    def from_upper_bound(x_max, indices=None, n=None):
        """
        Instantiates a Polyhedron in the form {x | x[indices] <= x_max}.
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

        # construct the polyhedron
        if n is None:
            n = x_max.size
        A = np.zeros((0, n))
        b = np.zeros(0)
        p = Polyhedron(A, b)
        p.add_upper_bound(x_max, indices)

        return p

    @staticmethod
    def from_bounds(x_min, x_max, indices=None, n=None):
        """
        Instantiates a Polyhedron in the form {x | x_min <= x[indices] <= x_max}.
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
        if x_min.size != x_max.size:
            raise ValueError("bounds must have the same size.")

        # construct the polyhedron
        if n is None:
            n = x_min.size
        A = np.zeros((0, n))
        b = np.zeros(0)
        p = Polyhedron(A, b)
        p.add_bounds(x_min, x_max, indices)

        return p

    @staticmethod
    def from_symbolic(x, ineq, eq=None):
        """
        Instantiates a Polyhedron in the form expr(x) <= 0, eq(x) = 0.

        Arguments
        ----------
        x : sympy matrix filled with sympy symbols
            Variables.
        ineq : sympy matrix filled with sympy symbolic affine expressions
            Left hand side of the inequality constraint.
        eq : sympy matrix filled with sympy symbolic affine expressions
            Left hand side of the inequality constraint.
        """

        # get polyhedron only for the inequalities
        p = Polyhedron(*get_matrices_affine_expression(x, ineq))

        # in case add equalities
        if eq is not None:
            p.add_equality(*get_matrices_affine_expression(x, eq))

        return p

    @staticmethod
    def _same_number_rows(E, f):
        """
        Checks that E and f have the same number of rows.

        Arguments
        ----------
        E : numpy.ndarray
            Left-hand side of the (in)equalities.
        f : numpy.ndarray
            Right-hand side of the (in)equalities.
        """

        # check nomber of rows
        if E.shape[0] != f.size:
            raise ValueError("incoherent size of the inputs.")

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
            r = np.linalg.norm(self.A[i])
            if r > tol:
                self.A[i] = self.A[i]/r
                self.b[i] = self.b[i]/r

        # equalities
        for i in range(self.C.shape[0]):
            r = np.linalg.norm(self.C[i])
            if r > tol:
                self.C[i] = self.C[i]/r
                self.d[i] = self.d[i]/r

    def minimal_facets(self, tol=1.e-7):
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
            List of indices of the non-redundant inequalities A x <= b (None if the polyhedron in empty).
        """

        # check emptyness
        if self.empty:
            return None

        # if there are equalities, project
        if self.C.shape[0] != 0:
            E, f, _, _ = self._remove_equalities()
        else:
            E = self.A
            f = self.b

        # initialize list of non-redundant facets
        minimal_facets = list(range(E.shape[0]))

        # check each facet
        for i in range(E.shape[0]):

            # remove redundant facets and relax ith inequality
            E_minimal = E[minimal_facets]
            f_relaxation = np.zeros(f.size)
            f_relaxation[i] += 1.
            f_relaxed = (f + f_relaxation)[minimal_facets]

            # solve linear program
            sol = linear_program(-E[i], E_minimal, f_relaxed)

            # remove redundant facets from the list
            if  - sol['min'] - f[i] < tol:
                minimal_facets.remove(i)

        return minimal_facets

    def remove_redundant_inequalities(self):
        """
        Removes the redundant facets of the polyhedron, it modifies the attributes A and b.
        """

        # get minimal facets
        minimal_facets = self.minimal_facets()

        # raise error if empty polyhedron
        if minimal_facets is None:
            raise ValueError('empty polyhedron, cannot remove redundant inequalities.')

        # remove redundancy
        self.A = self.A[minimal_facets]
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
        if N.shape[1] == 0:
            raise ValueError('equality constraints C x = d do not have a nullspace.')
        if N.shape[1] != self.C.shape[1] - self.C.shape[0]:
            raise ValueError('equality constraints C x = d are linearly dependent.')
        R = nullspace_basis(N.T)

        # new representation
        E = self.A.dot(N)
        r = np.linalg.inv(self.C.dot(R)).dot(self.d)
        f = self.b - self.A.dot(R.dot(r))

        return E, f, N, R

    @property
    def empty(self):
        """
        Checks if the polyhedron P is empty solving a QP for the x with minimum norm contained in P.

        Returns
        ----------
        empty : bool
            True if the polyhedron is empty, False otherwise.
        """

        # check if it has been already checked
        if self._empty is not None:
            return self._empty

        # if a sultion is found, return False
        H = np.eye(self.A.shape[1])
        f = np.zeros(self.A.shape[1])
        sol = quadratic_program(H, f, self.A, self.b, self.C, self.d)
        self._empty = sol['min'] is None

        return self._empty

    @property
    def bounded(self):
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
            True if the polyhedron is bounded (if the polyhedron is empty also True), False otherwise.
        """

        # check if it has been already checked
        if self._bounded is not None:
            return self._bounded

        # check emptyness
        if self.empty:
            return True

        # include equalities
        A = np.vstack((self.A, self.C, -self.C))

        # check kernel of A
        if nullspace_basis(A).shape[1] > 0:
            return False

        # check Stiemke's theorem of alternatives
        n, m = A.shape
        sol = linear_program(
            np.ones(n), # f
            -np.eye(n),      # A
            -np.ones(n), # b
            A.T,             # C
            np.zeros(m) # d
            )
        self._bounded = sol['min'] is not None

        return self._bounded

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
        P2 : instance of Polyhedron
            Polyhedron within which we want to check if this polyhedron is contained.
        tol : float
            Maximum distance of a point from P2 to be considered an internal point.

        Returns
        ----------
        included : bool
            True if this polyhedron is contained in P2, False otherwise.
        """

        # augment inequalities with equalities
        A1 = np.vstack((self.A, self.C, -self.C))
        b1 = np.concatenate((self.b, self.d, -self.d))
        P1 = Polyhedron(A1, b1)
        A2 = np.vstack((P2.A, P2.C, -P2.C))
        b2 = np.concatenate((P2.b, P2.d, -P2.d))

        # check inclusion, one facet per time
        included = True
        for i in range(A2.shape[0]):
            sol = linear_program(-A2[i], P1.A, P1.b)
            penetration = - sol['min'] - b2[i]
            if penetration > tol:
                included = False
                break

        return included

    def intersection(self, P2):
        """
        Returns the intersection between this instance of Polyhedron (P1) and the polyhedron P2.

        Arguments
        ----------
        P2 : instance of Polyhedron
            Polyhedron with which we want to intersect this polyhedron.

        Returns
        ----------
        P3 : instance of Polyhedron
            Intersection of the two polyhedra.
        """

        # copy P3 (to not modify P2) and intersect
        P3 = copy(P2)
        P3.add_inequality(self.A, self.b)
        P3.add_equality(self.C, self.d)

        return P3

    def cartesian_product(self, P2):
        """
        Returns the Cartesian product between this instance of Polyhedron (P1) and the polyhedron P2.

        Arguments
        ----------
        P2 : instance of Polyhedron
            Polyhedron with which we want to multiply this polyhedron.

        Returns
        ----------
        P3 : instance of Polyhedron
            Cartesian product of the two polyhedra.
        """

        return Polyhedron(
            block_diag(self.A, P2.A),
            np.concatenate((self.b, P2.b)),
            block_diag(self.C, P2.C),
            np.concatenate((self.d, P2.d)),
            )

    @property
    def radius(self):
        """
        Returns the Chebyshev radius of the polyhedron (see self._chebyshev()).

        Returns
        ----------
        radius : float
            Chebyshev radius of the polytope (negative if the polyhedron is empty, None if it is unbounded).
        """

        # check if it has been already checked
        if self._radius is not None:
            return self._radius

        # compute Chebyshev radius and center
        self._radius, self._center = self._chebyshev()

        return self._radius

    @property
    def center(self):
        """
        Returns the Chebyshev center of the polyhedron (see self._chebyshev()).

        Returns
        ----------
        center : numpy.ndarray
            Chebyshev center of the polytope (None if the polyhedron is unbounded).
        """

        # check if it has been already checked
        if self._center is not None:
            return self._center

        # compute Chebyshev radius and center
        self._radius, self._center = self._chebyshev()

        return self._center

    def _chebyshev(self):
        """
        Returns the Chebyshev radius and center of the polyhedron P := {x | A x <= b, C x = d} solving the LP: min_{z, e}  e s.t. F z <= g + F_{row_norm} e.
        If no equalities are provided, F = A, z = x, g = b.
        In case of equality constraints, F = A N, g = b - A R r, with: N basis of the nullspace of C, R orthogonal complement to N, r = (C R)^-1 d and x is retrived as x = N n + R r.
        (For the details of this operation see the method _remove_equalities().)
        Here F_{row_norm} dentes the vector whose ith entry is the 2-norm of the ith row of F.

        Returns
        ----------
        radius : float
            Chebyshev radius of the polytope (negative if the polyhedron is empty, None if it is unbounded).
        center : numpy.ndarray
            Chebyshev center of the polytope (None if the polyhedron is unbounded).
        """

        # project in case of equalities
        if self.C.shape[0] > 0:
            A, b, N, R = self._remove_equalities()
        else:
            A = self.A
            b = self.b

        # assemble linear program
        f_lp = np.concatenate((np.zeros(A.shape[1]), np.ones(1)))
        A_row_norm = np.reshape(np.linalg.norm(A, axis=1), (A.shape[0], 1))
        A_lp = np.hstack((A, -A_row_norm))

        # solve and reshape result
        sol = linear_program(f_lp, A_lp, b)
        radius = sol['min']
        center = sol['argmin']
        if radius is not None:
            radius = -radius
            center = center[:-1]

        # go back to the original coordinates in case of equalities
        if self.C.shape[0] > 0:
            r = np.linalg.inv(self.C.dot(R)).dot(self.d)
            center = np.hstack((N, R)).dot(np.concatenate((center, r)))

        return radius, center

    @property
    def vertices(self):
        """
        Returns the set of vertices of the polyhdron.
        It assumes the polyhedron to be bounded (i.e. to be a polytope) and full dimensional (equality constraints are allowed but inequalities cannot make the polytope lower dimensional).

        Returns
        ----------
        vertices : list of numpy.ndarray
            List of the vertices of the bounded polyhedron (None if the polyhedron is unbounded or empty).
        """

        # check if it has been already checked
        if self._vertices is not None:
            return self._vertices

        # check boundedness
        if not self.bounded:
            return None

        # check full dimensionality
        tol = 1.e-7
        if self.radius < tol:
            return None

        # handle equalities
        if self.C.shape[0] > 0:
            A, b, N, R = self._remove_equalities()
            T = np.hstack((N, R))
            center = np.linalg.inv(T).dot(self.center)
            center = center[:N.shape[1]]
        else:
            A = self.A
            b = self.b
            center = self.center

        # handle 1d cases
        if A.shape[1] == 1:
            p = Polyhedron(A, b)
            p.remove_redundant_inequalities()
            self._vertices = [np.array([p.b[i] / p.A[i,0]]) for i in [0,1]]

        # call qhull through scipy
        else:
            halfspaces = np.column_stack((A, -b))
            polyhedron = HalfspaceIntersection(halfspaces, center)
            V = polyhedron.intersections
            self._vertices = [V[i] for i in range(V.shape[0])]

        # go back to the original coordinates in case of equalities
        if self.C.shape[0] > 0:
            r = np.linalg.inv(self.C.dot(R)).dot(self.d)
            self._vertices = [T.dot(np.concatenate((v, r))) for v in self._vertices]

        return self._vertices

    def project_to(self, residual_dimensions):
        """
        Returns the orthogonal projection of the polytope.

        Arguments
        ----------
        residual_dimensions : list of int
            List of indices of the residual dimensions after the projection.

        Returns
        ----------
        proj : instance of Polyhedron
            Orthogonal projection of this instance of Polyhedron onto the residual dimensions.
        """

        # check emptyness, boundedness, and full-dimensionality
        if self.empty:
            raise ValueError('cannot project empty polyhedra.')
        if not self.bounded:
            raise ValueError('cannot project unbounded polyhedra.')
        if self.C.shape[0] > 0:
            raise ValueError('cannot project lower-dimensional polyhedra.')

        # call convex-hull method for orthogonal projections
        A, b, vertices = convex_hull_method(self.A, self.b, residual_dimensions)
        proj = Polyhedron(A, b)
        proj._vertices = vertices

        return proj

    @staticmethod
    def from_convex_hull(points):
        """
        Instantiates the polyhedron given from the conve hull of the given set of points.
        It assumes the polyhedron to be bounded.

        Arguments
        ----------
        points : list of numpy.ndarray
            List of points.
        """

        # call qhull thorugh scipy for the convex hull
        hull = ConvexHull(np.vstack(points))

        # create polyhedron
        A = hull.equations[:, :-1]
        b = - hull.equations[:, -1:].flatten()
        p = Polyhedron(A, b)

        return p


    def plot(self, residual_dimensions=[0,1], **kwargs):
        """
        Plots the 2d projection of the polyhedron in the given dimension.
        It assumes the polyhedron to be bounded and not empty.

        Arguments
        ----------
        residual_dimensions : list of int
            Dimensions in which to project the polyhedron.
        lable : str
            Name of the polyhedron to be plot in the center of it.
        """

        # check dimensions
        if len(residual_dimensions) != 2:
            raise ValueError('wrong number of residual dimensions.')

        # extract vertices components
        if self.vertices is None:
            print('Cannot plot unbounded or empty polyhedra.')
            return

        # call qhull thorugh scipy for the convex hull (needed to order the vertices in counterclockwise order)
        vertices = np.vstack(self.vertices)[:,residual_dimensions]
        hull = ConvexHull(vertices)
        vertices = [hull.points[i].tolist() for i in hull.vertices]
        vertices += [vertices[0]]

        # create path
        codes = [Path.MOVETO] + [Path.LINETO]*(len(vertices)-2) + [Path.CLOSEPOLY]
        path = Path(vertices, codes)

        # set up plot
        ax = plt.gca()
        patch = patches.PathPatch(path, **kwargs)
        ax.add_patch(patch)
        plt.xlabel(r'$x_' + str(residual_dimensions[0]+1) + '$')
        plt.ylabel(r'$x_' + str(residual_dimensions[1]+1) + '$')
        ax.autoscale_view()

        return

def get_matrices_affine_expression(x, expr):
    """
    Extracts from the symbolic affine expression the matrices such that expr(x) = A x - b.
    
    Arguments
    ----------
    x : sympy matrix filled with sympy symbols
        Variables.
    expr : sympy matrix filled with sympy symbolic affine expressions
        Left hand side of the inequality constraint.
    """

    # state transition matrices
    A = np.array(expr.jacobian(x)).astype(np.float64)

    # offset term
    b = - np.array(expr.subs({xi:0 for xi in x})).astype(np.float64).flatten()
    
    return A, b

def convex_hull_method(A, b, resiudal_dimensions):
    """
    Given a bouned polyhedron in the form P := {x | A x <= b}, returns the orthogonal projection to the given dimensions.
    Dividing the space in the residual dimensions y and the dropped dimensions z, we have proj_y(P) := {y | exists z s.t. A_y y + A_z z < b}.
    The projection is returned in both the halfspace representation {x | E x <= f} and the vertices representation {x in conv(vertices)}.
    This is an implementation of the Convex Hull Method for orthogonal projections of polytopes, see, e.g., http://www.ece.drexel.edu/walsh/JayantCHM.pdf.
    The polyhedron is assumed to be bounded and full dimensional.

    Arguments
    ----------
    A : numpy.ndarray
        Left-hand side of the inequalities describing the higher dimensional polytope.
    b : numpy.ndarray
        Right-hand side of the inequalities describing the higher dimensional polytope.
    residual_dimensions : list of int
        Indices of the dimensions onto which the polytope has to be projected.

    Returns
    ----------
    E : numpy.ndarray
        Left-hand side of the inequalities describing the projection.
    f : numpy.ndarray
        Right-hand side of the inequalities describing the projection.
    vertices : list of numpy.ndarray
        List of the vertices of the projection.
    """

    # reorder coordinates
    n = len(resiudal_dimensions)
    dropped_dimensions = [i for i in range(A.shape[1]) if i not in resiudal_dimensions]
    A = np.hstack((
        A[:, resiudal_dimensions],
        A[:, dropped_dimensions]
        ))

    # initialize projection
    vertices = _get_two_vertices(A, b, n)
    if n == 1:
        E = np.array([[1.],[-1.]])
        f = np.array([
            max(v[0] for v in vertices),
            - min(v[0] for v in vertices)
            ])
        return E, f, vertices
    vertices = _get_inner_simplex(A, b, vertices)

    # expand facets
    hull = ConvexHull(
        np.vstack(vertices),
        incremental=True
        )
    hull = _expand_simplex(A, b, hull)
    hull.close()

    # get outputs
    E = hull.equations[:, :-1]
    f = - hull.equations[:, -1:].flatten()
    vertices = hull.points

    return E, f, vertices

def _get_two_vertices(A, b, n):
    """
    Findes two vertices of the projection.

    Arguments
    ----------
    A : numpy.ndarray
        Left-hand side of the inequalities describing the higher dimensional polytope.
    b : numpy.ndarray
        Right-hand side of the inequalities describing the higher dimensional polytope.
    n : int
        Dimensionality of the space onto which the polytope has to be projected.

    Returns
    ----------
    vertices : list of numpy.ndarray
        List of two vertices of the projection.
    """

    # select any direction to explore (it has to belong to the projected space, i.e. a_i = 0 for all i > n)
    a = np.concatenate((
        np.ones(1),
        np.zeros(A.shape[1]-1)
        ))

    # minimize and maximize in the given direction
    vertices = []
    for f in [a, -a]:
        sol = linear_program(f, A, b)
        vertices.append(sol['argmin'][:n])

    return vertices

def _get_inner_simplex(A, b, vertices, tol=1.e-7):
    """
    Constructs a simplex contained in the porjection.

    Arguments
    ----------
    A : numpy.ndarray
        Left-hand side of the inequalities describing the higher dimensional polytope.
    b : numpy.ndarray
        Right-hand side of the inequalities describing the higher dimensional polytope.
    vertices : list of numpy.ndarray
        List of two vertices of the projection.
    tol : float
        Maximal expansion of a facet to consider it a facet of the projection.

    Returns
    ----------
    vertices : list of numpy.ndarray
        List of vertices of the simplex contained in the projection.
    """

    # initialize LPs
    n = vertices[0].size

    # expand increasing at every iteration the dimension of the space
    for i in range(2, n+1):
        a, d = plane_through_points([v[:i] for v in vertices])
        f = np.concatenate((a, np.zeros(A.shape[1]-i)))
        sol = linear_program(f, A, b)

        # check the length of the expansion wrt to the plane, if zero expand in the opposite direction
        expansion = np.abs(a.dot(sol['argmin'][:i]) - d) # >= 0
        if expansion < tol:
            sol = linear_program(-f, A, b)
        vertices.append(sol['argmin'][:n])

    return vertices

def _expand_simplex(A, b, hull, tol=1.e-7):
    """
    Expands the internal simplex to cover all the projection.

    Arguments
    ----------
    A : numpy.ndarray
        Left-hand side of the inequalities describing the higher dimensional polytope.
    b : numpy.ndarray
        Right-hand side of the inequalities describing the higher dimensional polytope.
    hull : instance of ConvexHull
        Convex hull of vertices of the input simplex.
    tol : float
        Maximal expansion of a facet to consider it a facet of the projection.

    Returns
    ----------
    hull : instance of ConvexHull
        Convex hull of vertices of the projection.
    """

    # initialize algorithm's variables
    n = hull.points[0].size
    a_explored = []

    # start convex-hull method
    convergence = False
    while not convergence:
        convergence = True

        # check if every facet of the inner approximation belongs to the projection
        for i in range(hull.equations.shape[0]):

            # get normalized halfplane {x | a' x <= d} of the ith facet
            a = hull.equations[i, :-1]
            d = - hull.equations[i, -1]
            a_norm = np.linalg.norm(a)
            a /= a_norm
            b /= a_norm

            # check if the direction a has been explored so far
            is_explored = any((np.allclose(a, a2) for a2 in a_explored))
            if not is_explored:
                a_explored.append(a)

                # maximize in the direction a
                f = np.concatenate((
                    - a,
                    np.zeros(A.shape[1]-n)
                    ))
                sol = linear_program(f, A, b)

                # check if expansion wrt to the halfplane is greater than zero
                expansion = - sol['min'] - d # >= 0
                if expansion > tol:
                    convergence = False
                    hull.add_points(sol['argmin'][:n].reshape(1,n))
                    break

    return hull