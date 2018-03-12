# external imports
import numpy as np
from copy import copy

# internal inputs
from pympc.optimization.pnnls import linear_program, quadratic_program
from pympc.geometry.polyhedron import Polyhedron

class LinearProgram(object):
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

class QuadraticProgram(object):
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

class MultiParametricQuadraticProgram(object):
    """
    mpQP in the form
                      |u|' |Huu  Hux| |u|   |fu|' |u|
    V(x) := min_u 1/2 |x|  |Hux' Hxx| |x| + |fx|  |x| + g
             s.t. Au u + Ax x <= b
    """

    def __init__(self, Huu, Hux, Hxx, fu, fx, g, Au, Ax, b):
        """
        Instantiates the parametric mpQP.

        Arguments : numpy.ndarray
        """
        self.Huu = Huu
        self.Hxx = Hxx
        self.Hux = Hux
        self.fu = fu
        self.fx = fx
        self.g = g
        self.Au = Au
        self.Ax = Ax
        self.b = b

    def explicit_solve_given_active_set(self, active_set):
        """
        Returns the explicit solution of the mpQP for a given active set.
        The solution turns out to be an affine function of x, i.e. u(x) = ux x + u0, p(x) = px x + p0, where p are the Lagrange multipliers for the inequality constraints.

        Math
        ----------
        Given an active set A and a set of multipliers p, the KKT conditions for the mpQP are a set of linear equations that can be solved for u(x) and p(x)
        Huu u + Hux + fu + Aua' pa = 0, (stationarity of the Lagrangian),
        Aua u + Axa x = ba,             (primal feasibility),
        pi = 0,                         (dual feasibility),
        where the subscripts a and i dentote active and inactive inequalities respectively.
        The inactive (primal and dual) constraints define the region of space where the given active set is optimal
        Aui u + Axi x < bi, (primal feasibility),
        pa > 0,             (dual feasibility).
        

        Arguments
        ----------
        active_set : list of int
            Indices of the active inequalities.

        Reuturns
        ----------
        instance of CriticalRegion
            Critical region for the given active set.
        """

        # split active and inactive
        inactive_set = [i for i in range(self.Ax.shape[0]) if i not in active_set]
        Aua = self.Au[active_set, :]
        Aui = self.Au[inactive_set, :]
        Axa = self.Ax[active_set, :]
        Axi = self.Ax[inactive_set, :]
        ba = self.b[active_set, :]
        bi = self.b[inactive_set, :]

        # multipliers
        Huu_inv = np.linalg.inv(self.Huu)
        M = np.linalg.inv(Aua.dot(Huu_inv).dot(Aua.T))
        pax = M.dot(Axa - Huu_inv.dot(self.Hux))
        pa0 = - M.dot(ba + Huu_inv.dot(self.fu))
        px = np.zeros(self.Ax.shape)
        p0 = np.zeros(self.Ax.shape[0])
        px[active_set, :] = pax
        p0[active_set, :] = pa0
        p = {'x': px, '0':p0}

        # primary variables
        ux = - Huu_inv.dot(self.Hux + Aua.T.dot(pax))
        u0 = - Huu_inv.dot(self.fu + Aua.T.dot(pa0))
        u = {'x':ux, '0':u0}

        # critical region
        Acr = np.vstack((
            - pax,
            Aui.dot(ux) + Axi
            ))
        bcr = np.vstack((
            pa0,
            bi - Aui.dot(u0)
            ))
        cr = Polyhedron(Acd, bcr)

        # optimal value function V(x) = 1/2 x' Vxx x + Vx' x + V0
        Vxx = ux.T.dot(self.Huu).dot(ux) + 2.*self.Hux.T.dot(ux) + self.Hxx
        Vx = (ux.T.dot(Huu.T) + self.Hux.T).dot(u0) + ux.T.dot(self.fu) + self.fx
        V0 = .5*u0.T.dot(self.Huu).dot(u0) + self.fu.T.dot(u0) + self.g
        V = {'xx':Vxx, 'x':Vx, '0':V0}

        return CriticalRegion(active_set, u, p, V, cr)

    def explicit_solve_given_point(self, x, active_set_guess=None):
        """
        Returns the explicit solution of the mpQP at a given point.
        In case a guess for the active set is provided, it first tries it.

        Arguments
        ----------
        x : numpy.ndarray
            Point where we want to get the solution.
        active_set_guess : list of int
            Indices of the inequalities we guess are active at the given point.

        Reuturns
        ----------
        instance of CriticalRegion
            Critical region that covers the give point (None if the given x is unfeasible).
        """

        # first try the guess for the active set
        if active_set_guess is not None:
            cr = self.explicit_solve_given_active_set(active_set)
            if cr.contains(x):
                return cr

        # otherwise solve the QP to get the active set
        sol = self.implicit_solve_fixed_point(x)
        if sol['active_set'] is None:
            return None

        return self.explicit_solve_given_active_set(sol['active_set'])

    def implicit_solve_fixed_point(self, x):
        """
        Solves the QP at the given point x.

        Arguments
        ----------
        x : numpy.ndarray
            Point where we want to get the solution.

        Returns
        ----------
        sol : dict
            Dictionary with the solution of the QP (see the documentation of pympc.optimization.pnnls.quadratic_program for the details of the fields of sol).

        """

        # fix constraints and cost function
        D = Polyhedron(self.Au, self.b - self.Ax.dot(x))
        f = self.Hux.dot(x) + self.fu

        # solve QP
        qp = QuadraticProgram(D, self.Huu, f)
        sol = qp.solve()

        # "lift" optimal alue function
        sol['min'] += .5*x.T.dot(self.Hxx).dot(x) + self.fx.T.dot(x) + self.g

        return sol

    def solve(self, step_size=1.e-5):
        """
        Returns the explicit solution of the mpQP.
        It assumes that the facet-to-facet property holds (i.e. each facet of a critical region is shared with another single critical region).

        Arguments
        ----------
        step_size : float
            Size of the step taken to explore a new critical region from the facet of its parent.

        Returns
        ----------
        instance of ExplicitSolution
            Explicit solution of the mpQP.

        """

        # start from the origin and guess its active set
        x = np.zeros((self.A.shape[0], 1))
        active_set_guess = []
        x_buffer = [(x, active_set_guess)]
        crs_found = []

        # loop until the are no points left 
        while len(x_buffer) > 0:

            # get critical region for the first point in the buffer
            cr = self.explicit_solve_given_point(x_buffer[0])
            del x_buffer[0]

            # if feasible
            if cr is not None:

                # step outside each minimal facet
                for i in cr.minimal_facets:
                    x = cr.facet_center(i) + step_size*cr.A[i:i+1,:].T

                    # check if the new point has been already explored
                    if not any([cr_found.contains(x) for cr_found in crs_found]):

                        # guess the active set on the other side of the facet
                        if i in cr.active_set:
                            active_set_guess = [j for j in cr.active_set if j != i]
                        else:
                            active_set_guess = sorted(cr.active_set + [i])

                        # add to the buffes
                        x_buffer.append((x, active_set_guess))

                # if feasible, add the the list of critical regions
                crs_found.append(cr)

        return ExplicitSolution(crs_found)
                
class CriticalRegion(object):
    """
    Critical Region (CR) of a multi-parametric quadratic program.
    A CR is the region of space where a fixed active set is optimal.
    """

    def __init__(self, active_set, u, p, V, polytope):
        """
        Instatiates the critical region.

        Arguments
        ----------
        active_set : list of int
            List of the indices of the active inequalities.
        u : dict
            Explicit primal solution for the given active set (with keys: 'x' for the linear term, '0' for the offset term).
        p : dict
            Explicit dual solution for the given active set (with keys: 'x' for the linear term, '0' for the offset term).
        V : dict
            Explicit expression of the optimal value function for the given active set (with keys: 'xx' for the quadratic term, 'x' for the linear term, '0' for the offset term).
        polytope : instance of Polyhedron
            Region of space where the given active set is actually optimal.
        """

        self.active_set = active_set
        self._u = u
        self._p = p
        self._V = V
        self.polytope = polytope

    def contains(self, x):
        """
        Checks if the point x is inside the critical region.

        Arguments
        ----------
        x : numpy.ndarray
            Point we want to check.

        Returns
        ----------
        bool
            True if the point is contained in the critical region, False otherwise.
        """
        return self.polytope.contains(x)

    def minimal_facets(self):
        """
        Returns the minimal facets of the critical region.

        Returns
        ----------
        list of int
            List of indices of the non-redundant inequalities.
        """

        return self.polytope.minimal_facets()

    def facet_center(self, i):
        """
        Returns the Cebyshec center of the i-th facet.

        Arguments
        ----------
        i : int
            Index of the inequality associated with the facet we want to get the center of.

        Returns
        ----------
        numpy.ndarray
            Chebyshev center of the i-th facet.
        """

        # add an equality to the original polytope
        facet = copy(self.polytope)
        facet.add_equality(
            facet.A[i:i+1, :],
            facet.b[i:i+1, :]
            )

        return facet.center
        
    def u(self, x):
        """
        Numeric value of the primal optimizer at the point x.

        Arguments
        ----------
        x : numpy.ndarray
            Point where we want to get the solution.

        Returns
        ----------
        numpy.ndarray
            Primal optimizer at the given point.
        """

        return self._u['x'].dot(x) + self._u['0']

    def p(self, x):
        """
        Numeric value of the dual optimizer at the point x.

        Arguments
        ----------
        x : numpy.ndarray
            Point where we want to get the solution.

        Returns
        ----------
        numpy.ndarray
            Dual optimizer at the given point.
        """

        return self._p['x'].dot(x) + self._p['0']

    def V(self, x):
        """
        Numeric value of the optimal value function at the point x.

        Arguments
        ----------
        x : numpy.ndarray
            Point where we want to get the solution.

        Returns
        ----------
        float
            Optimal value function at the given point.
        """

        V = .5*x.T.dot(self._V['xx']).dot(x) + self._V['x'].T.dot(x) + self._V['0']

        return V[0,0]

    @property
    def A(self):
        """
        Left hand side of the inequalities describing the critical region.

        Returns
        ----------
        numpy.ndarray
            Left hand side of self.polytope.
        """
        return self.polytope.A

    @property
    def b(self):
        """
        Right hand side of the inequalities describing the critical region.

        Returns
        ----------
        numpy.ndarray
            Right hand side of self.polytope.
        """
        return self.polytope.b

class ExplicitSolution(object):
    """
    Explicit solution of a multiparametric quadratic program.
    """

    def __init__(self, critical_regions):
        """
        Stores the set of critical regions.

        Arguments
        ----------
        critical_regions : list of intances of CriticalRegion
            List of crtical regions for the solution of the mpQP.
        """
        self.critical_regions = critical_regions

    def get_critical_region(self, x):
        """
        Returns the critical region that covers the given point.

        Arguments
        ----------
        x : numpy.ndarray
            Point where we want to get the critical region.

        Returns
        ----------
        instance of CriticalRegion
            Critical region that covers the given point (None if the point is not covered).
        """

        # loop over the critical regions
        for cr in self.critical_regions:
            if cr.contains(x):
                return cr

        # return None if not covered
        return None

    def u(self, x):
        """
        Numeric value of the primal optimizer at the point x.

        Arguments
        ----------
        x : numpy.ndarray
            Point where we want to get the solution.

        Returns
        ----------
        numpy.ndarray
            Primal optimizer at the given point (None if the point is not covered).
        """

        # loop over the critical regions
        cr = self.get_critical_region(x)
        if cr is not None:
            return cr.u(x)

        # return None if not covered
        return None

    def p(self, x):
        """
        Numeric value of the dual optimizer at the point x.

        Arguments
        ----------
        x : numpy.ndarray
            Point where we want to get the solution.

        Returns
        ----------
        numpy.ndarray
            Dual optimizer at the given point (None if the point is not covered).
        """

        # loop over the critical regions
        cr = self.get_critical_region(x)
        if cr is not None:
            return cr.p(x)

        # return None if not covered
        return None

    def V(self, x):
        """
        Numeric value of the optimal value function at the point x.

        Arguments
        ----------
        x : numpy.ndarray
            Point where we want to get the solution.

        Returns
        ----------
        float
            Optimal value function at the given point (None if the point is not covered).
        """

        # loop over the critical regions
        cr = self.get_critical_region(x)
        if cr is not None:
            return cr.V(x)

        # return None if not covered
        return None