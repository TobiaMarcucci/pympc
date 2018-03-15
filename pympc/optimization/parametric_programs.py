# external imports
import numpy as np
from copy import copy

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.convex_programs import QuadraticProgram
from pympc.optimization.gurobi import mixed_integer_quadratic_program

class MultiParametricQuadraticProgram(object):
    """
    mpQP in the form
                      |u|' |Huu  Hux| |u|   |fu|' |u|
    V(x) := min_u 1/2 |x|  |Hux' Hxx| |x| + |fx|  |x| + g
             s.t. Au u + Ax x <= b
    """

    def __init__(self, H, f, g, A, b):
        """
        Instantiates the parametric mpQP.

        Arguments
        ----------
        H : dict of numpy.ndarray
            Blocks of the quaratic term, entries: 'xx', 'ux', 'xx'.
        f : dict of numpy.ndarray
            Blocks of the linear term, entries: 'x', 'u'.
        g : numpy.ndarray
            Offset term in the cost function.
        A : dict of numpy.ndarray
            Left-hand side of the constraints, entries: 'x', 'u'.
        b : numpy.ndarray
            Right-hand side of the constraints.
        """
        self.H = H
        self.Huu_inv = np.linalg.inv(self.H['uu'])
        self.f = f
        self.g = g
        self.A = A
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

        # ensure that LICQ will hold
        Aua = self.A['u'][active_set, :]
        if len(active_set) > 0  and np.linalg.matrix_rank(Aua) < Aua.shape[0]:
            return None

        # split active and inactive
        inactive_set = [i for i in range(self.A['x'].shape[0]) if i not in active_set]
        Aui = self.A['u'][inactive_set, :]
        Axa = self.A['x'][active_set, :]
        Axi = self.A['x'][inactive_set, :]
        ba = self.b[active_set, :]
        bi = self.b[inactive_set, :]

        # multipliers
        M = np.linalg.inv(Aua.dot(self.Huu_inv).dot(Aua.T))
        pax = M.dot(Axa - Aua.dot(self.Huu_inv).dot(self.H['ux']))
        pa0 = - M.dot(ba + Aua.dot(self.Huu_inv).dot(self.f['u']))
        px = np.zeros(self.A['x'].shape)
        p0 = np.zeros((self.A['x'].shape[0], 1))
        px[active_set, :] = pax
        p0[active_set, :] = pa0
        p = {'x': px, '0':p0}

        # primary variables
        ux = - self.Huu_inv.dot(self.H['ux'] + Aua.T.dot(pax))
        u0 = - self.Huu_inv.dot(self.f['u'] + Aua.T.dot(pa0))
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
        cr = Polyhedron(Acr, bcr)
        cr.normalize()

        # optimal value function V(x) = 1/2 x' Vxx x + Vx' x + V0
        Vxx = ux.T.dot(self.H['uu']).dot(ux) + 2.*self.H['ux'].T.dot(ux) + self.H['xx']
        Vx = (ux.T.dot(self.H['uu'].T) + self.H['ux'].T).dot(u0) + ux.T.dot(self.f['u']) + self.f['x']
        V0 = .5*u0.T.dot(self.H['uu']).dot(u0) + self.f['u'].T.dot(u0) + self.g
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
            cr = self.explicit_solve_given_active_set(active_set_guess)
            if cr is not None and cr.contains(x):
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
        D = Polyhedron(self.A['u'], self.b - self.A['x'].dot(x))
        f = self.H['ux'].dot(x) + self.f['u']

        # solve QP
        qp = QuadraticProgram(D, self.H['uu'], f)
        sol = qp.solve()

        # "lift" optimal value function
        if sol['min'] is not None:
            sol['min'] += (.5*x.T.dot(self.H['xx']).dot(x) + self.f['x'].T.dot(x) + self.g)[0,0]

        return sol

    def solve(self, step_size=1.e-5, verbose=False):
        """
        Returns the explicit solution of the mpQP.
        It assumes that the facet-to-facet property holds (i.e. each facet of a critical region is shared with another single critical region).
        The following is a simple home-made algorithm.
        For every critical region, it looks at its non-redundat inequalities and guesses the active set beyond them.
        It then solves the KKTs for the given active set and check if the guess was right, if not it solves a QP to get the right active set and solves the KKTs again. 

        Arguments
        ----------
        step_size : float
            Size of the step taken to explore a new critical region from the facet of its parent.
        verbose : bool
            If True it prints the number active sets found at each iteration of the solver.

        Returns
        ----------
        instance of ExplicitSolution
            Explicit solution of the mpQP.

        """

        # start from the origin and guess its active set
        x = np.zeros(self.f['x'].shape)
        active_set_guess = []
        x_buffer = [(x, active_set_guess)]
        crs_found = []

        # loop until the are no points left 
        while len(x_buffer) > 0:

            # get critical region for the first point in the buffer
            cr = self.explicit_solve_given_point(*x_buffer[0])
            del x_buffer[0]

            # if feasible
            if cr is not None:

                # clean buffer from the points covered by the new critical region
                x_buffer = [x for x in x_buffer if not cr.contains(x[0])]

                # step outside each minimal facet
                for i in cr.minimal_facets():
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
                if verbose:
                    print('Critical region found: ' + str(len(crs_found)) + '.     \r'),

        return ExplicitSolution(crs_found)

    def get_feasible_set(self):
        """
        Returns the feasible set of the mqQP, i.e. {x | exists u: Au u + Ax x <= b}.

        Returns
        ----------
        instance of Polyhedron
            Feasible set.
        """

        # constraint set
        C = Polyhedron(
            np.hstack((self.A['x'], self.A['u'])),
            self.b
            )

        # feasible set
        return C.project_to(range(self.A['x'].shape[1]))

class CriticalRegion(object):
    """
    Critical Region (CR) of a multi-parametric quadratic program.
    A CR is the region of space where a fixed active set is optimal.
    """

    def __init__(self, active_set, u, p, V, polyhedron):
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
        polyhedron : instance of Polyhedron
            Region of space where the given active set is actually optimal.
        """

        self.active_set = active_set
        self._u = u
        self._p = p
        self._V = V
        self.polyhedron = polyhedron

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
        return self.polyhedron.contains(x)

    def minimal_facets(self):
        """
        Returns the minimal facets of the critical region.

        Returns
        ----------
        list of int
            List of indices of the non-redundant inequalities.
        """

        return self.polyhedron.minimal_facets()

    def facet_center(self, i):
        """
        Returns the Cebyshec center of the i-th facet.
        Implementation note: it is necessary to add the facet as an equality constraint, otherwise if we add it as an inequality the radius of the facet is zero and the center ends up in a vertex of the facet itself, and stepping out the facet starting from  vertex will not find the neighbour critical region.

        Arguments
        ----------
        i : int
            Index of the inequality associated with the facet we want to get the center of.

        Returns
        ----------
        numpy.ndarray
            Chebyshev center of the i-th facet.
        """

        # handle 1-dimensional case
        if self.polyhedron.A.shape[1] == 1:
            return np.linalg.inv(self.polyhedron.A[i:i+1, :]).dot(self.polyhedron.b[i:i+1, :])

        # add an equality to the original polyhedron
        facet = copy(self.polyhedron)
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
            Left hand side of self.polyhedron.
        """
        return self.polyhedron.A

    @property
    def b(self):
        """
        Right hand side of the inequalities describing the critical region.

        Returns
        ----------
        numpy.ndarray
            Right hand side of self.polyhedron.
        """
        return self.polyhedron.b

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

class MultiParametricMixedIntegerQuadraticProgram(object):
    """
    Multiparametric Mixed Integer Quadratic Program (mpMIQP) in the form that comes out from the MPC problem fro a piecewise affine system, i.e.
                                |u|' |Huu   0 0   0| |u|
                                |z|  |  0 Hzz 0 Hzx| |z|
        V(x) := min_{u,z,d} 1/2 |d|  |        0   0| |d|
                                |x|  |sym       Hxx| |x|
                      s.t. Au u + Az z + Ad d + Ax x <= b
        where:
        u := (u(0), ..., u(N-1)), continuous,
        z := (z(0), ..., z(N-1)), continuous,
        d := (d(0), ..., d(N-1)), binary,
        while x  is the intial condition.
    """
    def __init__(self, H, A, b):
        self.H = H
        self.A = A
        self.b = b

    def solve(self, x):
        sol = mixed_integer_quadratic_program(
            self.H['uu'],
            self.H['zz'],
            self.H['zx'].dot(x),
            self.A['u'],
            self.A['z'],
            self.A['d'],
            self.b - self.A['x'].dot(x)
            )
        if sol['min'] is not None:
            sol['min'] += .5*x.T.dot(self.H['xx']).dot(x)[0,0]
        return sol