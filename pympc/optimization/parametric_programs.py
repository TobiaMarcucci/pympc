# external imports
import numpy as np
from scipy.linalg import block_diag
from copy import copy

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.programs import quadratic_program, mixed_integer_quadratic_program

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
            Blocks of the quaratic term, keys: 'xx', 'ux', 'xx'.
        f : dict of numpy.ndarray
            Blocks of the linear term, keys: 'x', 'u'.
        g : numpy.ndarray
            Offset term in the cost function.
        A : dict of numpy.ndarray
            Left-hand side of the constraints, keys: 'x', 'u'.
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
        Aua = self.A['u'][active_set]
        if len(active_set) > 0  and np.linalg.matrix_rank(Aua) < Aua.shape[0]:
            return None

        # split active and inactive
        inactive_set = [i for i in range(self.A['x'].shape[0]) if i not in active_set]
        Aui = self.A['u'][inactive_set]
        Axa = self.A['x'][active_set]
        Axi = self.A['x'][inactive_set]
        ba = self.b[active_set]
        bi = self.b[inactive_set]

        # multipliers
        M = np.linalg.inv(Aua.dot(self.Huu_inv).dot(Aua.T))
        pax = M.dot(Axa - Aua.dot(self.Huu_inv).dot(self.H['ux']))
        pa0 = - M.dot(ba + Aua.dot(self.Huu_inv).dot(self.f['u']))
        px = np.zeros(self.A['x'].shape)
        p0 = np.zeros(self.A['x'].shape[0])
        px[active_set] = pax
        p0[active_set] = pa0
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
        bcr = np.concatenate((
            pa0,
            bi - Aui.dot(u0)
            ))
        cr = Polyhedron(Acr, bcr)
        cr.normalize()

        # optimal value function V(x) = 1/2 x' Vxx x + Vx' x + V0
        Vxx = ux.T.dot(self.H['uu']).dot(ux) + 2.*self.H['ux'].T.dot(ux) + self.H['xx']
        Vx = (ux.T.dot(self.H['uu'].T) + self.H['ux'].T).dot(u0) + ux.T.dot(self.f['u']) + self.f['x']
        V0 = .5*u0.dot(self.H['uu']).dot(u0) + self.f['u'].dot(u0) + self.g
        V = {'xx':Vxx, 'x':Vx, '0':V0}

        return CriticalRegion(active_set, u, p, V, cr)

    def explicit_solve_given_point(self, x, active_set_guess=None, verbose=False):
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
            elif verbose:
                print('Wrong active-set guess:'),

        # otherwise solve the QP to get the active set
        sol = self.solve(x)
        if sol['active_set'] is None:
            if verbose:
                print('unfeasible sample.')
            return None
        if verbose:
            print('feasible sample with active set ' + str(sol['active_set']) + '.')

        return self.explicit_solve_given_active_set(sol['active_set'])

    def solve(self, x):
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

        # fix cost function and constraints
        f = self.H['ux'].dot(x) + self.f['u']
        b = self.b - self.A['x'].dot(x)
        sol = quadratic_program(self.H['uu'], f, self.A['u'], b)

        # "lift" optimal value function
        if sol['min'] is not None:
            sol['min'] += .5 * x.dot(self.H['xx']).dot(x) + self.f['x'].dot(x) + self.g

        return sol

    def explicit_solve(self, step_size=1.e-5, verbose=False):
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
        x = np.zeros(self.f['x'].size)
        active_set_guess = []
        x_buffer = [(x, active_set_guess)]
        crs_found = []

        # loop until the are no points left
        while len(x_buffer) > 0:

            # discard points that have been already covered
            x_buffer = [x for x in x_buffer if not any([cr.contains(x[0]) for cr in crs_found])]
            if len(x_buffer) == 0:
                break

            # get critical region for the first point in the buffer
            cr = self.explicit_solve_given_point(x_buffer[0][0], x_buffer[0][1], verbose)
            del x_buffer[0]

            # if feasible
            if cr is not None:

                # step outside each minimal facet
                for i in cr.minimal_facets():
                    x = cr.facet_center(i) + step_size * cr.A[i]

                    # guess the active set on the other side of the facet
                    active_set_guess = set(cr.active_set).symmetric_difference({i})

                    # add to the buffer
                    x_buffer.append((x, sorted(list(active_set_guess))))

                # if feasible, add the the list of critical regions
                crs_found.append(cr)
                if verbose:
                    print('CR found, active set: ' + str(cr.active_set) + '.')
        if verbose:
            print('Explicit solution found, CRs are: ' + str(len(crs_found)) + '.')

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
            return self.polyhedron.b.flatten()[i] / self.polyhedron.A[i][0]

        # add an equality to the original polyhedron
        facet = copy(self.polyhedron)
        facet.add_equality(
            facet.A[i:i+1, :],
            facet.b[i:i+1]
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

        V = .5*x.dot(self._V['xx']).dot(x) + self._V['x'].dot(x) + self._V['0']

        return V

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
    Multiparametric Mixed Integer Quadratic Program (mpMIQP) in the form that comes out from the MPC problem for a piecewise affine system, i.e.
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
        """
        Initializes the mpMIQP.

        Arguments
        -----------
        H : dict of numpy.ndarry
            Dictionary with the blocks of the cost function Hessian, keys: 'uu', 'zz', 'xx', 'zx'.
        A : dict of numpy.ndarry
            Dictionary with the blocks of the constraint Jacobian, keys: 'u', 'z', 'd', 'x'.
        b : numpy.ndarray
            Right-hand side of the constraints.
        """

        # store matrices
        self.H = H
        self.A = A
        self.b = b

    def solve(self, x):
        """
        Solves the mpMIQP for the given value of the parameter x.

        Arguments
        ----------
        x : numpy.ndarry
            Numeric value of the parameter vector.

        Returns
        ----------
        sol : dict
            Dictionary with the solution of the MIQP, keys: 'min', 'u', 'z', 'd'.
        """

        # MIQP dimensions
        nu = self.A['u'].shape[1]
        nz = self.A['z'].shape[1]
        nd = self.A['d'].shape[1]
        nc = nu + nz

        # put MIQP in standard form
        H = block_diag(
            self.H['uu'],
            self.H['zz'],
            np.zeros((nd, nd))
            )
        f = np.concatenate((
            np.zeros(nu),
            self.H['zx'].dot(x),
            np.zeros(nd)
            ))
        A = np.hstack((
            self.A['u'],
            self.A['z'],
            self.A['d']
            ))
        b = self.b - self.A['x'].dot(x)

        # solve MIQP
        sol_sf = mixed_integer_quadratic_program(nc, H, f, A, b)

        # reshape solution
        sol = {
            'min': sol_sf['min'],
            'u': None,
            'z': None,
            'd': None
            }

        # if feasible lift the cost function with the offset term
        if sol['min'] is not None:
            sol['min'] += .5*x.dot(self.H['xx']).dot(x)
            sol['u'] = sol_sf['argmin'][:nu]
            sol['z'] = sol_sf['argmin'][nu:nu+nz]
            sol['d'] = sol_sf['argmin'][nu+nz:]

        return sol