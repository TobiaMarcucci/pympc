import numpy as np
import scipy.linalg as linalg
import itertools
import time
from pympc.ndpiecewise import NDPiecewise
from pnnls import linear_program
from gurobi import quadratic_program
from pympc.geometry import Polytope

class MPQPSolver:
    """
    Solves a mp-QP in the form
    u^*(x) = argmin_u 0.5 u' H u + x' F u
    subject to G u <= W + E x
    """

    def __init__(self, canonical_qp):

        self.qp = canonical_qp

        # start clock
        tic = time.time()

        # initialize the search with the origin (to which the empty AS is associated)
        active_set = []
        cr0 = CriticalRegion(active_set, self.qp)
        cr_to_be_explored = [cr0]
        explored_cr = []
        tested_active_sets =[cr0.active_set]

        # explore the state space
        while cr_to_be_explored:

            # choose the first candidate in the list and remove it
            cr = cr_to_be_explored[0]
            cr_to_be_explored = cr_to_be_explored[1:]

            # if the CR is not empty, find all the potential neighbors
            if cr.polytope.empty:
                print('Empty critical region detected')
            else:
                [cr_to_be_explored, tested_active_sets] = self.spread_critical_region(cr, cr_to_be_explored, tested_active_sets)
                explored_cr.append(cr)

        # collect all the critical regions and report the result
        self.critical_regions = NDPiecewise(explored_cr)
        toc = time.time()
        print('\nExplicit solution computed in ' + str(toc-tic) + ' s:')
        print('parameter space partitioned in ' + str(len(self.critical_regions)) + ' critical regions.')

    def spread_critical_region(self, cr, cr_to_be_explored, tested_active_sets):

        # for all the facets of the CR and all candidate active sets across each facet
        for facet_index in range(0, len(cr.polytope.minimal_facets)):
            for active_set in cr.candidate_active_sets[facet_index]:

                # add active set to the list of tested be sure to not explore an active set twice
                if active_set not in tested_active_sets:
                    tested_active_sets.append(active_set)

                    # check LICQ for the given active set
                    licq_flag = self.licq_check(self.qp.G, active_set)

                    # if LICQ holds, determine the critical region
                    if licq_flag:
                        cr_to_be_explored.append(CriticalRegion(active_set, self.qp))

                    # if LICQ doesn't hold, correct the active set and determine the critical region
                    else:
                        print('LICQ does not hold for the active set ' + str(active_set))
                        active_set = self.active_set_if_not_licq(active_set, facet_index, cr)
                        if active_set and active_set not in tested_active_sets:
                            print('    corrected active set ' + str(active_set))
                            cr_to_be_explored.append(CriticalRegion(active_set, self.qp))
                        else:
                            print('    unfeasible critical region detected')
        return [cr_to_be_explored, tested_active_sets]

    def active_set_if_not_licq(self, candidate_active_set, facet_index, cr, dist=1e-6, lambda_bound=1e6, toll=1e-6):
        """
        Returns the active set of a critical region in case that licq does not hold (Theorem 4 revisited)

        INPUTS:
            candidate_active_set: candidate active set for which LICQ doesn't hold
            facet_index: index of this active set hypothesis in the parent's list of neighboring active sets
            cr: critical region from which the new active set is generated

        OUTPUTS:
            active_set: real active set of the child critical region ([] if the region is unfeasible)
        """

        # differences between the active set of the parent and the candidate active set
        active_set_change = list(set(cr.active_set).symmetric_difference(set(candidate_active_set)))

        # if there is more than one change, nothing can be done...
        if len(active_set_change) > 1:
            print('Cannot solve degeneracy with multiple active set changes! The solution of a QP is required...')
            active_set = self.solve_qp_beyond_facet(facet_index, cr)

        # if there is one change solve the lp from Theorem 4
        else:
            active_set = self.solve_lp_on_facet(candidate_active_set, facet_index, cr)

        return active_set

    def solve_qp_beyond_facet(self, facet_index, cr, dist=1e-8, toll=1.e-6):

        """
        Solves a QP a step of length "dist" beyond the facet wich index is "facet_index"
        to determine the active set in that region.

        INPUTS:
            facet_index: index of this active set hypothesis in the parent's list of neighboring active sets
            cr: critical region from which the new active set is generated

        OUTPUTS:
            active_set: real active set of the child critical region ([] if the region is unfeasible)
        """
        
        # relate step distance with polytope dimensions
        dist = min(cr.polytope.facet_radii(facet_index)/100., cr.polytope.radius/100., dist)

        # relates step length to polytope dimension
        dist = min(dist, cr.polytope.radius, cr.polytope.facet_radii(facet_index))

        # center of the facet in the parameter space
        x_center = cr.polytope.facet_centers(facet_index)

        # solve the QP inside the new critical region to derive the active set
        x_beyond = x_center + dist*cr.polytope.lhs_min[facet_index,:].reshape(x_center.shape)
        x_beyond = x_beyond.reshape(x_center.shape[0],1)
        z = quadratic_program(self.qp.H, np.zeros((self.qp.H.shape[0],1)), self.qp.G, self.qp.W + self.qp.S.dot(x_beyond))[0]

        # new active set for the child
        constraints_residuals = self.qp.G.dot(z) - self.qp.W - self.qp.S.dot(x_beyond)
        active_set = [i for i in range(0,self.qp.G.shape[0]) if constraints_residuals[i] > -toll]

        return active_set

    def solve_lp_on_facet(self, candidate_active_set, facet_index, cr, toll=1e-6):
        """
        Solves a LP on the center of the facet wich index is "facet_index" to determine
        the active set in that region (Theorem 4)

        INPUTS:
            candidate_active_set: candidate active set for which LICQ doesn't hold
            facet_index: index of this active set hypothesis in the parent's list of neighboring active sets
            cr: critical region from which the new active set is generated

        OUTPUTS:
            active_set: real active set of the child critical region ([] if the region is unfeasible)
        """

        # differences between the active set of the parent and the candidate active set
        active_set_change = list(set(cr.active_set).symmetric_difference(set(candidate_active_set)))

        # compute optimal solution in the center of the shared facet
        x_center = cr.polytope.facet_centers(facet_index)
        z_center = cr.z_optimal(x_center)

        # solve lp from Theorem 4
        G_A = self.qp.G[candidate_active_set,:]
        n_lam = G_A.shape[0]
        cost = np.zeros((n_lam,1))
        cost[candidate_active_set.index(active_set_change[0])] = -1.
        cons_lhs = np.vstack((G_A.T, -G_A.T, -np.eye(n_lam)))
        cons_rhs = np.vstack((-self.qp.H.dot(z_center), self.qp.H.dot(z_center), np.zeros((n_lam,1))))
        sol = linear_program(cost, cons_lhs, cons_rhs)

        # if the solution in unbounded the region is unfeasible
        active_set = []
        if any(np.isnan(sol.argmin)):
            return active_set

        # if the solution in bounded look at the indices of the solution to derive the active set
        for i in range(0, n_lam):
            if sol.argmin[i] > toll:
                active_set += [candidate_active_set[i]]
        return active_set

    @staticmethod
    def licq_check(G, active_set, max_cond=1e9):
        """
        Checks if LICQ holds for the given active set

        INPUTS:
            G: gradient of the constraints
            active_set: active set
            max_cond: maximum condition number of the squared active constraints

        OUTPUTS:
            licq -> flag (True if licq holds, False if licq doesn't hold)
        """

        # select active constraints
        G_A = G[active_set,:]

        # check condion number of the squared active constraints
        licq = True
        cond = np.linalg.cond(G_A.dot(G_A.T))
        if cond > max_cond:
            licq = False

        return licq


class CriticalRegion:
    """
    Implements the algorithm from Tondel et al. "An algorithm for multi-parametric quadratic programming and explicit MPC solutions"

    VARIABLES:
        n_constraints: number of contraints in the qp
        n_parameters: number of parameters of the qp
        active_set: active set inside the critical region
        inactive_set: list of indices of non active contraints inside the critical region
        polytope: polytope describing the ceritical region in the parameter space
        weakly_active_constraints: list of indices of constraints that are weakly active iside the entire critical region
        candidate_active_sets: list of lists of active sets, its ith element collects the set of all the possible
            active sets that can be found crossing the ith minimal facet of the polyhedron
        z_linear: linear term in the piecewise affine primal solution z_opt = z_linear*x + z_offset
        z_offset: offset term in the piecewise affine primal solution z_opt = z_linear*x + z_offset
        u_linear: linear term in the piecewise affine primal solution u_opt = u_linear*x + u_offset
        u_offset: offset term in the piecewise affine primal solution u_opt = u_linear*x + u_offset
        lambda_A_linear: linear term in the piecewise affine dual solution (only active multipliers) lambda_A = lambda_A_linear*x + lambda_A_offset
        lambda_A_offset: offset term in the piecewise affine dual solution (only active multipliers) lambda_A = lambda_A_linear*x + lambda_A_offset
    """

    def __init__(self, active_set, qp):

        # store active set
        print 'Computing critical region for the active set ' + str(active_set)
        [self.n_constraints, self.n_parameters] = qp.S.shape
        self.active_set = active_set
        self.inactive_set = sorted(list(set(range(self.n_constraints)) - set(active_set)))

        # find the polytope
        self.polytope(qp)
        if self.polytope.empty:
            return

        # find candidate active sets for the neighboiring regions
        minimal_coincident_facets = [self.polytope.coincident_facets[i] for i in self.polytope.minimal_facets]
        self.candidate_active_sets = self.candidate_active_sets(active_set, minimal_coincident_facets)

        # find weakly active constraints
        self.find_weakly_active_constraints()

        # expand the candidates if there are weakly active constraints
        if self.weakly_active_constraints:
            self.candidate_active_set = self.expand_candidate_active_sets(self.candidate_active_set, self.weakly_active_constraints)

        return

    def polytope(self, qp):
        """
        Stores a polytope that describes the critical region in the parameter space.
        """

        # multipliers explicit solution
        [G_A, W_A, S_A] = [qp.G[self.active_set,:], qp.W[self.active_set,:], qp.S[self.active_set,:]]
        [G_I, W_I, S_I] = [qp.G[self.inactive_set,:], qp.W[self.inactive_set,:], qp.S[self.inactive_set,:]]
        H_A = np.linalg.inv(G_A.dot(qp.H_inv.dot(G_A.T)))
        self.lambda_A_offset = - H_A.dot(W_A)
        self.lambda_A_linear = - H_A.dot(S_A)

        # primal variables explicit solution
        self.z_offset = - qp.H_inv.dot(G_A.T.dot(self.lambda_A_offset))
        self.z_linear = - qp.H_inv.dot(G_A.T.dot(self.lambda_A_linear))

        # optimal value function explicit solution: V_star = .5 x' V_quadratic x + V_linear x + V_offset
        self.V_quadratic = self.z_linear.T.dot(qp.H).dot(self.z_linear) + qp.F_xx_q
        self.V_linear = self.z_offset.T.dot(qp.H).dot(self.z_linear) + qp.F_x_q.T
        self.V_offset = qp.F_q + .5*self.z_offset.T.dot(qp.H).dot(self.z_offset)

        # primal original variables explicit solution
        self.u_offset = self.z_offset - qp.H_inv.dot(qp.F_u)
        self.u_linear = self.z_linear - qp.H_inv.dot(qp.F_xu.T)

        # equation (12) (modified: only inactive indices considered)
        lhs_type_1 = G_I.dot(self.z_linear) - S_I
        rhs_type_1 = - G_I.dot(self.z_offset) + W_I

        # equation (13)
        lhs_type_2 = - self.lambda_A_linear
        rhs_type_2 = self.lambda_A_offset

        # gather facets of type 1 and 2 to define the polytope (note the order: the ith facet of the cr is generated by the ith constraint)
        lhs = np.zeros((self.n_constraints, self.n_parameters))
        rhs = np.zeros((self.n_constraints, 1))
        lhs[self.inactive_set + self.active_set, :] = np.vstack((lhs_type_1, lhs_type_2))
        rhs[self.inactive_set + self.active_set] = np.vstack((rhs_type_1, rhs_type_2))

        # construct polytope
        self.polytope = Polytope(lhs, rhs)
        self.polytope.assemble()

        return

    def find_weakly_active_constraints(self, toll=1e-8):
        """
        Stores the list of constraints that are weakly active in the whole critical region
        enumerated in the as in the equation G z <= W + S x ("original enumeration")
        (by convention weakly active constraints are included among the active set,
        so that only constraints of type 2 are anlyzed)
        """

        # equation (13), again...
        lhs_type_2 = - self.lambda_A_linear
        rhs_type_2 = self.lambda_A_offset

        # weakly active constraints are included in the active set
        self.weakly_active_constraints = []
        for i in range(0, len(self.active_set)):

            # to be weakly active in the whole region they can only be in the form 0^T x <= 0
            if np.linalg.norm(lhs_type_2[i,:]) + np.absolute(rhs_type_2[i,:]) < toll:
                print('Weakly active constraint detected!')
                self.weakly_active_constraints.append(self.active_set[i])

        return

    @staticmethod
    def candidate_active_sets(active_set, minimal_coincident_facets):
        """
        Computes one candidate active set for each non-redundant facet of a critical region
        (Theorem 2 and Corollary 1).

        INPUTS:
        active_set: active set of the parent critical region
        minimal_coincident_facets: list of facets coincident to the minimal facets
            (i.e.: [coincident_facets[i] for i in minimal_facets])

        OUTPUTS:
            candidate_active_sets: list of the candidate active sets for each minimal facet
        """

        # initialize list of condidate active sets
        candidate_active_sets = []

        # cross each non-redundant facet of the parent CR
        for coincident_facets in minimal_coincident_facets:

            # add or remove each constraint crossed to the active set of the parent CR
            candidate_active_set = set(active_set).symmetric_difference(set(coincident_facets))
            candidate_active_sets.append([sorted(list(candidate_active_set))])

        return candidate_active_sets

    @staticmethod
    def expand_candidate_active_sets(candidate_active_sets, weakly_active_constraints):
        """
        Expands the candidate active sets if there are some weakly active contraints (Theorem 5).

        INPUTS:
            candidate_active_sets: list of the candidate active sets for each minimal facet
            weakly_active_constraints: list of weakly active constraints (in the "original enumeration")

        OUTPUTS:
            candidate_active_sets: list of the candidate active sets for each minimal facet
        """

        # determine every possible combination of the weakly active contraints
        wac_combinations = []
        for n in range(1, len(weakly_active_constraints)+1):
            wac_combinations_n = itertools.combinations(weakly_active_constraints, n)
            wac_combinations += [list(c) for c in wac_combinations_n]

        # for each minimal facet of the CR add or remove each combination of wakly active constraints
        for i in range(0, len(candidate_active_sets)):
            active_set = candidate_active_sets[i][0]
            for combination in wac_combinations:
                further_active_set = set(active_set).symmetric_difference(combination)
                candidate_active_sets[i].append(sorted(list(further_active_set)))

        return candidate_active_sets

    def z_optimal(self, x):
        """
        Returns the explicit solution of the mpQP as a function of the parameter.

        INPUTS:
            x: value of the parameter

        OUTPUTS:
            z_optimal: solution of the QP
        """

        z_optimal = self.z_offset + self.z_linear.dot(x).reshape(self.z_offset.shape)
        return z_optimal

    def lambda_optimal(self, x):
        """
        Returns the explicit value of the multipliers of the mpQP as a function of the parameter.

        INPUTS:
            x: value of the parameter

        OUTPUTS:
            lambda_optimal: optimal multipliers
        """

        lambda_A_optimal = self.lambda_A_offset + self.lambda_A_linear.dot(x)
        lambda_optimal = np.zeros(len(self.active_set + self.inactive_set))
        for i in range(0, len(self.active_set)):
            lambda_optimal[self.active_set[i]] = lambda_A_optimal[i]
        return lambda_optimal

    def applies_to(self, x):
        """
        Determines is a given point belongs to the critical region.

        INPUTS:
            x: value of the parameter

        OUTPUTS:
            is_inside: flag (True if x is in the CR, False otherwise)
        """

        # check if x is inside the polytope
        is_inside = self.polytope.applies_to(x)

        return is_inside
