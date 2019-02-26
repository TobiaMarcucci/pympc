# external imports
import numpy as np
import gurobipy as grb
from copy import copy, deepcopy
from operator import le, ge, eq

# internal inputs
from pympc.control.hybrid_benchmark.branch_and_bound_with_warm_start import Node, branch_and_bound, best_first

class GurobiModel(grb.Model):
    '''
    Class which inherits from gurobi.Model.
    It facilitates the process of adding (retrieving) multiple variables and
    constraints to (from) the optimization problem.
    '''

    def __init__(self, **kwargs):

        # inherit from gurobi.Model
        super(GurobiModel, self).__init__(**kwargs)

    def add_variables(self, n, lb=None, **kwargs):
        '''
        Adds n optimization variables to the problem.
        It stores the new variables in a numpy array so that they can be readily used for computations.

        Arguments
        ---------
        n : int
            Number of optimization variables to be added to the problem.
        lb : list of floats
            Lower bounds for the optimization variables.
            This is set by default to -inf.
            Note that Gurobi by default would set this to zero.

        Returns
        -------
        x : np.array
            Numpy array that collects the new optimization variables.
        '''

        # change the default lower bound to -inf
        if lb is None:
            lb = [-grb.GRB.INFINITY]*n

        # add variables to the optimization problem
        x = self.addVars(n, lb=lb, **kwargs)

        # update model to make the new variables visible
        # this can inefficient but prevents headaches!
        self.update()

        # organize new variables in a numpy array
        x = np.array(x.values())
        
        return x

    def get_variables(self, name):
        '''
        Gets a set of variables from the problem and returns them in a numpy array.

        Arguments
        ---------
        name : string
            Name of the family of variables we want to get from the problem.

        Returns
        -------
        x : np.array
            Numpy array that collects the asked variables.

        '''

        # initilize vector of variables
        x = np.array([])

        # there cannnot be more x than optimization variables
        for i in range(self.NumVars):

            # get new element and append
            xi = self.getVarByName(name+'[%d]'%i)
            if xi:
                x = np.append(x, xi)

            # if no more elements are available break the for loop
            else:
                break

        return x

    def add_linear_constraints(self, x, operator, y, **kwargs):
        '''
        Adds a linear constraint of the form x (<=, ==, or >=) y to the optimization problem.

        Arguments
        ---------
        x : np.array of floats, gurobi.Var, or gurobi.LinExpr
            Left hand side of the constraint.
        operator : python operator
            Either le (less than or equal to), ge (greater than or equal to), or eq (equal to)
        y : np.array of floats, gurobi.Var, or gurobi.LinExpr
            Right hand side of the constraint.

        Returns
        -------
        c : np.array of gurobi.Constr
            Numpy array that collects the new constraints.
        '''

        # check that the size of the lhs and the rhs match
        assert len(x) == len(y)

        # add linear constraints to the problem
        c = self.addConstrs((operator(x[k], y[k]) for k in range(len(x))), **kwargs)

        # update model to make the new variables visible
        # this can inefficient but prevents headaches!
        self.update()

        # organize the constraints in a numpy array
        c = np.array(c.values())

        return c

    def get_constraints(self, name):
        '''
        Gets a set of constraints from the problem and returns them in a numpy array.

        Arguments
        ---------
        name : string
            Name of the family of constraints we want to get from the problem.

        Returns
        -------
        c : np.array
            Numpy array that collects the asked constraints.
        '''

        # initilize vector of constraints
        c = np.array([])

        # there cannnot be more c than constraints in the problem
        for i in range(self.NumConstrs):

            # get new constraint and append
            ci = self.getConstrByName(name+'[%d]'%i)
            if ci:
                c = np.append(c, ci)

            # if no more constraints are available break the for loop
            else:
                break

        return c

class HybridModelPredictiveController(object):
    '''
    Optimal controller for Mixed Logical Dynamical (mld) systems.
    Solves the mixed-integer quadratic optimization problem:
    min ||C x_N + c||_P^2 + sum_{t=0}^{N-1} (||C x_t + c||_Q^2 + ||D u_t + d||_R^2)
    s.t. x_0 given
         x_{t+1} = A x_t + Buc uc_t + Bub ub_t + Bsc sc_t + Bsb sb_t + b, t = 0, 1, ..., N-1,
         F x_t + Guc uc_t + Gub ub_t + Gsc sc_t + Gsb sb_t <= g, t = 0, 1, ..., N-1,
         ub_t binary, t = 0, 1, ..., N-1,
         sb_t binary, t = 0, 1, ..., N-1.
    '''

    def __init__(self, mld, N, weight_matrices, performance_output=None):
        '''
        Instantiates the hybrid MPC controller.

        Arguments
        ---------
        mld : instance of MixedLogicalDynamicalSystem
            System to be controlled.
        N : int
            Horizon of the controller.
        weight_matrices : dict
            Dictionary containing the weight matrices that penalize the state and the input.
            Entries are: 'P' terminal cost matrix, 'Q' stage cost matrix for the state, 'R' stage cost matrix for the input.
        performance_output : dict
            Dictionary containing the matrices that select the vectors to be minimized.
            Entries are: 'C' and 'c' for the state, 'D' and 'd' for the input.
            If None: C=I, c=0, D=I, d=0.
        '''

        # set default state and input selection
        if performance_output is None:
            performance_output = {
                'C': np.eye(mld.nx),
                'c': np.zeros(mld.nx),
                'D': np.eye(mld.nuc + mld.nub),
                'd': np.zeros(mld.nuc + mld.nub)
                }

        # store inputs
        self.mld = mld
        self.N = N
        [self.P, self.Q, self.R] = [weight_matrices['P'], weight_matrices['Q'], weight_matrices['R']]
        [self.C, self.c] = [performance_output['C'], performance_output['c']]
        [self.D, self.d] = [performance_output['D'], performance_output['d']]

        # build mixed integer program 
        self._check_inputs()
        self.model = self._build_mip()

    def _check_inputs(self, tol=1.e-5):
        '''
        Checks that the matrices passed as inputs in the initialization of the class have the right properties.

        Arguments
        ---------
        tol : float
            Numerical tolerance to check if the wight matrices are positive definite.
        '''

        # weight matrices
        assert self.P.shape[0] == self.P.shape[1]
        assert self.Q.shape[0] == self.Q.shape[1]
        assert self.R.shape[0] == self.R.shape[1]
        assert np.all(np.linalg.eigvals(self.P) > tol)
        assert np.all(np.linalg.eigvals(self.Q) > tol)
        assert np.all(np.linalg.eigvals(self.R) > tol)

        # state selection matrix
        assert self.Q.shape[0] == self.C.shape[0]
        assert self.C.shape[1] == self.mld.nx
        assert self.c.size == self.Q.shape[0]

        # input selection matrix
        assert self.R.shape[0] == self.D.shape[0]
        assert self.D.shape[1] == self.mld.nuc + self.mld.nub
        assert self.d.size == self.R.shape[0]

    def _build_mip(self):
        '''
        Builds the guorbi model for the opitmization problem to be solved.

        Returns
        -------
        model : GurobiModel
            Gurobi model of the mathematical program.
        '''

        # initialize program
        model = GurobiModel()
        obj = 0.

        # initial state (initialized to zero)
        x_next = model.add_variables(self.mld.nx, name='x_0')
        model.add_linear_constraints(x_next, eq, [0.]*self.mld.nx, name='alpha_0')

        # loop over the time horizon
        for t in range(self.N):

            # stage variables
            x = x_next
            uc = model.add_variables(self.mld.nuc, name='uc_%d'%t)
            ub = model.add_variables(self.mld.nub, name='ub_%d'%t)
            sc = model.add_variables(self.mld.nsc, name='sc_%d'%t)
            sb = model.add_variables(self.mld.nsb, name='sb_%d'%t)
            x_next = model.add_variables(self.mld.nx, name='x_%d'%(t+1))

            # bounds on the binaries
            # inequalities must be stated as expr <= num to get negative duals
            # note that num <= expr would be modified to expr => num
            # and would give positive duals
            model.add_linear_constraints(-ub, le, [0.]*self.mld.nub, name='lbu_%d'%t)
            model.add_linear_constraints(-sb, le, [0.]*self.mld.nsb, name='lbs_%d'%t)
            model.add_linear_constraints(ub, le, [1.]*self.mld.nub, name='ubu_%d'%t)
            model.add_linear_constraints(sb, le, [1.]*self.mld.nsb, name='ubs_%d'%t)

            # mld dynamics
            model.add_linear_constraints(
                x_next,
                eq,
                self.mld.A.dot(x) +
                self.mld.Buc.dot(uc) + self.mld.Bub.dot(ub) +
                self.mld.Bsc.dot(sc) + self.mld.Bsb.dot(sb) +
                self.mld.b,
                name='alpha_%d'%(t+1)
                )

            # mld constraints
            model.add_linear_constraints(
                self.mld.F.dot(x) +
                self.mld.Guc.dot(uc) + self.mld.Gub.dot(ub) +
                self.mld.Gsc.dot(sc) + self.mld.Gsb.dot(sb),
                le,
                self.mld.g,
                name='beta_%d'%t
                )

            # stage cost
            u = np.concatenate((uc, ub))
            y = model.add_variables(self.C.shape[0], name='y_%d'%t)
            v = model.add_variables(self.D.shape[0], name='v_%d'%t)
            model.add_linear_constraints(y, eq, self.C.dot(x) + self.c, name='gamma_%d'%t)
            model.add_linear_constraints(v, eq, self.D.dot(u) + self.d, name='delta_%d'%t)
            obj += y.dot(self.Q).dot(y) + v.dot(self.R).dot(v)

        # terminal cost
        y = model.add_variables(self.C.shape[0], name='y_%d'%self.N)
        model.add_linear_constraints(y, eq, self.C.dot(x_next) + self.c, name='gamma_%d'%self.N)
        obj += y.dot(self.P).dot(y)

        # set cost
        model.setObjective(obj)

        return model

    def set_initial_condition(self, x0):
        '''
        Sets the initial state in the model to be equal to x0.

        Arguments
        ---------
        x0 : numpy.array
            Initial conditions for the optimization problem.
        '''

        # check size of x0
        assert self.mld.nx == x0.size

        # get the equality constraint for the initial condtions
        # (named as its Lagrange multiplier)
        alpha_0 = self.model.get_constraints('alpha_0')

        # set initial conditions
        for k, xk in enumerate(x0):
            alpha_0[k].RHS = xk

        # update gurobi model to be safe
        self.model.update()

    def set_bounds_binaries(self, identifier):
        '''
        Sets the lower and upper bounds of the binary optimization variables
        in the problem to the values passed in the identifier.
        An identifier is a dictionary with tuples as keys.
        A key is in the form, e.g., ('u', 22, 4) where:
        - 'u' denotes in this case binary inputs (s' for binary auxiliary variables),
        - 22 is the time step,
        - 4 denotes the 4th element of the vector.

        Arguments
        ---------
        identifier : dict
            Dictionary containing the values for some of the binaries.
            Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.
        '''

        # loop over the time horizon
        for t in range(self.N):

            # fix the bounds for the binary inputs
            lbu = self.model.get_constraints('lbu_%d'%t)
            ubu = self.model.get_constraints('ubu_%d'%t)
            for k in range(self.mld.nub):
                if identifier.has_key(('u', t, k)):
                    # the minus is because constraints are stated as -u <= -u_lb
                    lbu[k].RHS = - identifier[('u', t, k)]
                    ubu[k].RHS = identifier[('u', t, k)]
                else:
                    lbu[k].RHS = 0.
                    ubu[k].RHS = 1.

            # fix the bounds for the binary auxiliary variables
            lbs = self.model.get_constraints('lbs_%d'%t)
            ubs = self.model.get_constraints('ubs_%d'%t)
            for k in range(self.mld.nsb):
                if identifier.has_key(('s', t, k)):
                    # the minus is because constraints are stated as -s <= -s_lb
                    lbs[k].RHS = - identifier[('s', t, k)]
                    ubs[k].RHS = identifier[('s', t, k)]
                else:
                    lbs[k].RHS = 0.
                    ubs[k].RHS = 1.

        # update gurobi model to be safe
        self.model.update()

    def set_type_ub_sb(self, var_type):
        '''
        Sets the type of the variables ub and sb in the optimization problem.

        Arguments
        ---------
        var_type : string
            Sting containing the type of ub and sb.
            'C' for continuous ub and sb, and 'D' for binary ub and sb.
        '''

        # loop over the time horizon
        for t in range(self.N):

            # fix the type for ub
            ub = self.model.get_variables('ub_%d'%t)
            for k in range(self.mld.nub):
                ub[k].VType = var_type

            # fix the type for sb
            sb = self.model.get_variables('sb_%d'%t)
            for k in range(self.mld.nsb):
                sb[k].VType = var_type

        # update gurobi model to be safe
        self.model.update()

    def solve_subproblem(self, x0, identifier, tol=5.e-4, tol_farkas_objective=1.e-6):
        '''
        Solves the QP relaxation uniquely indentified by its identifier for the given initial state.

        Arguments
        ---------
        x0 : np.array
            Initial state of the system.
        identifier : dict
            Dictionary containing the values for some of the binaries.
            Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.
        tol : float
            Numeric tolerance in the checks.

        Returns
        -------
        feasible : bool
            True if the subproblem is feasible, False otherwise.
        integer_feasible : bool
            True if the solution of the subproblem is integer feasible, False otherwise.
        objective : float or None
            Optimal objective for the subproblem, None if infeasible.
        sol : dict
            Dictionary with the primal and dual solution of the subproblem.
        '''

        # reset model (gurobi does not try to use the last solve to warm start)
        self.model.reset()

        # set up miqp
        self.set_type_ub_sb('C')
        self.set_initial_condition(x0)
        self.set_bounds_binaries(identifier)

        # gurobi parameters
        self.model.setParam('OutputFlag', 0)

        # run the optimization and structure the result
        self.model.optimize()
        sol = {}

        # if optimal
        if self.model.status == 2:
            feasible = True
            objective = self.model.objVal
            sol['primal'] = self._get_primal_solution()
            sol['dual'] = self._get_dual_solution(feasible)

        # if infeasible, infeasible_or_unbounded, numeric_errors, suboptimal
        elif self.model.status in [3, 4, 12, 13]:
            self._do_farkas_proof()
            feasible = False
            objective = None
            sol['farkas_proof'] = self._get_dual_solution(feasible)
            sol['farkas_objective'] = self._evaluate_dual_solution(x0, identifier, sol['farkas_proof'])

        # if none of the previous raise error
        else:
            raise ValueError('unknown model status %d.' % self.model.status)

        # integer feasibility
        n_binaries = self.N * (self.mld.nub + self.mld.nsb)
        integer_feasible = feasible and len(identifier) == n_binaries

        # check that dual solution
        # self._check_dual_solution(x0, identifier, feasible, objective, sol, tol, tol_farkas_objective)

        return feasible, integer_feasible, objective, sol

    def _get_primal_solution(self):
        '''
        Organizes the primal solution of the convex subproblem.

        Returns
        -------
        primal : dict
            Dictionary containing the primal solution of the convex subproblem.
            Keys are 'x', 'uc', 'ub', 'sc', 'sb', 'y', 'v'.
            Each one of these is a list (ordered in time) of numpy arrays.
        '''

        # initialize primal solution
        primal = {}

        # primal stage variables
        for l in ['x', 'uc', 'ub', 'sc', 'sb', 'y', 'v']:
            primal[l] = []
            for t in range(self.N):
                v = self.model.get_variables('%s_%d'%(l,t))
                primal[l].append(np.array([vi.x for vi in v]))

        # primal terminal variables
        for l in ['x', 'y']:
            v = self.model.get_variables('%s_%d'%(l,self.N))
            primal[l].append(np.array([vi.x for vi in v]))

        return primal

    def _get_dual_solution(self, feasible):
        '''
        Organizes the dual solution of the convex subproblem.

        Returns
        -------
        dual : dict
            Dictionary containing the dual solution of the convex subproblem.
            Keys are 'alpha', 'beta', 'gamma', 'delta', 'lbu', 'ubu', 'lbs', 'ubs'.
            Each one of these is a list (ordered in time) of numpy arrays.
        '''

        # dual stage variables
        dual = {}
        for l in ['alpha', 'beta', 'gamma', 'delta', 'lbu', 'ubu', 'lbs', 'ubs']:
            dual[l] = []
            for t in range(self.N):
                c = self.model.get_constraints('%s_%d'%(l,t))
                if feasible:
                    # gurobi gives negative multipliers and positive farkas duals!
                    dual[l].append(- np.array([ci.Pi for ci in c]))
                else:
                    dual[l].append(np.array([ci.FarkasDual for ci in c]))

        # dual terminal variables
        for l in ['alpha', 'gamma']:
            c = self.model.get_constraints('%s_%d'%(l,self.N))
            if feasible:
                dual[l].append(- np.array([ci.Pi for ci in c]))
            else:
                dual[l].append(np.array([ci.FarkasDual for ci in c]))

        return dual

    def _do_farkas_proof(self):
        '''
        Performes the Farkas proof of infeasibility for the subproblem.
        It momentarily sets the objective to zero because gurobi can do the farkas proof only for linear programs.
        This can be very slow.
        '''

        # copy objective
        obj = self.model.getObjective()

        # rerun the optimization with linear objective
        # (only linear accepted for farkas proof)
        self.model.setParam('InfUnbdInfo', 1)
        self.model.setObjective(0.)
        self.model.optimize()

        # ensure new problem is actually infeasible
        assert self.model.status == 3

        # reset quadratic objective
        self.model.setObjective(obj)

    def _check_dual_solution(self, x0, identifier, feasible, objective, sol, tol, tol_farkas_objective):
        '''
        Checks that the dual variables given by gurobi are correct.
        Mostly useful for debugging the signs of the multipliers.

        Arguments
        ---------
        x0 : np.array
            Initial state of the system.
        identifier : dict
            Dictionary containing the values for some of the binaries.
            Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.
        sol : dict
            Dictionary with the primal and dual solution of the subproblem.
        tol : float
            Numeric tolerance in the checks.
        '''

        # if the optimization problem was primal feasible
        if feasible:
            self._check_dual_feasibility(sol['dual'], tol)
            dual_obj = self._evaluate_dual_solution(x0, identifier, sol['dual'])
            assert np.isclose(objective, dual_obj)

        # if the optimization problem was primal infeasible
        else:
            self._check_dual_feasibility(sol['farkas_proof'], tol)
            assert sol['farkas_objective'] > tol_farkas_objective
            assert np.linalg.norm(np.concatenate(sol['farkas_proof']['gamma'])) < tol
            assert np.linalg.norm(np.concatenate(sol['farkas_proof']['delta'])) < tol
            

    def _check_dual_feasibility(self, dual, tol):
        '''
        Checks that the given dual solution is feasible.

        Arguments
        ---------
        dual : dict
            Dictionary containing the dual solution of the convex subproblem.
            Keys are 'alpha', 'beta', 'gamma', 'delta', 'lbu', 'ubu', 'lbs', 'ubs'.
            Each one of these is a list (ordered in time) of numpy arrays.
        tol : float
            Numeric tolerance in the checks.
        '''

        # check inequalities
        for c in ['beta', 'lbu', 'ubu', 'lbs', 'ubs']:
            multipliers = np.concatenate(dual[c])
            if multipliers.size > 0:
                assert np.min(multipliers) > - tol

        # check stationarity wrt x_t for t = 0, ..., N-1
        for t in range(self.N):
            residuals = dual['alpha'][t] - \
                self.mld.A.T.dot(dual['alpha'][t+1]) + \
                self.mld.F.T.dot(dual['beta'][t]) - \
                self.C.T.dot(dual['gamma'][t])
            assert np.linalg.norm(residuals) < tol

        # check stationarity wrt x_N
        residuals = dual['alpha'][self.N] - self.C.T.dot(dual['gamma'][self.N])
        assert np.linalg.norm(residuals) < tol

        # test stationarity wrt u_t for all t
        for t in range(self.N):
            residuals = - self.mld.Bu.T.dot(dual['alpha'][t+1]) + \
                self.mld.Gu.T.dot(dual['beta'][t]) - \
                self.D.T.dot(dual['delta'][t]) + \
                self.mld.Wu.T.dot(dual['ubu'][t] - dual['lbu'][t])
            assert np.linalg.norm(residuals) < tol

        # test stationarity wrt s_t for all t
        for t in range(self.N):
            residuals = - self.mld.Bs.T.dot(dual['alpha'][t+1]) + \
                self.mld.Gs.T.dot(dual['beta'][t]) + \
                self.mld.Ws.T.dot(dual['ubs'][t] - dual['lbs'][t])
            assert np.linalg.norm(residuals) < tol

    def _evaluate_dual_solution(self, x0, identifier, dual):
        '''
        Given a dual solution, returns it cost.

        Arguments
        ---------
        x0 : np.array
            Initial state of the system.
        identifier : dict
            Dictionary containing the values for some of the binaries.
            Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.
        dual : dict
            Dictionary containing the dual solution of the convex subproblem.
            Keys are 'alpha', 'beta', 'gamma', 'delta', 'lbu', 'ubu', 'lbs', 'ubs'.
            Each one of these is a list (ordered in time) of numpy arrays.

        Returns
        -------
        obj : float
            Dual objective associated with the given dual solution.
        '''

        # evaluate quadratic terms
        Qinv = np.linalg.inv(self.Q)
        Rinv = np.linalg.inv(self.R)
        Pinv = np.linalg.inv(self.P)
        obj = - .25 * dual['gamma'][self.N].dot(Pinv).dot(dual['gamma'][self.N])
        for t in range(self.N):
            obj -= .25 * dual['gamma'][t].dot(Qinv).dot(dual['gamma'][t])
            obj -= .25 * dual['delta'][t].dot(Rinv).dot(dual['delta'][t])

        # evaluate linear terms
        w_lb, w_ub, r_lb, r_ub = self._get_bounds_on_binaries(identifier)
        obj -= x0.dot(dual['alpha'][0])
        obj -= self.c.dot(dual['gamma'][self.N])
        for t in range(self.N):
            obj -= self.mld.b.dot(dual['alpha'][t+1])
            obj -= self.mld.g.dot(dual['beta'][t])
            obj -= self.c.dot(dual['gamma'][t])
            obj -= self.d.dot(dual['delta'][t])
            obj += w_lb[t].dot(dual['lbu'][t])
            obj += r_lb[t].dot(dual['lbs'][t])
            obj -= w_ub[t].dot(dual['ubu'][t])
            obj -= r_ub[t].dot(dual['ubs'][t])

        return obj

    def _get_bounds_on_binaries(self, identifier):
        '''
        Restates the identifier in terms of lower an upper bounds on the binary variables in the problem.

        Arguments
        ---------
        identifier : dict
            Dictionary containing the values for some of the binaries.
            Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.

        Returns
        -------
        w_lb : list of numpy arrays
            Lower bound imposed by the identifier on the binary inputs in the problem.
        w_ub : list of numpy arrays
            Upper bound imposed by the identifier on the binary inputs in the problem.
        r_lb : list of numpy arrays
            Lower bound imposed by the identifier on the binary auxiliary variables in the problem.
        r_ub : list of numpy arrays
            Upper bound imposed by the identifier on the binary auxiliary variables in the problem.
        '''

        # initialize bounds on the binary inputs
        w_lb = [np.zeros(self.mld.nub) for t in range(self.N)]
        w_ub = [np.ones(self.mld.nub) for t in range(self.N)]

        # initialize bounds on the binary auxiliary variables
        r_lb = [np.zeros(self.mld.nsb) for t in range(self.N)]
        r_ub = [np.ones(self.mld.nsb) for t in range(self.N)]

        # parse identifier
        for key, val in identifier.items():
            if key[0] == 'u':
                w_lb[key[1]][key[2]] = val
                w_ub[key[1]][key[2]] = val
            elif key[0] == 's':
                r_lb[key[1]][key[2]] = val
                r_ub[key[1]][key[2]] = val

        return w_lb, w_ub, r_lb, r_ub

    def feedforward(self, x0, **kwargs):
        '''
        Solves the mixed integer program using the branch_and_bound function.

        Arguments
        ---------
        x0 : np.array
            Initial state of the system.

        Returns
        -------
        sol : dict
            Solution associated with the incumbent node at optimality.
            (See documentation of the method solve_subproblem.)
            None if problem is infeasible.
        optimal_leaves : list of instances of Node
            Leaves of the branch and bound tree that proved optimality of the returned solution. 
        '''

        # generate a solver for the branch and bound algorithm
        def solver(identifier):
            return self.solve_subproblem(x0, identifier)

        # solve the mixed integer program using branch and bound
        return branch_and_bound(
            solver,
            best_first,
            self.explore_in_chronological_order,
            **kwargs
        )

    def explore_in_chronological_order(self, identifier):
        '''
        Heuristic search for the branch and bound algorithm.

        Arguments
        ---------
        identifier : dict
            Dictionary containing the values for some of the binaries.
            Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.

        Returns
        -------
        branches : list of dict
            List of sub-identifier that, if merged with the identifier of the parent, give the identifier of the children.
        '''

        # idices of the last binary fixed in time
        t = max([0] + [k[1] for k in identifier.keys()])
        index_u = max([0] + [k[2]+1 for k in identifier.keys() if k[:2] == ('u',t)])
        index_s = max([0] + [k[2]+1 for k in identifier.keys() if k[:2] == ('s',t)])

        # try to fix one more ub at time t
        if index_u < self.mld.nub:
            branches = [{('u',t,index_u): 0.}, {('u',t,index_u): 1.}]

        # try to fix one more sb at time t
        elif index_s < self.mld.nsb:
            branches = [{('s',t,index_s): 0.}, {('s',t,index_s): 1.}]

        # if everything is fixed at time t, move to time t+1
        else:
            if self.mld.nub > 0:
                branches = [{('u',t+1,0): 0.}, {('u',t+1,0): 1.}]
            else:
                branches = [{('s',t+1,0): 0.}, {('s',t+1,0): 1.}]

        return branches

    def construct_warm_start(self, leaves, stage_variables):
        '''
        stage_variables are x0, e0, uc0, ub0, sc0, sb0
        '''

        # needed for a (redundant) check
        x_1 = self.mld.A.dot(stage_variables['x_0']) + \
            self.mld.Buc.dot(stage_variables['uc_0']) + \
            self.mld.Bub.dot(stage_variables['ub_0']) + \
            self.mld.Bsc.dot(stage_variables['sc_0']) + \
            self.mld.Bsb.dot(stage_variables['sb_0']) + \
            stage_variables['e_0']

        # initialize nodes for warm start
        warm_start = []

        # check each on of the optimal leaves
        for l in leaves:

            # it the identifier of the leaf does not agree with the stage_variables drop the leaf
            if self._retain_leaf(l.identifier, stage_variables):
                identifier = self._get_new_identifier(l.identifier)
                extra_data = {}

                # propagate lower bounds if leaf is feasible
                if l.feasible is None or l.feasible:
                    feasible = None
                    lam = self._get_lambdas(l.identifier, l.extra_data['dual'], stage_variables)
                    lower_bound = l.objective + sum(lam)
                    extra_data['objective_dual'] = lower_bound
                    extra_data['dual'] = self._propagate_dual_solution(l.extra_data['dual'])

                    # double check that the whole lambda thing is correct
                    assert np.isclose(
                        extra_data['objective_dual'],
                        self._evaluate_dual_solution(x_1, identifier, extra_data['dual'])
                        )
                else:

                    # propagate infeasibility if leaf is still infeasible
                    feasible = False
                    lam = self._get_lambdas(l.identifier, l.extra_data['farkas_proof'], stage_variables)
                    if stage_variables['e_0'].dot(l.extra_data['farkas_proof']['alpha'][1]) < l.extra_data['farkas_objective'] + lam[2]:
                        lower_bound = np.inf
                        extra_data['farkas_objective'] = l.extra_data['farkas_objective'] + lam[2] + lam[4]
                        extra_data['farkas_proof'] = self._propagate_dual_solution(l.extra_data['farkas_proof'])

                        # double check that the whole lambda thing is correct
                        assert np.isclose(
                            extra_data['farkas_objective'],
                            self._evaluate_dual_solution(x_1, identifier, extra_data['farkas_proof'])
                            )

                    # if potentially feasible
                    else:
                        lower_bound = - np.inf

                # add new node to the list for the warm start
                warm_start.append(Node(None, identifier, feasible, lower_bound, extra_data))

        return warm_start

    @staticmethod
    def _retain_leaf(identifier, stage_variables):
        '''
        '''

        # retain until proven otherwise
        retain = True

        # loop over the elements of the identifier and check if they agree with stage variables at time zero
        for key, value in identifier.items():

            # it the element of the identifier is associated with time zero
            if key[1] == 0:

                # check if the value of the variable agrees with the one forced by the identifier
                for v in ['u', 's']:
                    if key[0] == v and not np.isclose(value, stage_variables[v+'b_0'][key[2]]):
                        retain = False
                        break

        return retain

    @staticmethod
    def _propagate_dual_solution(dual):
        '''
        '''

        # copy the old solution
        new_dual = deepcopy(dual)

        # for all the dual variables
        for l in ['alpha', 'beta', 'gamma', 'delta', 'lbu', 'ubu', 'lbs', 'ubs']:

            # add a zero multiplier at time N+1
            new_dual[l].append(0.*new_dual[l][0])

            # delete the multiplier for time zero
            del new_dual[l][0]

        return new_dual

    def _get_lambdas(self, identifier, dual, stage_variables, tol=1.e-5):

        # shortcuts
        Pinv = np.linalg.inv(self.P)
        Qinv = np.linalg.inv(self.Q)
        Rinv = np.linalg.inv(self.R)
        x_0 = stage_variables['x_0']
        e_0 = stage_variables['e_0']
        u_0 = np.concatenate((stage_variables['uc_0'], stage_variables['ub_0']))
        s_0 = np.concatenate((stage_variables['sc_0'], stage_variables['sb_0']))

        # lambda 1
        y_0 = self.C.dot(x_0) + self.c
        v_0 = self.D.dot(u_0) + self.d
        lam1 = - y_0.dot(self.Q).dot(y_0) - v_0.dot(self.R).dot(v_0)

        # lambda 2
        lam2 = .25 * dual['gamma'][self.N].dot(Pinv - Qinv).dot(dual['gamma'][self.N])

        # lambda 3
        w_lb, w_ub, r_lb, r_ub = self._get_bounds_on_binaries(identifier)
        lam3 = - (self.mld.F.dot(x_0) + self.mld.Gu.dot(u_0) + self.mld.Gs.dot(s_0) - self.mld.g).dot(dual['beta'][0])
        lam3 -= (w_lb[0] - self.mld.Wu.dot(u_0)).dot(dual['lbu'][0])
        lam3 -= (r_lb[0] - self.mld.Ws.dot(s_0)).dot(dual['lbs'][0])
        lam3 -= (self.mld.Wu.dot(u_0) - w_ub[0]).dot(dual['ubu'][0])
        lam3 -= (self.mld.Ws.dot(s_0) - r_ub[0]).dot(dual['ubs'][0])
        assert lam3 > - tol

        # lambda 4
        lam4_x = .5*dual['gamma'][0] + self.Q.dot(y_0)
        lam4_u = .5*dual['delta'][0] + self.R.dot(v_0)
        lam4 = lam4_x.dot(Qinv).dot(lam4_x) + lam4_u.dot(Rinv).dot(lam4_u)

        # lambda 5
        lam5 = - e_0.dot(dual['alpha'][1])

        return lam1, lam2, lam3, lam4, lam5

    @staticmethod
    def _get_new_identifier(identifier):
        new_identifier = {}
        for k, v in identifier.items():
            if k[1] > 0:
                new_identifier[(k[0],k[1]-1,k[2])] = v
        return new_identifier

    # def feedforward_gurobi(self, x0):

    #     # reset program
    #     self.prog.reset()

    #     # set up miqp
    #     self.reset_bounds_binaries()
    #     self.set_type_binaries('B')
    #     self.set_initial_condition(x0)

    #     # parameters
    #     self.prog.setParam('OutputFlag', 1)

    #     # run the optimization
    #     self.prog.optimize()
    #     sol = self._get_subproblem_solution()

    #     return sol

    # def feedback_gurobi(self, x0):
    #     u_feedforward = self.feedforward_gurobi(x0)[0]
    #     if u_feedforward is None:
    #         return None
    #     return u_feedforward[0]

    # def _get_subproblem_solution(self):

    #     sol = {
    #     'x': None,
    #     'uc': None,
    #     'ub': None,
    #     'sc': None,
    #     'sb': None,
    #     'objective': None
    #     }

    #     # primal solution
    #     if self.prog.status in [2, 9, 11] and self.prog.SolCount > 0: # optimal, interrupted, time limit

    #         # sol['x'] = [np.array([self.prog.getVarByName('x_%d[%d]'%(t,k)).x for k in range(self.mld.nx)]) for t in range(self.N+1)]
    #         # sol['uc'] = [np.array([self.prog.getVarByName('uc_%d[%d]'%(t,k)).x for k in range(self.mld.nuc)]) for t in range(self.N)]
    #         # sol['ub'] = [np.array([self.prog.getVarByName('ub_%d[%d]'%(t,k)).x for k in range(self.mld.nub)]) for t in range(self.N)]
    #         # sol['sc'] = [np.array([self.prog.getVarByName('sc_%d[%d]'%(t,k)).x for k in range(self.mld.nsc)]) for t in range(self.N)]
    #         # sol['sb'] = [np.array([self.prog.getVarByName('sb_%d[%d]'%(t,k)).x for k in range(self.mld.nsb)]) for t in range(self.N)]
    #         # sol['objective'] = self.prog.objVal

    #         # primal solution
    #         sol['primal'] = {}

    #         for k, v in self.variables.items():
    #             sol['primal'][k] = np.array([vk.x for vk in v])

    #         # dual solution
    #         sol['dual'] = {}
    #         for k, v in self.constraints.items():
    #             sol['dual'][k] = np.array([vk.Pi for vk in v])

    #     return sol

    # def propagate_bounds(self, feasible_leaves, x0, u0):
    #     delta = .5 * (x0.dot(self.Q.dot(x0)) + u0.dot(self.R.dot(u0)))
    #     warm_start = []
    #     for leaf in feasible_leaves:
    #         identifier = self.get_new_identifier(leaf.identifier)
    #         lower_bound = leaf.lower_bound - delta
    #         warm_start.append(Node(None, identifier, lower_bound))
    #     return warm_start

    # def get_new_identifier(self, old_id):
    #     new_id = copy(old_id)
    #     for t in range(self.N):
    #         for i in range(self.S.nm):
    #             if new_id.has_key((t, i)):
    #                 if t == 0:
    #                     new_id.pop((t, i))
    #                 else:
    #                     new_id[(t-1, i)] = new_id.pop((t, i))
    #     return new_id