# external imports
import numpy as np

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.programs import linear_program
from pympc.optimization.solvers.branch_and_bound import Tree, draw_tree

# pydrake imports
from pydrake.all import MathematicalProgram, SolutionResult
from pydrake.solvers.gurobi import GurobiSolver
from pydrake.solvers.mosek import MosekSolver

class HybridModelPredictiveController(object):

    def __init__(self, S, N, Q, R, P, X_N, method='big_m'):

        # store inputs
        self.S = S
        self.N = N
        self.Q = Q
        self.R = R
        self.P = P
        self.X_N = X_N

        # mpMIQP
        self.build_mpmiqp(method) # adds the variables: prog, objective, u, x, d, initial_condition, binaries_lower_bound
        self.fixed_mode_sequence = []

        # get bigMs
        self._alpha, self._beta = self._get_bigM_dynamics()
        self._gamma = self._get_bigM_domains()

    def _get_bigM_dynamics(self):

        # initialize list of bigMs
        alpha = []
        beta = []

        # outer loop over the number of affine systems
        for i, S_i in enumerate(self.S.affine_systems):
            alpha_i = []
            beta_i = []
            A_i = np.hstack((S_i.A, S_i.B))

            # inner loop over the number of affine systems
            for j, S_j in enumerate(self.S.affine_systems):
                alpha_ij = []
                beta_ij = []
                D_j = self.S.domains[j]

                # solve two LPs for each component of the state vector
                for k in range(S_i.nx):
                    f = A_i[k:k+1,:].T
                    sol = linear_program(f, D_j.A, D_j.b, D_j.C, D_j.d)
                    alpha_ij.append(sol['min'] + S_i.c[k,0])
                    sol = linear_program(-f, D_j.A, D_j.b, D_j.C, D_j.d)
                    beta_ij.append(- sol['min'] + S_i.c[k,0])

                # close inner loop appending bigMs
                alpha_i.append(np.vstack(alpha_ij))
                beta_i.append(np.vstack(beta_ij))

            # close outer loop appending bigMs
            alpha.append(alpha_i)
            beta.append(beta_i)

        return alpha, beta

    def _get_bigM_domains(self):

        # initialize list of bigMs
        gamma = []

        # outer loop over the number of affine systems
        for i, D_i in enumerate(self.S.domains):
            gamma_i = []

            # inner loop over the number of affine systems
            for j, D_j in enumerate(self.S.domains):
                gamma_ij = []

                # solve one LP for each inequality of the ith domain
                for k in range(D_i.A.shape[0]):
                    f = -D_i.A[k:k+1,:].T
                    sol = linear_program(f, D_j.A, D_j.b, D_j.C, D_j.d)
                    gamma_ij.append(- sol['min'] - D_i.b[k,0])

                # close inner loop appending bigMs
                gamma_i.append(np.vstack(gamma_ij))

            # close outer loop appending bigMs
            gamma.append(gamma_i)

        return gamma

    def feedforward_bm(self, x0):

        # loose big m
        alpha = [min([self._alpha[i][j][k][0] for i in range(self.S.nm) for j in range(self.S.nm)]) for k in range(self.S.nx)]
        beta = [max([self._beta[i][j][k][0] for i in range(self.S.nm) for j in range(self.S.nm)]) for k in range(self.S.nx)]
        gamma = [[max([self._gamma[i][j][k][0] for j in range(self.S.nm)]) for k in range(self._gamma[i][0].shape[0])] for i in range(self.S.nm)]

        # initialize program and cost function
        prog = MathematicalProgram()
        obj = 0.

        # state and input variables
        u = [prog.NewContinuousVariables(self.S.nu) for t in range(self.N)]
        x = [prog.NewContinuousVariables(self.S.nx) for t in range(self.N+1)]

        # auxiliary variables
        z = [[prog.NewContinuousVariables(self.S.nx) for i in range(self.S.nm)] for t in range(self.N)]
        d = [[prog.NewBinaryVariables(1)[0] for i in range(self.S.nm)] for t in range(self.N)]

        # initial conditions
        for k in range(self.S.nx):
            prog.AddLinearConstraint(x[0][k] == x0[k])

        # loop over time
        for t in range(self.N):
            
            # add stage cost
            obj += .5*x[t].dot(self.Q).dot(x[t])
            obj += .5*u[t].dot(self.R).dot(u[t])
            
            # only one mode per time step
            prog.AddLinearConstraint(sum(d[t]) == 1.)
            
            # recompose the state
            for k in range(self.S.nx):
                prog.AddLinearConstraint(x[t+1][k] == sum(z[t])[k])
            
            # loop over number of modes of the PWA system
            for i in range(self.S.nm):
                
                # mixed-integer contraints to enforce the dynamics
                Ai = self.S.affine_systems[i].A
                Bi = self.S.affine_systems[i].B
                ci = self.S.affine_systems[i].c
                for k in range(self.S.nx):
                    prog.AddLinearConstraint(z[t][i][k] >= alpha[k] * d[t][i])
                    prog.AddLinearConstraint(z[t][i][k] <= beta[k] * d[t][i])
                    prog.AddLinearConstraint(z[t][i][k] <= Ai[k].dot(x[t]) + Bi[k].dot(u[t]) + ci[k] - alpha[k]*(1.-d[t][i]))
                    prog.AddLinearConstraint(z[t][i][k] >= Ai[k].dot(x[t]) + Bi[k].dot(u[t]) + ci[k] - beta[k]*(1.-d[t][i]))

                # mixed-integer contraints to enforce state and input constraints
                Ai = self.S.domains[i].A
                bi = self.S.domains[i].b
                for k in range(Ai.shape[0]):
                    prog.AddLinearConstraint(Ai[k].dot(np.concatenate((x[t], u[t]))) <= bi[k,0] + gamma[i][k]*(1.-d[t][i]))

        # terminal constraint
        for k in range(self.X_N.A.shape[0]):
            prog.AddLinearConstraint(self.X_N.A[k].dot(x[self.N]) <= self.X_N.b[k,0])

        # terminal cost
        obj += .5*x[self.N].dot(self.P).dot(x[self.N])
        prog.AddQuadraticCost(obj)

        # set solver
        #solver = MosekSolver()
        solver = GurobiSolver()
        prog.SetSolverOption(solver.solver_type(), 'OutputFlag', 1) # prints on the terminal!
        #prog.SetSolverOption(solver.solver_type(), 'Heuristics', 0.)

        # solve MIQP
        result = solver.Solve(prog)

        # solve and check feasibility
        if result != SolutionResult.kSolutionFound:
            return None, None, None, None

        # from vector to list of vectors
        u_list = [prog.GetSolution(u[t]).reshape(self.S.nu,1) for t in range(self.N)]
        x_list = [prog.GetSolution(x[t]).reshape(self.S.nx,1) for t in range(self.N+1)]
        d_list = [[prog.GetSolution(d[t][i]) for i in range(self.S.nm)] for t in range(self.N)]
        mode_sequence = [d_list[t].index(max(d_list[t])) for t in range(self.N)]
        cost  = .5 * sum([u.T.dot(self.R).dot(u) for u in u_list])
        cost += .5 * sum([x.T.dot(self.Q).dot(x) for x in x_list[:self.N]])
        cost += .5 * x_list[self.N].T.dot(self.P).dot(x_list[self.N])

        return u_list, x_list, mode_sequence, cost[0,0]

    def build_mpmiqp(self, method):

        # express the constrained dynamics as a list of polytopes in the (x,u,x+)-space
        P = get_graph_representation(self.S)

        # get big-Ms for some of the solution methods
        m, mi = get_big_m(P)

        # initialize program
        self.prog = MathematicalProgram()
        self.x = []
        self.u = []
        self.d = []
        obj = 0.
        self.binaries_lower_bound = []

        # auxiliary continuous variables (y auxiliary for x, v auxiliary for u, z auxiliary for x+)
        if method == 'convex_hull':
            y = []
            v = []
            z = []

        # initial conditions (set arbitrarily to zero in the building phase)
        self.x.append(self.prog.NewContinuousVariables(self.S.nx)) # x(0)
        self.initial_condition = []
        for k in range(self.S.nx):
            self.initial_condition.append(self.prog.AddLinearConstraint(self.x[0][k] == 0.).evaluator())

        # loop over time
        for t in range(self.N):

            # create variables
            self.u.append(self.prog.NewContinuousVariables(self.S.nu)) # u(t)
            self.d.append([self.prog.NewContinuousVariables(1)[0] for i in range(self.S.nm)]) # d_i(t)
            self.x.append(self.prog.NewContinuousVariables(self.S.nx)) # x(t+1)
            
            # auxiliary continuous variables for the convex-hull method
            if method == 'convex_hull':
                y.append([self.prog.NewContinuousVariables(self.S.nx) for i in range(self.S.nm)])
                v.append([self.prog.NewContinuousVariables(self.S.nu) for i in range(self.S.nm)])
                z.append([self.prog.NewContinuousVariables(self.S.nx) for i in range(self.S.nm)])

            # enforce constrained dynamics (big-m methods)
            if method in ['big_m', 'improved_big_m']:
                xux = np.concatenate((self.x[t], self.u[t], self.x[t+1]))
                for i in range(self.S.nm):
                    for k in range(P[i].A.shape[0]):
                        if method == 'big_m':
                            self.prog.AddLinearConstraint(P[i].A[k].dot(xux) <= P[i].b[k,0] + mi[i][k,0] * (1. - self.d[t][i]))
                        if method == 'improved_big_m':
                            sum_mik = sum(m[i][j][k,0] * self.d[t][j] for j in range(self.S.nm) if j != i)
                            self.prog.AddLinearConstraint(P[i].A[k].dot(xux) <= P[i].b[k,0] + sum_mik)

            # enforce constrained dynamics (convex hull method)
            elif method == 'convex_hull':
                for i in range(self.S.nm):
                    yvyi = np.concatenate((y[t][i], v[t][i], z[t][i]))
                    for k in range(P[i].A.shape[0]):
                        self.prog.AddLinearConstraint(P[i].A[k].dot(yvyi) <= P[i].b[k,0] * self.d[t][i])

                # recompose the state and input (convex hull method)
                for k in range(self.S.nx):
                    self.prog.AddLinearConstraint(self.x[t][k] == sum(y[t])[k])
                    self.prog.AddLinearConstraint(self.x[t+1][k] == sum(z[t])[k])
                for k in range(self.S.nu):
                    self.prog.AddLinearConstraint(self.u[t][k] == sum(v[t])[k])

            # raise error for unknown method
            else:
                raise ValueError('unknown method ' + method + '.')

            # constraints on the binaries
            self.binaries_lower_bound.append([self.prog.AddLinearConstraint(dti >= 0.).evaluator() for dti in self.d[t]])
            self.prog.AddLinearConstraint(sum(self.d[t]) == 1.)

            # stage cost to the objective
            obj += .5 * self.u[t].dot(self.R).dot(self.u[t])
            obj += .5 * self.x[t].dot(self.Q).dot(self.x[t])

        # terminal constraint
        for k in range(self.X_N.A.shape[0]):
            self.prog.AddLinearConstraint(self.X_N.A[k].dot(self.x[self.N]) <= self.X_N.b[k,0])

        # terminal cost
        obj += .5 * self.x[self.N].dot(self.P).dot(self.x[self.N])
        self.objective = self.prog.AddQuadraticCost(obj)

    def set_initial_condition(self, x0):
        for k, c in enumerate(self.initial_condition):
            c.UpdateLowerBound(x0[k])
            c.UpdateUpperBound(x0[k])

    def update_mode_sequence(self, ms):
        for t in range(self.N):
            if t < len(ms) and t < len(self.fixed_mode_sequence):
                if ms[t] != self.fixed_mode_sequence[t]:
                    self.binaries_lower_bound[t][self.fixed_mode_sequence[t]].UpdateLowerBound([0.])
                    self.binaries_lower_bound[t][ms[t]].UpdateLowerBound([1.])
            elif t >= len(ms) and t < len(self.fixed_mode_sequence):
                self.binaries_lower_bound[t][self.fixed_mode_sequence[t]].UpdateLowerBound([0.])
            elif t < len(ms) and t >= len(self.fixed_mode_sequence):
                self.binaries_lower_bound[t][ms[t]].UpdateLowerBound([1.])
        self.fixed_mode_sequence = ms

    def solve_relaxation(self, ms):

        # fix part of the mode sequence
        self.update_mode_sequence(ms)

        # solve MIQP
        #solver = MosekSolver()
        solver = GurobiSolver()
        self.prog.SetSolverOption(solver.solver_type(), 'Method', 0) # -1 default, 0 primal simplex, 1 dual simplex, 2 barrier
        # self.prog.SetSolverOption(solver.solver_type(), 'OutputFlag', 1)
        result = solver.Solve(self.prog)
        #result = self.prog.Solve()

        # check feasibility
        if result != SolutionResult.kSolutionFound:
            return None, None, None, None

        # get cost
        objective = self.prog.EvalBindingAtSolution(self.objective)[0]

        # store argmin in list of vectors
        u_list = [self.prog.GetSolution(ut).reshape(self.S.nu,1) for ut in self.u]
        x_list = [self.prog.GetSolution(xt).reshape(self.S.nx,1) for xt in self.x]
        d_list = [[self.prog.GetSolution(dti) for dti in dt] for dt in self.d]

        # retrieve mode sequence and check integer feasibility
        mode_sequence = [dt.index(max(dt)) for dt in d_list]
        integer_feasible = all([np.allclose(sorted(dt), [0.]*(len(dt)-1)+[1.]) for dt in d_list])

        # best guess for the mode at the first relaxed time step
        if len(ms) < self.N:
            mode_score = d_list[len(ms)]
        else:
            mode_score = None

        return objective, integer_feasible, mode_score, {'u': u_list, 'x': x_list, 'ms': mode_sequence}

    def feedforward(self, x0, draw_solution=False):

        # overwrite initial condition
        self.set_initial_condition(x0)

        # call branch and bound algorithm
        tree = Tree(self.solve_relaxation)
        tree.explore()

        # draw the tree
        if draw_solution:
            draw_tree(tree)

        # output
        if tree.incumbent is None:
            return None, None, None, None
        else:
            return(
                tree.incumbent.others['u'],
                tree.incumbent.others['x'],
                tree.incumbent.others['ms'],
                tree.incumbent.cost
                )

def get_graph_representation(S):
    P = []
    for i in range(S.nm):
        Di = S.domains[i]
        Si = S.affine_systems[i]
        Ai = np.vstack((
            np.hstack((Di.A, np.zeros((Di.A.shape[0], S.nx)))),
            np.hstack((Si.A, Si.B, -np.eye(S.nx))),
            np.hstack((-Si.A, -Si.B, np.eye(S.nx))),
            ))
        bi = np.vstack((Di.b, -Si.c, Si.c))
        P.append(Polyhedron(Ai, bi))
    return P

def get_big_m(P_list, tol=1.e-6):
    m = []
    for i, Pi in enumerate(P_list):
        mi = []
        for j, Pj in enumerate(P_list):
            mij = []
            for k in range(Pi.A.shape[0]):
                f = -Pi.A[k:k+1,:].T
                sol = linear_program(f, Pj.A, Pj.b)
                mijk = - sol['min'] - Pi.b[k,0]
                if np.abs(mijk) < tol:
                    mijk = 0.
                mij.append(mijk)
            mi.append(np.vstack(mij))
        m.append(mi)
    mi = [np.maximum.reduce([mij for mij in mi]) for mi in m]
    return m, mi