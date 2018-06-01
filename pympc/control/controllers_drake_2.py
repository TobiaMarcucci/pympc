# external imports
import numpy as np

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.programs import linear_program

# pydrake imports
from pydrake.all import MathematicalProgram, SolutionResult
from pydrake.solvers.gurobi import GurobiSolver
from pydrake.solvers.mosek import MosekSolver

class HybridModelPredictiveController(object):

    def __init__(self, S, N, Q, R, P, X_N):

        # store inputs
        self.S = S
        self.N = N
        self.Q = Q
        self.R = R
        self.P = P
        self.X_N = X_N

        # mpMIQP
        self.prog = None

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
        cost += .5 * sum([x.T.dot(self.Q).dot(x) for x in x_list])
        cost += .5 * x_list[self.N].T.dot(self.P).dot(x_list[self.N])

        return u_list, x_list, mode_sequence, cost[0,0]

    # def set_initial_state(self, x0):
    # 	self.initial_state_constraint.UpdateCoefficients(rhs=x0)

    # def update_mode_sequence(self, new_ms):
    # 	for t in range(self.N):
    # 	    if t < len(new_ms) and t < len(self.current_ms):
    # 	    	if new_ms[t] != self.current_ms[t]:
    # 	    		self.binary_constraints[t][self.current_ms[t]].UpdateCoefficients(rhs=0.)
    # 	    		self.binary_constraints[t][new_ms[t]].UpdateCoefficients(rhs=1.)
    # 	    elif t >= len(new_ms) and t < len(self.current_ms):
    # 	    	self.binary_constraints[t][self.current_ms[t]].UpdateCoefficients(rhs=0.)
    # 	    elif t < len(new_ms) and t >= len(self.current_ms):
    # 	    	self.binary_constraints[t][new_ms[t]].UpdateCoefficients(rhs=1.)


	    for t, mode in enumerate(new_ms):
	    	if t >= len(self.current_ms):
	    		cons_t = self.prog.AddLinearConstraint(d[t][mode] >= 1.)
	    		self.ms_constraints.append(cons_t)
		    elif mode != self.current_ms[t]:
		    	self.prog.EraseLinearConstraint(self.ms_constraints[t])
		    	cons_t = self.prog.AddLinearConstraint(self.d[t][mode] >= 1.)
		    	self.ms_constraints[t] = cons_t
		self.current_ms = new_ms
		
    def feedforward(self, x0, ms, method='big_m'):

        P = get_graph_representation(self.S)
        m, mi = get_big_m(P)

        # initialize program and cost function
        prog = MathematicalProgram()
        obj = 0.

        # state and input variables
        u = [prog.NewContinuousVariables(self.S.nu) for t in range(self.N)]
        x = [prog.NewContinuousVariables(self.S.nx) for t in range(self.N+1)]

        # auxiliary continuous variables
        if method == 'convex_hull':
            y = [[prog.NewContinuousVariables(self.S.nx) for i in range(self.S.nm)] for t in range(self.N-len(ms))]
            v = [[prog.NewContinuousVariables(self.S.nu) for i in range(self.S.nm)] for t in range(self.N-len(ms))]
            y_next = [[prog.NewContinuousVariables(self.S.nx) for i in range(self.S.nm)] for t in range(self.N-len(ms))]

        # auxiliary binary variables
        d = [[prog.NewContinuousVariables(1)[0] for i in range(self.S.nm)] for t in range(self.N-len(ms))]
        for dt in d:
            for dti in dt:
                prog.AddLinearConstraint(dti >= 0.)

        # initial conditions
        for k in range(self.S.nx):
            prog.AddLinearConstraint(x[0][k] == x0[k])

        # loop over time
        for t in range(self.N):

            # add stage cost
            obj += .5*x[t].dot(self.Q).dot(x[t])
            obj += .5*u[t].dot(self.R).dot(u[t])

            # scheduled horizon
            if t < len(ms):
                St = self.S.affine_systems[ms[t]]
                Dt = self.S.domains[ms[t]]
                xut = np.concatenate((x[t], u[t]))
                for k in range(St.nx):
                    prog.AddLinearConstraint(x[t+1][k] == St.A[k].dot(x[t]) + St.B[k].dot(u[t]) + St.c[k,0])
                for k in range(Dt.A.shape[0]):
                    prog.AddLinearConstraint(Dt.A[k].dot(xut) <= Dt.b[k,0])

            # unscheduled horizon
            else:

                # one mode per time
                prog.AddLinearConstraint(sum(d[t-len(ms)]) == 1.)

                # standard big-m
                if method in ['big_m', 'improved_big_m']:

                    # add mode constraints and dynamics
                    xux = np.concatenate((x[t], u[t], x[t+1]))
                    for i in range(self.S.nm):
                        for k in range(P[i].A.shape[0]):
                            if method == 'big_m':
                                prog.AddLinearConstraint(P[i].A[k].dot(xux) <= P[i].b[k,0] + mi[i][k,0] * (1. - d[t-len(ms)][i]))
                            if method == 'improved_big_m':
                                sum_mik = sum(m[i][j][k,0] * d[t-len(ms)][j] for j in range(self.S.nm) if j != i)
                                prog.AddLinearConstraint(P[i].A[k].dot(xux) <= P[i].b[k,0] + sum_mik)

                # convex hull method
                if method == 'convex_hull':

                    # recompose the state and input
                    for k in range(self.S.nx):
                        prog.AddLinearConstraint(x[t][k] == sum(y[t-len(ms)])[k])
                        prog.AddLinearConstraint(x[t+1][k] == sum(y_next[t-len(ms)])[k])
                    for k in range(self.S.nu):
                        prog.AddLinearConstraint(u[t][k] == sum(v[t-len(ms)])[k])

                    # add mode constraints and dynamics
                    for i in range(self.S.nm):
                        yvyi = np.concatenate((y[t-len(ms)][i], v[t-len(ms)][i], y_next[t-len(ms)][i]))
                        for k in range(P[i].A.shape[0]):
                            prog.AddLinearConstraint(P[i].A[k].dot(yvyi) <= P[i].b[k,0] * d[t-len(ms)][i])

        # terminal constraint
        for k in range(self.X_N.A.shape[0]):
            prog.AddLinearConstraint(self.X_N.A[k].dot(x[self.N]) <= self.X_N.b[k,0])

        # terminal cost
        obj += .5*x[self.N].dot(self.P).dot(x[self.N])
        prog.AddQuadraticCost(obj)

        # set solver
        #solver = MosekSolver()
        solver = GurobiSolver()
        #prog.SetSolverOption(solver.solver_type(), 'OutputFlag', 1) # prints on the terminal!
        #prog.SetSolverOption(solver.solver_type(), 'Heuristics', 0.)

        # solve MIQP
        result = solver.Solve(prog)

        if result != SolutionResult.kSolutionFound:
            others = {'u': None, 'x': None, 'ms': None}
            return None, None, None, others

        # from vector to list of vectors
        u_list = [prog.GetSolution(ut).reshape(self.S.nu,1) for ut in u]
        x_list = [prog.GetSolution(xt).reshape(self.S.nx,1) for xt in x]
        d_list = [[prog.GetSolution(dti) for dti in dt] for dt in d]
        mode_sequence = ms + [dt.index(max(dt)) for dt in d_list]
        if len(d_list) > 0:
            d0 = d_list[0]
            next_mode_order = list(reversed(np.argsort(d_list[0]).tolist()))
        else:
            d0 = None
            next_mode_order = None
        cost  = .5 * sum([u.T.dot(self.R).dot(u) for u in u_list])
        cost += .5 * sum([x.T.dot(self.Q).dot(x) for x in x_list])
        cost += .5 * x_list[self.N].T.dot(self.P).dot(x_list[self.N])

        is_integer = all([np.allclose(sorted(dt), [0.]*(len(dt)-1)+[1.]) for dt in d_list])

        others = {'u': u_list, 'x': x_list, 'ms': mode_sequence, 'binaries':d0}

        return cost[0,0], is_integer, next_mode_order, others

# def feedforward_rverse_time ??

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