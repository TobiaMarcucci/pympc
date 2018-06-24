# external imports
import numpy as np
import gurobipy as grb
import scipy as sp

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.programs import linear_program, quadratic_program
from pympc.optimization.solvers.branch_and_bound import Tree

class HybridModelPredictiveController(object):

    def __init__(self, S, N, Q, R, P, X_N, method='big_m'):

        # store inputs
        self.S = S
        self.N = N
        self.Q = Q
        self.R = R
        self.P = P
        self.X_N = X_N
        self.transition_costs, self.transition_map = self.get_transition_costs()

        # mpMIQP
        # self.prog = self.build_mpmiqp(method) # adds the variables: prog, objective, u, x, d, initial_condition
        self.prog = self.build_mpmiqp_socp() # adds the variables: prog, objective, u, x, d, initial_condition
        self.partial_mode_sequence = []

    def get_transition_costs(self):
        nx = self.S.nx
        nu = self.S.nu
        transition_costs = []
        for i in range(self.S.nm):
            Si = self.S.affine_systems[i]
            Di = self.S.domains[i]
            transition_costs_i = []
            for j in range(self.S.nm):
                Dj = self.S.domains[j]
                prog = grb.Model()
                x = np.array([prog.addVars(nx, lb=[-grb.GRB.INFINITY]*nx)[k] for k in range(nx)])
                u = np.array([prog.addVars(nu, lb=[-grb.GRB.INFINITY]*nu)[k] for k in range(nu)])
                x_next = np.array([prog.addVars(nx, lb=[-grb.GRB.INFINITY]*nx)[k] for k in range(nx)])
                u_next = np.array([prog.addVars(nu, lb=[-grb.GRB.INFINITY]*nu)[k] for k in range(nu)])
                prog.update()
                xu = np.concatenate((x, u))
                xu_next = np.concatenate((x_next, u_next))
                prog.setObjective(
                    .5*(
                        x.dot(self.Q).dot(x) +
                        u.dot(self.R).dot(u) +
                        x_next.dot(self.Q).dot(x_next) +
                        u_next.dot(self.R).dot(u_next)
                        )
                    )
                for k in range(nx):
                    prog.addConstr(x_next[k] == Si.A[k].dot(x) + Si.B[k].dot(u) + Si.c[k,0])
                for k in range(Di.A.shape[0]):
                    prog.addConstr(Di.A[k].dot(xu) <= Di.b[k,0])
                for k in range(Dj.A.shape[0]):
                    prog.addConstr(Dj.A[k].dot(xu_next) <= Dj.b[k,0])
                prog.setParam('OutputFlag', 0)
                prog.optimize()
                if read_gurobi_status(prog.status) == 'optimal':
                    transition_costs_i.append(prog.objVal)
                else:
                    transition_costs_i.append(None)
            transition_costs.append(transition_costs_i)
        transition_map = [[i for i in np.argsort(transition_costs[m]) if transition_costs[m][i] is not None] for m in range(self.S.nm)]
        return transition_costs, transition_map

    # def build_mpmiqp(self, method):

    #     # shortcuts
    #     [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

    #     # express the constrained dynamics as a list of polytopes in the (x,u,x+)-space
    #     P = get_graph_representation(self.S)

    #     # get big-Ms for some of the solution methods
    #     m, mi = get_big_m(P)

    #     # initialize program
    #     prog = grb.Model()
    #     obj = 0.

    #     # loop over time
    #     for t in range(self.N):

    #         # initial conditions (set arbitrarily to zero in the building phase)
    #         if t == 0:
    #             x = prog.addVars(nx, lb=[0.]*nx, ub=[0.]*nx, name='x0')

    #         # create stage variables
    #         else:
    #             x = x_next
    #         x_next = prog.addVars(nx, lb=[-grb.GRB.INFINITY]*nx, name='x%d'%(t+1))
    #         u = prog.addVars(nu, lb=[-grb.GRB.INFINITY]*nu, name='u%d'%t)
    #         d = prog.addVars(nm, name='d%d'%t)

    #         # auxiliary continuous variables for the convex-hull method
    #         if method == 'convex_hull':
    #             y = prog.addVars(nm, nx, lb=[-grb.GRB.INFINITY]*nm*nx, name='y%d'%t)
    #             z = prog.addVars(nm, nx, lb=[-grb.GRB.INFINITY]*nm*nx, name='z%d'%t)
    #             v = prog.addVars(nm, nu, lb=[-grb.GRB.INFINITY]*nm*nu, name='v%d'%t)
    #         prog.update()

    #         # enforce constrained dynamics (big-m methods)
    #         if method in ['big_m', 'improved_big_m']:
    #             xux = np.array(x.values() + u.values() + x_next.values())
    #             for i in range(nm):
    #                 if method == 'big_m':
    #                     for k in range(P[i].A.shape[0]):
    #                         prog.addConstr(P[i].A[k].dot(xux) <= P[i].b[k,0] + mi[i][k,0] * (1. - d[i]))
    #                 if method == 'improved_big_m':
    #                     sum_mi = sum(m[i][j] * d[j] for j in range(self.S.nm) if j != i)
    #                     for k in range(P[i].A.shape[0]):
    #                         prog.addConstr(P[i].A[k].dot(xux) <= P[i].b[k,0] + sum_mi[k,0])

    #         # enforce constrained dynamics (convex hull method)
    #         elif method == 'convex_hull':
    #             for i in range(nm):
    #                 yvyi = np.array(
    #                     [y[i,k] for k in range(nx)] +
    #                     [v[i,k] for k in range(nu)] +
    #                     [z[i,k] for k in range(nx)]
    #                     )
    #                 for k in range(P[i].A.shape[0]):
    #                     prog.addConstr(P[i].A[k].dot(yvyi) <= P[i].b[k,0] * d[i])

    #             # recompose the state and input (convex hull method)
    #             for k in range(nx):
    #                 prog.addConstr(x[k] == sum(y[i,k] for i in range(nm)))
    #                 prog.addConstr(x_next[k] == sum(z[i,k] for i in range(nm)))
    #             for k in range(nu):
    #                 prog.addConstr(u[k] == sum(v[i,k] for i in range(nm)))

    #         # raise error for unknown method
    #         else:
    #             raise ValueError('unknown method ' + method + '.')

    #         # constraints on the binaries
    #         prog.addConstr(sum(d.values()) == 1.)

    #         # stage cost to the objective
    #         obj += .5 * np.array(u.values()).dot(self.R).dot(np.array(u.values()))
    #         obj += .5 * np.array(x.values()).dot(self.Q).dot(np.array(x.values()))

    #     # terminal constraint
    #     for k in range(self.X_N.A.shape[0]):
    #         prog.addConstr(self.X_N.A[k].dot(np.array(x_next.values())) <= self.X_N.b[k,0])

    #     # terminal cost
    #     obj += .5 * np.array(x_next.values()).dot(self.P).dot(np.array(x_next.values()))
    #     prog.setObjective(obj)

    #     return prog

    def build_mpmiqp_socp(self):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # express the constrained dynamics as a list of polytopes in the (x,u,x+)-space
        P = get_graph_representation(self.S)

        # get big-Ms for some of the solution methods
        m, mi = get_big_m(P)

        # initialize program
        prog = grb.Model()
        obj = 0.
        LQ = np.linalg.cholesky(self.Q)
        LR = np.linalg.cholesky(self.R)
        LP = np.linalg.cholesky(self.P)

        # loop over time
        for t in range(self.N):

            # initial conditions (set arbitrarily to zero in the building phase)
            if t == 0:
                x = prog.addVars(nx, lb=[1.]*nx, ub=[1.]*nx, name='x0')

            # create stage variables
            else:
                x = x_next
            x_next = prog.addVars(nx, lb=[-grb.GRB.INFINITY]*nx, name='x%d'%(t+1))
            u = prog.addVars(nu, lb=[-grb.GRB.INFINITY]*nu, name='u%d'%t)
            d = prog.addVars(nm, name='d%d'%t)

            # auxiliary continuous variables
            x_aux = prog.addVars(nm, nx, lb=[-grb.GRB.INFINITY]*nm*nx, name='x_aux%d'%t)
            u_aux = prog.addVars(nm, nu, lb=[-grb.GRB.INFINITY]*nm*nu, name='u_aux%d'%t)
            x_next_aux = prog.addVars(nm, nx, lb=[-grb.GRB.INFINITY]*nm*nx, name='x_next_aux%d'%t)

            # slack and uxiliary continuous variables for the cost function
            s = prog.addVars(nm, name='s%d'%t)
            prog.update()

            # enforce constrained dynamics
            for i in range(nm):
                var = np.array(
                    [x_aux[i,k] for k in range(nx)] +
                    [u_aux[i,k] for k in range(nu)] +
                    [x_next_aux[i,k] for k in range(nx)]
                    )
                for k in range(P[i].A.shape[0]):
                    prog.addConstr(P[i].A[k].dot(var) <= P[i].b[k,0] * d[i])

            # enforce cost function
            for i in range(nm):
                x_aux_i = np.array([x_aux[i,k] for k in range(nx)])
                u_aux_i = np.array([u_aux[i,k] for k in range(nu)])
                x_chol = np.array([prog.addVars(nx, lb=[-grb.GRB.INFINITY]*nx, name='x_chol_%d_%d'%(t,i))[k]for k in range(nx)])
                u_chol = np.array([prog.addVars(nu, lb=[-grb.GRB.INFINITY]*nu, name='u_chol_%d_%d'%(t,i))[k]for k in range(nu)])
                prog.update()
                for k in range(nx):
                    prog.addConstr(x_chol[k] == LQ.T[k].dot(x_aux_i))
                for k in range(nu):
                    prog.addConstr(u_chol[k] == LR.T[k].dot(u_aux_i))

                if t < self.N - 1:
                    prog.addConstr(.5 * (x_chol.dot(x_chol) + u_chol.dot(u_chol)) <= s[i] * d[i])
                else:
                    x_next_aux_i = np.array([x_next_aux[i,k] for k in range(nx)])
                    x_next_chol = np.array([prog.addVars(nx, lb=[-grb.GRB.INFINITY]*nx, name='x_next_chol_%d_%d'%(t,i))[k] for k in range(nx)])
                    prog.update()
                    for k in range(nx):
                        prog.addConstr(x_next_chol[k] == LP.T[k].dot(x_next_aux_i))
                    prog.addConstr(.5 * (x_chol.dot(x_chol) + u_chol.dot(u_chol) + x_next_chol.dot(x_next_chol)) <= s[i] * d[i])

                    # terminal constraint
                    for k in range(self.X_N.A.shape[0]):
                        prog.addConstr(self.X_N.A[k].dot(x_next_aux_i) <= self.X_N.b[k,0] * d[i])

            obj += sum(s.values())

            # recompose the state and input
            for k in range(nx):
                prog.addConstr(x[k] == sum(x_aux[i,k] for i in range(nm)))
                prog.addConstr(x_next[k] == sum(x_next_aux[i,k] for i in range(nm)))
            for k in range(nu):
                prog.addConstr(u[k] == sum(u_aux[i,k] for i in range(nm)))

            # constraints on the binaries
            prog.addConstr(sum(d.values()) == 1.)

        # set cost
        prog.setObjective(obj)

        return prog

    def set_initial_condition(self, x0):
        for k in range(self.S.nx):
            self.prog.getVarByName('x0[%d]'%k).LB = x0[k,0]
            self.prog.getVarByName('x0[%d]'%k).UB = x0[k,0]

    def update_mode_sequence(self, partial_mode_sequence):

        # loop over the time horizon
        for t in range(self.N):

            # write and erase
            if t < len(partial_mode_sequence) and t < len(self.partial_mode_sequence):
                if partial_mode_sequence[t] != self.partial_mode_sequence[t]:
                    self.prog.getVarByName('d%d[%d]'%(t,self.partial_mode_sequence[t])).LB = 0.
                    self.prog.getVarByName('d%d[%d]'%(t,partial_mode_sequence[t])).LB = 1.

            # erase only
            elif t >= len(partial_mode_sequence) and t < len(self.partial_mode_sequence):
                self.prog.getVarByName('d%d[%d]'%(t,self.partial_mode_sequence[t])).LB = 0.

            # write only
            elif t < len(partial_mode_sequence) and t >= len(self.partial_mode_sequence):
                self.prog.getVarByName('d%d[%d]'%(t,partial_mode_sequence[t])).LB = 1.

        # update partial mode sequence
        self.partial_mode_sequence = partial_mode_sequence

    def solve_relaxation(self, partial_mode_sequence, cutoff_value=None, warm_start=None):

        # reset program
        self.prog.reset()

        # # warm start for active set method
        # if warm_start is not None:
        #     for i, v in enumerate(self.prog.getVars()):
        #         v.VBasis = warm_start['variable_basis'][i]
        #     for i, c in enumerate(self.prog.getConstrs()):
        #         c.CBasis = warm_start['constraint_basis'][i]
        # self.prog.update()

        # parameters
        self.prog.setParam('OutputFlag', 0)
        # self.prog.setParam('Method', 0)
        # self.prog.setParam('BarQCPConvTol', 1.e-9)

        # set cut off from best upper bound
        if cutoff_value is None:
            self.prog.setParam('Cutoff', grb.GRB.INFINITY)
        else:
            self.prog.setParam('Cutoff', cutoff_value)

        # fix part of the mode sequence
        self.update_mode_sequence(partial_mode_sequence)

        # run the optimization
        self.prog.optimize()
        result = dict()
        result['solve_time'] = self.prog.Runtime

        # check status
        result['cutoff'] = read_gurobi_status(self.prog.status) == 'cutoff'
        if result['cutoff']:
        	result['feasible'] = None
        else:
        	result['feasible'] = read_gurobi_status(self.prog.status) == 'optimal'

        # return if cutoff or unfeasible
        if result['cutoff'] or not result['feasible']:
        	return result

        # store argmin in list of vectors
        result['x'] = [np.vstack([self.prog.getVarByName('x%d[%d]'%(t,k)).x for k in range(self.S.nx)]) for t in range(self.N+1)]
        result['u'] = [np.vstack([self.prog.getVarByName('u%d[%d]'%(t,k)).x for k in range(self.S.nu)]) for t in range(self.N)]
        d = [[self.prog.getVarByName('d%d[%d]'%(t,k)).x for k in range(self.S.nm)] for t in range(self.N)]

        # for dt in d:
        #     print [round(dti, 4) for dti in dt]

        # retrieve mode sequence and check integer feasibility
        result['mode_sequence'] = [dt.index(max(dt)) for dt in d]
        result['integer_feasible'] = all([np.allclose(sorted(dt), [0.]*(len(dt)-1)+[1.], atol=1.e-6) for dt in d])

        # heuristic to guess the optimal mode at the first relaxed time step
        if len(partial_mode_sequence) < self.N:
            result['children_order'], result['children_score'] = self.mode_heuristic(d)
        else:
            result['children_order'] = None
            result['children_score'] = None
        # if len(partial_mode_sequence) == 0.:
        #     result['children_order'] = self.transition_map[2]
        #     result['children_score'] = self.transition_costs[2]
        # elif len(partial_mode_sequence) < self.N:
        #     result['children_order'] = self.transition_map[partial_mode_sequence[-1]]
        #     result['children_score'] = self.transition_costs[partial_mode_sequence[-1]]
        # else:
        #     result['children_order'] = None
        #     result['children_score'] = None

        # other solver outputs
        result['cost'] = self.prog.objVal
        # result['variable_basis'] = [v.VBasis for v in self.prog.getVars()]
        # result['constraint_basis'] = [c.CBasis for c in self.prog.getConstrs()]

        return result

    def mode_heuristic(self, d):

        # order by the value of the relaxed binaries
        children_score = d[len(self.partial_mode_sequence)]
        children_order = np.argsort(children_score)[::-1].tolist()

        # # put in front the mode of the parent node
        # if len(self.partial_mode_sequence) > 0:
        #     children_order.insert(0, children_order.pop(children_order.index(self.partial_mode_sequence[-1])))

        return children_order, children_score

    def feedforward(self, x0, draw_solution=False):

        # reset program
        self.prog.reset()

        # overwrite initial condition
        self.prog.reset()
        self.set_initial_condition(x0)

        # call branch and bound algorithm
        tree = Tree(self.solve_relaxation)
        tree.explore()
        print tree.get_solve_time()

        # draw the tree
        if draw_solution:
            tree.draw()

        # output
        if tree.incumbent is None:
            return [None]*4
        else:
            return [tree.incumbent.result[key] for key in ['u', 'x', 'mode_sequence', 'cost']]

    def feedforward_gurobi(self, x0):

        # reset program
        self.prog.reset()

        # set up miqp
        self.set_d_type('B')
        self.update_mode_sequence([])
        self.set_initial_condition(x0)

        # parameters
        self.prog.setParam('OutputFlag', 1)
        self.prog.setParam('Threads', 0)

        # run the optimization
        self.prog.optimize()
        print self.prog.Runtime
        self.set_d_type('C')

        # output
        if read_gurobi_status(self.prog.status) == 'optimal':
            x = [np.vstack([self.prog.getVarByName('x%d[%d]'%(t,k)).x for k in range(self.S.nx)]) for t in range(self.N+1)]
            u = [np.vstack([self.prog.getVarByName('u%d[%d]'%(t,k)).x for k in range(self.S.nu)]) for t in range(self.N)]
            d = [[self.prog.getVarByName('d%d[%d]'%(t,k)).x for k in range(self.S.nm)] for t in range(self.N)]
            ms = [dt.index(max(dt)) for dt in d]
            cost = self.prog.objVal
            return u, x, ms, cost
        else:
            return [None]*4

    def set_d_type(self, d_type):
        for t in range(self.N):
            for i in range(self.S.nm):
                self.prog.getVarByName('d%d[%d]'%(t,i)).VType = d_type

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

def read_gurobi_status(status):
	return {
		1: 'loaded', # Model is loaded, but no solution information is available.'
		2: 'optimal',	# Model was solved to optimality (subject to tolerances), and an optimal solution is available.
		3: 'infeasible', # Model was proven to be infeasible.
		4: 'inf_or_unbd', # Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.
		5: 'unbounded', # Model was proven to be unbounded. Important note: an unbounded status indicates the presence of an unbounded ray that allows the objective to improve without limit. It says nothing about whether the model has a feasible solution. If you require information on feasibility, you should set the objective to zero and reoptimize.
		6: 'cutoff', # Optimal objective for model was proven to be worse than the value specified in the Cutoff parameter. No solution information is available. (Note: problem might also be infeasible.)
		7: 'iteration_limit', # Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterationLimit parameter, or because the total number of barrier iterations exceeded the value specified in the BarIterLimit parameter.
		8: 'node_limit', # Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter.
		9: 'time_limit', # Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter.
		10: 'solution_limit', # Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter.
		11: 'interrupted', # Optimization was terminated by the user.
		12: 'numeric', # Optimization was terminated due to unrecoverable numerical difficulties.
		13: 'suboptimal', # Unable to satisfy optimality tolerances; a sub-optimal solution is available.
		14: 'in_progress', # An asynchronous optimization call was made, but the associated optimization run is not yet complete.
		15: 'user_obj_limit' # User specified an objective limit (a bound on either the best objective or the best bound), and that limit has been reached.
		}[status]