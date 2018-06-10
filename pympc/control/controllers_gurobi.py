# external imports
import numpy as np
import gurobipy as grb
from collections import OrderedDict

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.optimization.programs import linear_program
from pympc.optimization.solvers.branch_and_bound import Tree, draw_tree
from pympc.optimization.solvers.gurobi import linear_expression, quadratic_expression

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
        self.prog = self.build_mpmiqp(method) # adds the variables: prog, objective, u, x, d, initial_condition
        self.fixed_mode_sequence = []

    def build_mpmiqp(self, method):

        # shortcuts
        [nx, nu, nm] = [self.S.nx, self.S.nu, self.S.nm]

        # express the constrained dynamics as a list of polytopes in the (x,u,x+)-space
        P = get_graph_representation(self.S)

        # get big-Ms for some of the solution methods
        m, mi = get_big_m(P)

        # initialize program
        prog = grb.Model()
        obj = 0.

        # parameters
        prog.setParam('OutputFlag', 0)
        prog.setParam('Method', 0)

        # loop over time
        for t in range(self.N):

            # initial conditions (set arbitrarily to zero in the building phase)
            if t == 0:
                x = prog.addVars(nx, lb=[0.]*nx, ub=[0.]*nx, name='x0')

            # create stage variables
            else:
                x = x_next
            x_next = prog.addVars(nx, lb=[-grb.GRB.INFINITY]*nx, name='x%d'%(t+1))
            u = prog.addVars(nu, lb=[-grb.GRB.INFINITY]*nu, name='u%d'%t)
            d = prog.addVars(nm, name='d%d'%t)

            # auxiliary continuous variables for the convex-hull method
            if method == 'convex_hull':
                y = prog.addVars(nm, nx, lb=[-grb.GRB.INFINITY]*nm*nx, name='y%d'%t)
                z = prog.addVars(nm, nx, lb=[-grb.GRB.INFINITY]*nm*nx, name='z%d'%t)
                v = prog.addVars(nm, nu, lb=[-grb.GRB.INFINITY]*nm*nu, name='v%d'%t)
            prog.update()

            # enforce constrained dynamics (big-m methods)
            if method in ['big_m', 'improved_big_m']:
                xux = np.array(x.values() + u.values() + x_next.values())
                for i in range(nm):
                    if method == 'big_m':
                        for k in range(P[i].A.shape[0]):
                            prog.addConstr(P[i].A[k].dot(xux) <= P[i].b[k,0] + mi[i][k,0] * (1. - d[i]))
                    if method == 'improved_big_m':
                        sum_mi = sum(m[i][j] * d[j] for j in range(self.S.nm) if j != i)
                        for k in range(P[i].A.shape[0]):
                            prog.addConstr(P[i].A[k].dot(xux) <= P[i].b[k,0] + sum_mi[k,0])

            # enforce constrained dynamics (convex hull method)
            elif method == 'convex_hull':
                for i in range(nm):
                    yvyi = np.array(
                        [y[i,k] for k in range(nx)] +
                        [v[i,k] for k in range(nu)] +
                        [z[i,k] for k in range(nx)]
                        )
                    for k in range(P[i].A.shape[0]):
                        prog.addConstr(P[i].A[k].dot(yvyi) <= P[i].b[k,0] * d[i])

                # recompose the state and input (convex hull method)
                for k in range(nx):
                    prog.addConstr(x[k] == sum(y[i,k] for i in range(nm)))
                    prog.addConstr(x_next[k] == sum(z[i,k] for i in range(nm)))
                for k in range(nu):
                    prog.addConstr(u[k] == sum(v[i,k] for i in range(nm)))

            # raise error for unknown method
            else:
                raise ValueError('unknown method ' + method + '.')

            # constraints on the binaries
            prog.addConstr(sum(d.values()) == 1.)

            # stage cost to the objective
            obj += .5 * np.array(u.values()).dot(self.R).dot(np.array(u.values()))
            obj += .5 * np.array(x.values()).dot(self.Q).dot(np.array(x.values()))

        # terminal constraint
        for k in range(self.X_N.A.shape[0]):
            prog.addConstr(self.X_N.A[k].dot(np.array(x_next.values())) <= self.X_N.b[k,0])

        # terminal cost
        obj += .5 * np.array(x_next.values()).dot(self.P).dot(np.array(x_next.values()))
        prog.setObjective(obj)

        return prog

    def set_initial_condition(self, x0):
        for k in range(self.S.nx):
            self.prog.getVarByName('x0[%d]'%k).LB = x0[k,0]
            self.prog.getVarByName('x0[%d]'%k).UB = x0[k,0]

    def update_mode_sequence(self, ms):
        for t in range(self.N):
            if t < len(ms) and t < len(self.fixed_mode_sequence):
                if ms[t] != self.fixed_mode_sequence[t]:
                    self.prog.getVarByName('d%d[%d]'%(t,self.fixed_mode_sequence[t])).LB = 0.
                    self.prog.getVarByName('d%d[%d]'%(t,ms[t])).LB = 1.
            elif t >= len(ms) and t < len(self.fixed_mode_sequence):
                self.prog.getVarByName('d%d[%d]'%(t,self.fixed_mode_sequence[t])).LB = 0.
            elif t < len(ms) and t >= len(self.fixed_mode_sequence):
                self.prog.getVarByName('d%d[%d]'%(t,ms[t])).LB = 1.
        self.fixed_mode_sequence = ms

    def solve_relaxation(self, ms, variable_basis=None, constraint_basis=None, upper_bound=None):

        # self.prog.reset()

        # warm start for active set method
        if variable_basis is not None:
            for i, v in enumerate(self.prog.getVars()):
                v.VBasis = variable_basis[i]
        if constraint_basis is not None:
            for i, c in enumerate(self.prog.getConstrs()):
                c.CBasis = constraint_basis[i]
        self.prog.update()

        # set cut off from best upper bound
        if upper_bound is not None:
            self.prog.setParam('Cutoff', upper_bound)

        # fix part of the mode sequence
        self.update_mode_sequence(ms)

        # run the optimization
        self.prog.optimize()

        # check status
        print read_gurobi_status(self.prog.status)





        if self.prog.status != 2: # optimal
            return {
                'feasible': self.prog.status == 6,
                'cost': None,
                'integer_feasible': None,
                'children_score': None,
                'solve_time': self.prog.Runtime,
                'mode_sequence': None,
                'variable_basis': None,
                'constraint_basis': None,
                'cutoff': self.prog.status == 6, # 2: optimal, 3: infeasible, 4: infeasible or unbounded, 6: cutoff
                'u': None,
                'x': None,
                }

        # store argmin in list of vectors
        x_list = [[self.prog.getVarByName('x%d[%d]'%(t,k)).x for k in range(self.S.nx)] for t in range(self.N+1)]
        u_list = [[self.prog.getVarByName('u%d[%d]'%(t,k)).x for k in range(self.S.nu)] for t in range(self.N)]
        d_list = [[self.prog.getVarByName('d%d[%d]'%(t,k)).x for k in range(self.S.nm)] for t in range(self.N)]

        # retrieve mode sequence and check integer feasibility
        mode_sequence = [dt.index(max(dt)) for dt in d_list]
        integer_feasible = all([np.allclose(sorted(dt), [0.]*(len(dt)-1)+[1.]) for dt in d_list])

        # best guess for the mode at the first relaxed time step
        if len(ms) < self.N:
            children_score = d_list[len(ms)]
        else:
            children_score = None

        return {
            'feasible': True,
            'cost': self.prog.objVal,
            'integer_feasible': integer_feasible,
            'children_score': children_score,
            'solve_time': self.prog.Runtime,
            'mode_sequence': mode_sequence,
            'variable_basis': [v.VBasis for v in self.prog.getVars()],
            'constraint_basis': [c.CBasis for c in self.prog.getConstrs()],
            'cutoff': False, # 2: optimal, 3: infeasible, 4: infeasible or unbounded, 6: cutoff
            'u': u_list,
            'x': x_list,
            }

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
                tree.incumbent.result['u'],
                tree.incumbent.result['x'],
                tree.incumbent.result['mode_sequence'],
                tree.incumbent.result['cost']
                )

def read_gurobi_status(status):
	return {
		1: 'loaded', # Model is loaded, but no solution information is available.'
		2: 'optimal',	# Model was solved to optimality (subject to tolerances), and an optimal solution is available.
		3: 'infeasible', # Model was proven to be infeasible.
		4: 'inf_or_unbd', # Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.
		5: 'unbounded', # Model was proven to be unbounded. Important note: an unbounded status indicates the presence of an unbounded ray that allows the objective to improve without limit. It says nothing about whether the model has a feasible solution. If you require information on feasibility, you should set the objective to zero and reoptimize.
		6: 'cutoff', # Optimal objective for model was proven to be worse than the value specified in the Cutoff parameter. No solution information is available. Problem might also be infeasible.
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