import numpy as np
import matplotlib.pyplot as plt
import dynamical_systems as ds
import plot as mpc_plt
from control import MPCHybridController
from geometry import Polytope
from optimization.pnnls import linear_program
from optimization.gurobi import real_variable, iff_point_in_polyhedron
from control import suppress_stdout
import gurobipy as grb
import time


# dynamics

t_s = .1

A_0 = np.array([[-1.]])
B_0 = np.array([[1.]])
c_0 = np.array([[-2.]])
sys_0 = ds.DTAffineSystem.from_continuous(A_0, B_0, c_0, t_s)

A_1 = np.array([[1.]])
B_1 = np.array([[1.]])
c_1 = np.array([[0.]])
sys_1 = ds.DTAffineSystem.from_continuous(A_1, B_1, c_1, t_s)

A_2 = np.array([[-1.]])
B_2 = np.array([[1.]])
c_2 = np.array([[2.]])
sys_2 = ds.DTAffineSystem.from_continuous(A_2, B_2, c_2, t_s)

sys = [sys_0, sys_1, sys_2]

# domains

x_min_0 = np.array([[-3.]])
x_max_0 = np.array([[-1.]])
X_0 = Polytope.from_bounds(x_max_0, x_min_0)
X_0.assemble()

x_min_1 = x_max_0
x_max_1 =  - x_max_0
X_1 = Polytope.from_bounds(x_max_1, x_min_1)
X_1.assemble()

x_min_2 = x_max_1
x_max_2 = - x_min_0
X_2 = Polytope.from_bounds(x_max_2, x_min_2)
X_2.assemble()

u_max = np.array([[10.]])
u_min = -u_max

U_0 = Polytope.from_bounds(u_max, u_min)
U_0.assemble()

U_1 = U_0

U_2 = U_0

X = [X_0, X_1, X_2]
U = [U_0, U_1, U_2]

pwa_sys = ds.DTPWASystem(sys, X, U)

# controller

N = 2
Q = np.eye(A_1.shape[0])
R = np.eye(B_1.shape[1])
P = Q
objective_norm = 'one'
X_N = X_1
controller = MPCHybridController(pwa_sys, N, objective_norm, Q, R, P, X_N)

# # simulation

# N_sim = 100
# x_0 = np.array([[-2.5]])
# u = []
# x = []
# x.append(x_0)
# for k in range(N_sim):
#     u.append(controller.feedback(x[k]))
#     x_next = pwa_sys.simulate(x[k], [u[k]])[0][1]
#     x.append(x_next)


# mpc_plt.input_sequence(u, t_s, N_sim, (u_max, u_min))
# plt.show()
# mpc_plt.state_trajectory(x, t_s, N_sim, (x_min_0, x_min_1, x_min_2, x_max_2))
# plt.show()

# enumeration of switching sequences

ss_list = [[1]*N]
for len_contact in range(1,N+1):
    ss_0 = [1]*(N-len_contact) + [0]*len_contact
    ss_2 = [1]*(N-len_contact) + [2]*len_contact
    for i in range(N-len_contact+1):
        ss_list += [ss_0, ss_2]
        ss_0 = ss_0[1:] + [1]
        ss_2 = ss_2[1:] + [1]

# plot optimal value functions

n_samples = 100
fs_list = []
for i, ss in enumerate(ss_list):
    col = np.random.rand(3,1)
    fs = controller.backward_reachability_analysis(ss)
    fs_list.append(fs)
    prog = controller.condense_program(ss)
    x_samples = np.linspace(fs.vertices[0][0,0], fs.vertices[1][0,0], n_samples)
    V_samples = []
    for j, x in enumerate(x_samples):
        V = prog.solve(np.array([[x]]))[1]
        V_samples.append(V)
        if j == int(n_samples/2):
            plt.text(x, V, str(i))
    plt.plot(x_samples, V_samples, color=col)

#plt.show()

# single feasibility regions

fs_single_feasibility = []
fs_multiple_feasibility = []

n_x = 1
n_u = 1
tic = time.time()
for i, fs in enumerate(fs_list):
    print i
    other_fs = [fs_j for j, fs_j in enumerate(fs_list) if j != i]
    if fs.included_in_union_of(other_fs):
        fs_multiple_feasibility.append(i)
    else:
        fs_single_feasibility.append(i)
print('Single feasibility sets ' + str(fs_single_feasibility) + ' computed in ' + str(time.time() - tic) + ' s')



# # candidate feasible regions

# bigM = 1.e3

# candidate_fs = fs_single_feasibility

# for i in fs_multiple_feasibility:
# #for i in [0]:
#     print i

#     fs = fs_list[i]

#     # inclusions
#     intersections = []
#     inclusions = []
#     for j, fs_j in enumerate(fs_list):
#         if j != i:
#             if fs.included_in(fs_j):
#                 inclusions.append(j)
#             elif fs.intersect_with(fs_j):
#                 intersections.append(j)
#     print intersections, inclusions

#     # gurobi model
#     model = grb.Model()

#     # state
#     x, model = real_variable(model, [n_x])
#     x_np = np.array([[x[k]] for k in range(n_x)])

#     # flag on other polytopes
#     d_dom = []
#     objective = 0.
#     for j in intersections:
#         model, objective, d_j = iff_point_in_polyhedron(model, objective, fs_list[j].lhs_min, fs_list[j].rhs_min, x, fs)
#         d_dom.append(d_j)
#     model.update()

#     # objective
#     model.setObjective(objective)

#     # KKT for this polyhedron

#     prog_i = controller.condense_program(ss_list[i])
#     z_i, model = real_variable(model, [prog_i.n_var])
#     l_i = model.addVars(prog_i.n_cons) # added this way since l >= 0
#     d_i_com = model.addVars(prog_i.n_cons, vtype=grb.GRB.BINARY)
#     z_i_np = np.array([[z_i[k]] for k in range(prog_i.n_var)])
#     l_i_np = np.array([[l_i[k]] for k in range(prog_i.n_cons)])
#     # stationarity
#     expr = prog_i.f + prog_i.A.T.dot(l_i_np)
#     model.addConstrs((expr[k,0] == 0. for k in range(expr.shape[0])))
#     # primal feasibility
#     expr = prog_i.A.dot(z_i_np) - prog_i.B.dot(x_np) - prog_i.c
#     model.addConstrs((expr[k,0] <= 0. for k in range(expr.shape[0])))
#     # mixed-integer complementarity
#     model.addConstrs((expr[k,0] >= -bigM*(1.-d_i_com[k]) for k in range(expr.shape[0])))
#     model.addConstrs((l_i[k] <= bigM*d_i_com[k] for k in range(prog_i.n_cons)))

#     for j in inclusions:

#         prog = controller.condense_program(ss_list[j])
#         z, model = real_variable(model, [prog.n_var])
#         l = model.addVars(prog.n_cons) # added this way since l >= 0
#         d_com = model.addVars(prog.n_cons, vtype=grb.GRB.BINARY)
#         z_np = np.array([[z[k]] for k in range(prog.n_var)])
#         l_np = np.array([[l[k]] for k in range(prog.n_cons)])
#         # stationarity
#         expr = prog.f + prog.A.T.dot(l_np)
#         model.addConstrs((expr[k,0] == 0. for k in range(expr.shape[0])))
#         # primal feasibility
#         expr = prog.A.dot(z_np) - prog.B.dot(x_np) - prog.c
#         model.addConstrs((expr[k,0] <= 0. for k in range(expr.shape[0])))
#         # mixed-integer complementarity
#         model.addConstrs((expr[k,0] >= -bigM*(1.-d_com[k]) for k in range(expr.shape[0])))
#         model.addConstrs((l[k] <= bigM*d_com[k] for k in range(prog.n_cons)))

#         # compare costs
#         expr = (prog_i.f.T.dot(z_i_np) - prog.f.T.dot(z_np))[0,0]
#         model.addConstr(expr <= 0.)

#     # KKT for the other polyhedra
#     for t, j in enumerate(intersections):
#         prog = controller.condense_program(ss_list[j])
#         z, model = real_variable(model, [prog.n_var])
#         l = model.addVars(prog.n_cons) # added this way since l >= 0
#         d_com = model.addVars(prog.n_cons, vtype=grb.GRB.BINARY)
#         z_np = np.array([[z[k]] for k in range(prog.n_var)])
#         l_np = np.array([[l[k]] for k in range(prog.n_cons)])
#         # stationarity
#         expr = prog.f + prog.A.T.dot(l_np)
#         model.addConstrs((expr[k,0] <= bigM*(1.-d_dom[t]) for k in range(expr.shape[0])))
#         model.addConstrs((expr[k,0] >= - bigM*(1.-d_dom[t]) for k in range(expr.shape[0])))
#         # primal feasibility
#         expr = prog.A.dot(z_np) - prog.B.dot(x_np) - prog.c
#         model.addConstrs((expr[k,0] <= bigM*(1.-d_dom[t]) for k in range(expr.shape[0])))
#         # mixed-integer complementarity
#         model.addConstrs((expr[k,0] >= - bigM*(1.-d_com[k]) - bigM*(1.-d_dom[t]) for k in range(expr.shape[0])))
#         model.addConstrs((l[k] <= bigM*d_com[k] + bigM*(1.-d_dom[t]) for k in range(prog.n_cons)))

#         # recover cost function
#         phi = model.addVar(lb=-grb.GRB.INFINITY)
#         phi_min = (prog.f.T.dot(z_np) - bigM*(1.-d_dom[t]))[0,0]
#         phi_max = (prog.f.T.dot(z_np) + bigM*(1.-d_dom[t]))[0,0]
#         model.addConstr(phi >= phi_min)
#         model.addConstr(phi <= phi_max)

#         # compare costs
#         expr = (prog_i.f.T.dot(z_i_np) - phi)[0,0]
#         model.addConstr(expr <= 0.)

#         model.addConstr(x[0] >= -1.1)

#     # run optimization
#     model.setParam('OutputFlag', False)
#     model.optimize()

#     # return solution
#     if model.status == grb.GRB.Status.OPTIMAL:
#         print model.getAttr('x', d_dom)
#         print model.getAttr('x', x)
#         candidate_fs.append(i)
#         print 'candidate:', i

# print('Candidate feasible sets:' + str(candidate_fs))

# plt.show()