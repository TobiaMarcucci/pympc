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
X_0 = Polytope.from_bounds(x_min_0, x_max_0)
X_0.assemble()

x_min_1 = x_max_0
x_max_1 =  - x_max_0
X_1 = Polytope.from_bounds(x_min_1, x_max_1)
X_1.assemble()

x_min_2 = x_max_1
x_max_2 = - x_min_0
X_2 = Polytope.from_bounds(x_min_2, x_max_2)
X_2.assemble()

u_max = np.array([[10.]])
u_min = -u_max

U_0 = Polytope.from_bounds(u_min, u_max)
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
objective_norm = 'two'
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
    x_samples = list(np.linspace(fs.vertices[0][0,0], fs.vertices[1][0,0], n_samples))
    x_samples.sort()
    V_samples = []
    for j, x in enumerate(x_samples):
        V = prog.solve(np.array([[x]]))[1]
        V_samples.append(V)
        if j == int(n_samples/2):
            plt.text(x, V, str(i))
    plt.plot(x_samples, V_samples, color=col)

    x_list = [x_samples[2], x_samples[n_samples-2]]
    V_list = [V_samples[2], V_samples[n_samples-2]]
    _, x_min, V_min = prog.solve_free_x()
    x_min = x_min[0,0]
    if x_min > min(x_list) and x_min < max(x_list):
        x_list.insert(1, x_min)
        V_list.insert(1, V_min)
    T_samples = []
    A_list = []
    b_list = []
    print x_list
    for x in x_list:
        A, b = prog.get_cost_sensitivity(np.array([[x]]))
        A_list.append(A)
        b_list.append(b)
    for x in x_samples:
        sample = max([(A_list[i].dot(x) + b_list[i])[0,0] for i in range(len(x_list))])
        T_samples.append(sample)
    plt.plot(x_list, V_list, color=col, linestyle='--')
    plt.plot(x_samples, T_samples, color=col, linestyle='--')






# # single feasibility regions

# fs_single_feasibility = []
# fs_multiple_feasibility = []

# n_x = 1
# n_u = 1
# tic = time.time()
# for i, fs in enumerate(fs_list):
#     other_fs = [fs_j for j, fs_j in enumerate(fs_list) if j != i]
#     if fs.included_in_union_of(other_fs):
#         fs_multiple_feasibility.append(i)
#     else:
#         fs_single_feasibility.append(i)
# print('Single feasibility sets ' + str(fs_single_feasibility) + ' computed in ' + str(time.time() - tic) + ' s')

plt.show()
