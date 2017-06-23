import numpy as np
import matplotlib.pyplot as plt
import dynamical_systems as ds
from control import MPCHybridController, HybridPolicyLibrary
from geometry import Polytope
import plot as mpc_plt
import time

# dynamics

t_s = .1

A_1 = np.array([[-1.]])
B_1 = np.array([[1.]])
c_1 = np.array([[-2.]])
sys_1 = ds.DTAffineSystem.from_continuous(A_1, B_1, c_1, t_s)

A_2 = np.array([[1.]])
B_2 = np.array([[1.]])
c_2 = np.array([[0.]])
sys_2 = ds.DTAffineSystem.from_continuous(A_2, B_2, c_2, t_s)

A_3 = np.array([[-1.]])
B_3 = np.array([[1.]])
c_3 = np.array([[2.]])
sys_3 = ds.DTAffineSystem.from_continuous(A_3, B_3, c_3, t_s)

sys = [sys_1, sys_2, sys_3]

# state domains

x_min_1 = np.array([[-3.]])
x_max_1 = np.array([[-1.]])
X_1 = Polytope.from_bounds(x_min_1, x_max_1)
X_1.assemble()

x_min_2 = x_max_1
x_max_2 =  - x_max_1
X_2 = Polytope.from_bounds(x_min_2, x_max_2)
X_2.assemble()

x_min_3 = x_max_2
x_max_3 = - x_min_1
X_3 = Polytope.from_bounds(x_min_3, x_max_3)
X_3.assemble()

X = [X_1, X_2, X_3]

# inoput domains

u_max = np.array([[5.]])
u_min = -u_max

U_1 = Polytope.from_bounds(u_min, u_max)
U_1.assemble()

U_2 = U_1
U_3 = U_1

U = [U_1, U_2, U_3]

# pwa system

pwa_sys = ds.DTPWASystem.from_orthogonal_domains(sys, X, U)

# controller

N = 5
Q = np.eye(A_2.shape[0])
R = np.eye(B_2.shape[1])
P = Q
objective_norm = 'two'

P, K = ds.dare(sys_2.A, sys_2.B, Q, R)
X_N = ds.moas_closed_loop(sys_2.A, sys_2.B, K, X_2, U_2)

controller = MPCHybridController(pwa_sys, N, objective_norm, Q, R, P, X_N)

# initiliazation library

policy = HybridPolicyLibrary(controller)

# coverage

n_samples = 100
terminal_domain = 1
policy.sample_policy(n_samples, terminal_domain)

# bound optimal value functions

n_samples = 1000
policy.add_vertices_of_feasible_regions()
policy.bound_optimal_value_functions(n_samples)




### plot

# sample hybrid optimal value function

x_lb = x_min_1
x_ub = x_max_3
n_samples = 100
x_samples = list(np.linspace(x_lb[0,0], x_ub[0,0], n_samples))
V_samples = [controller.feedforward(np.array([[x]]))[1] for x in x_samples]
plt.plot(x_samples, V_samples, color='black', linewidth=3)

# bounds

for ss, ss_value in policy.library.items():

    # x samples
    col = np.random.rand(3,1)
    vertices = sorted([vertex[0,0] for vertex in ss_value['feasible_set'].vertices])
    x_samples = list(np.linspace(vertices[0], vertices[1], n_samples))

    # sample real value function
    V_samples = [ss_value['program'].solve(np.array([[x]]))[1] for x in x_samples]
    plt.plot(x_samples, V_samples, color=col)

    # sample bounds
    lb_samples = [policy.get_lower_bound(ss, x) for x in x_samples]
    ub_samples = [policy.get_upper_bound(ss, x) for x in x_samples]
    plt.plot(x_samples, lb_samples, color=col, linestyle='--')
    plt.plot(x_samples, ub_samples, color=col, linestyle='-.')

plt.show()


# simulate closed loop

N_sim = 50
x_0 = np.array([[2.9]])
u = []
x = []
u_miqp = []
my_times = []
miqp_times = []
x.append(x_0)
for k in range(N_sim):

    tic = time.time()
    u.append(policy.feedback(x[k]))
    my_times.append(time.time() - tic)

    tic = time.time()
    u_miqp.append(controller.feedback(x[k]))
    miqp_times.append(time.time() - tic)

    x_next = pwa_sys.simulate(x[k], [u[k]])[0][1]
    x.append(x_next)


print 'my times (min, max, mean):', min(my_times), max(my_times), np.mean(my_times)
print 'miqp times (min, max, mean):', min(miqp_times), max(miqp_times), np.mean(miqp_times)

print 'max feedback error', max([np.linalg.norm(u[i] - u_miqp[i]) for i in range(len(u))])

mpc_plt.input_sequence(u, t_s, (u_min, u_max))
plt.show()
mpc_plt.state_trajectory(x, t_s, (x_min_1, x_min_2, x_min_3, x_max_3))
plt.show()