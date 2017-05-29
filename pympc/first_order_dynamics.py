import numpy as np
import matplotlib.pyplot as plt
import dynamical_systems as ds
from control import MPCHybridController, HybridPolicyLibrary
from geometry import Polytope
import plot as mpc_plt
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

# state domains

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

X = [X_0, X_1, X_2]

# inoput domains

u_max = np.array([[5.]])
u_min = -u_max

U_0 = Polytope.from_bounds(u_min, u_max)
U_0.assemble()

U_1 = U_0
U_2 = U_0

U = [U_0, U_1, U_2]

# pwa system

pwa_sys = ds.DTPWASystem(sys, X, U)

# controller

N = 10
Q = np.eye(A_1.shape[0])
R = np.eye(B_1.shape[1])
P = Q
objective_norm = 'two'

P, K = ds.dare(sys_1.A, sys_1.B, Q, R)
X_N = ds.moas_closed_loop(sys_1.A, sys_1.B, K, X_1, U_1)

controller = MPCHybridController(pwa_sys, N, objective_norm, Q, R, P, X_N)

# initiliazation library

n_samples = 100
policy = HybridPolicyLibrary(controller)
policy.sample_policy_randomly(n_samples)

# add shifted sequences

terminal_domain = 1
policy.add_shifted_sequences(terminal_domain)

# upper bounds

policy.add_vertices_of_feasible_regions()
policy.bound_cost_from_above()

# lower bounds

for ss, ss_values in policy.library.items():
    for as_values in ss_values['active_sets'].values():
        for i, x in enumerate(as_values['x']):
            if as_values['optimal'][i]:
                fss = policy.feasible_switching_sequences(x)
                fss.remove(ss)
                for ss_lb in fss:
                    lb = policy.get_lower_bound(ss_lb, x)
                    if lb < as_values['V'][i]:
                        policy.sample_policy([x], ss_lb, True)

### plot

# sample hybrid optuimal value function

x_lb = x_min_0
x_ub = x_max_2
n_samples = 100
x_samples = list(np.linspace(x_lb[0,0], x_ub[0,0], n_samples))
V_samples = [controller.feedforward(np.array([[x]]))[1] for x in x_samples]
plt.plot(x_samples, V_samples, color='black', linewidth=3)

# bounds

for ss, book in policy.library.items():

    # x samples
    col = np.random.rand(3,1)
    vertices = sorted([vertex[0,0] for vertex in book['feasible_set'].vertices])
    x_samples = list(np.linspace(vertices[0], vertices[1], n_samples))

    # sample real value function
    V_samples = [book['program'].solve(np.array([[x]]))[1] for x in x_samples]
    plt.plot(x_samples, V_samples, color=col)

    # sample bounds
    lb_samples = [policy.get_lower_bound(ss, x) for x in x_samples]
    ub_samples = [policy.get_upper_bound(ss, x) for x in x_samples]
    plt.plot(x_samples, lb_samples, color=col, linestyle='--')
    plt.plot(x_samples, ub_samples, color=col, linestyle='-.')

# plot samples

for ss_values in policy.library.values():
    for as_values in ss_values['active_sets'].values():
        for i, x in enumerate(as_values['x']):
            if as_values['optimal'][i]:
                plt.scatter(x[0,0], as_values['V'][i], facecolors='none', edgecolors='r')

plt.show()

# simulate closed loop

N_sim = 100
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

mpc_plt.input_sequence(u, t_s, N_sim, (u_min, u_max))
plt.show()
mpc_plt.state_trajectory(x, t_s, N_sim, (x_min_0, x_min_1, x_min_2, x_max_2))
plt.show()