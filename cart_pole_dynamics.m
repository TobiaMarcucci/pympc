clear
clc

% cart-pole hybrid dynamics
% mode 0 -> if: pp(1) < -d, for all u
% mode 1 -> if: -d <= pp(1) <= d, for all u
% mode 2 -> if: pp(1) > d, for all u

% symbolic variables
syms q1 q2 qd1 qd2 u mc mp l d k g
assume([q1, q2, qd1, qd2, u, mc, mp, l, d, k, g], 'real')

% numeric parameter
mc_n = 1;
mp_n = 1;
l_n = 1;
d_n = 1;
k_n = 100;
g_n = 10;
par = [mc, mp, l, d, k, g];
par_n = [mc_n, mp_n, l_n, d_n, k_n, g_n];

% equilibrium point
x_eq = [0; 0; 0; 0];
u_eq = 0;

% gravity
gr = [0; -g];

% state
q = [q1; q2];
qd = [qd1; qd2];
x = [q; qd];

% positions
pc = [q1; 0];
pp = [q1-l*sin(q2); l*cos(q2)];

% velocities
vc = jacobian(pc, q)*qd;
vp = jacobian(pp, q)*qd;

% lagrangian functions
T = .5*mc*(vc'*vc) + .5*mp*(vp'*vp);
U0 = - mp*(gr'*pp) + .5*k*(pp(1) + d)^2;
U1 = - mp*(gr'*pp);
U2 = - mp*(gr'*pp) + .5*k*(pp(1) - d)^2;
L0 = T - U0;
L1 = T - U1;
L2 = T - U2;

% generalized forces
f = [u; 0];
Q = (f'*jacobian(pc, q))';

% equations of motion
M0 = jacobian(jacobian(L0, qd)', qd);
M1 = jacobian(jacobian(L1, qd)', qd);
M2 = jacobian(jacobian(L2, qd)', qd);
C0 = jacobian(jacobian(L0, qd)', q)*qd - jacobian(L0, q)';
C1 = jacobian(jacobian(L1, qd)', q)*qd - jacobian(L1, q)';
C2 = jacobian(jacobian(L2, qd)', q)*qd - jacobian(L2, q)';

% state space form
f1 = simplify([qd; M1\(Q-C1)]);
f2 = simplify([qd; M2\(Q-C2)]);
f0 = simplify([qd; M0\(Q-C0)]);

% symbolic linearization
var = [x', u'];
var_n = [x_eq', u_eq'];
A0 = subs(jacobian(f0, x), var, var_n);
A1 = subs(jacobian(f1, x), var, var_n);
A2 = subs(jacobian(f2, x), var, var_n);
B0 = subs(jacobian(f0, u), var, var_n);
B1 = subs(jacobian(f1, u), var, var_n);
B2 = subs(jacobian(f2, u), var, var_n);
c0 = subs(f0, var, var_n);
c1 = subs(f1, var, var_n);
c2 = subs(f2, var, var_n);

% numeric linearization
var = [par, x', u'];
var_n = [par_n, x_eq', u_eq'];
A0_n = subs(jacobian(f0, x), var, var_n);
A1_n = subs(jacobian(f1, x), var, var_n);
A2_n = subs(jacobian(f2, x), var, var_n);
B0_n = subs(jacobian(f0, u), var, var_n);
B1_n = subs(jacobian(f1, u), var, var_n);
B2_n = subs(jacobian(f2, u), var, var_n);
c0_n = subs(f0, var, var_n);
c1_n = subs(f1, var, var_n);
c2_n = subs(f2, var, var_n);
