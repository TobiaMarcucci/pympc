clear
clc
syms c
assume(c, 'real')
assume(c > 0)

%% control in relative velocity 1
% d_qf = u
% dd_qb = - c d_qb - c u
% x = [qf, qb, d_qb]^T
A = [0 0 0; 0 0 1; 0 0 -c];
B = [1; 0; -c];
R = [B, A*B, A*A*B];
T_R = simplify(orth(R));
T_N1 = null(R')/norm(null(R'));
T = [T_R, T_N1];
A_sf = simplify(inv(T)*A*T);
B_sf = simplify(inv(T)*B);

double(subs(T(3,:)/norm(T(3,:)), c, 1))

%% control in relative velocity 2
% d_qf = d_qb + u
% dd_qb = - c d_qb - c u
% x = [qf, qb, d_qb]^T
A = [0 0 1; 0 0 1; 0 0 -c];
B = [1; 0; -c];
R = [B, A*B, A*A*B]
T_R = simplify(orth(R));
T_N = null(R')/norm(null(R'));
T = [T_R, T_N];
A_sf = simplify(inv(T)*A*T);
B_sf = simplify(inv(T)*B);

%% transform the previous

T12 = [1 1 0; 0 1 0; 0 0 1];

A1 = inv(T12)*A*T12;
B1 = inv(T12)*B;
R1 = inv(T12)*R;

%R2' = R1 ' T12'

%R2' T2_N = 0
%R1' * T12' * T2_N = 0

T3 = T12'*T_N;
T12'*T_N
T_N1
double(subs(T3/norm(T3), c, 1))

%% control in absolute velocity
% d_qf = u
% dd_qb = - c u
% x = [qf, qb, d_qb]^T
A = [0 0 0; 0 0 1; 0 0 0];
B = [1; 0; -c];
R = [B, A*B, A*A*B];
T_R = simplify(orth(R));
T_N = null(R')/norm(null(R'));
T = [T_R, T_N];
A_sf = simplify(inv(T)*A*T);
B_sf = simplify(inv(T)*B);