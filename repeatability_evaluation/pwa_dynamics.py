# external imports
import numpy as np
import sympy as sp

# internal imports
from pympc.geometry.polyhedron import Polyhedron
from pympc.dynamics.discrete_time_systems import LinearSystem, AffineSystem, PieceWiseAffineSystem
from pympc.control.hscc.controllers import HybridModelPredictiveController
import numeric_parameters as params

# symbolic state
xb, yb, tb = sp.symbols('xb yb tb') # position of the ball
xf, yf = sp.symbols('xf yf') # position of the floor
xdb, ydb, tdb = sp.symbols('xdb ydb tdb') # velocity of the ball
xdf, ydf = sp.symbols('xdf ydf') # velocity of the floor
x = sp.Matrix([
    xb, yb, tb,
    xf, yf,
    xdb, ydb, tdb,
    xdf, ydf
])

# symbolic input
xd2f, yd2f = sp.symbols('xd2f yd2f') # acceleration of the floor
u = sp.Matrix([
    xd2f, yd2f
])

# symbolic contact forces
ftf, fnf = sp.symbols('ftf fnf') # floor force
ftc, fnc = sp.symbols('ftc fnc') # ceiling force

# symbolic ball velocity update
xdb_next = xdb + params.h*ftf/params.m - params.h*ftc/params.m
ydb_next = ydb + params.h*fnf/params.m - params.h*fnc/params.m - params.h*params.g
tdb_next = tdb + params.r*params.h*ftf/params.j + params.r*params.h*ftc/params.j

# symbolic ball position update
xb_next = xb + params.h*xdb_next
yb_next = yb + params.h*ydb_next
tb_next = tb + params.h*tdb_next

# symbolic floor velocity update
xdf_next = xdf + params.h*xd2f
ydf_next = ydf + params.h*yd2f

# symbolic floor position update
xf_next = xf + params.h*xdf_next
yf_next = yf + params.h*ydf_next

# symbolic state update
x_next = sp.Matrix([
    xb_next, yb_next, tb_next,
    xf_next, yf_next,
    xdb_next, ydb_next, tdb_next,
    xdf_next, ydf_next
])

# symbolic relative tangential velocity (ball wrt floor and ceiling)
sliding_velocity_floor = xdb_next + params.r*tdb_next - xdf_next
sliding_velocity_ceiling = xdb_next - params.r*tdb_next

# symbolic gap functions (ball wrt floor and ceiling)
gap_floor = yb_next - yf_next
gap_ceiling = params.d - 2.*params.r - yb_next

# symbolic ball distance to floor and ceiling boundaries
ball_on_floor = sp.Matrix([
    xb_next - xf_next - params.l,
    xf_next - xb_next - params.l
])
ball_on_ceiling = sp.Matrix([
    xb_next - params.l,
    - xb_next - params.l
])

# state + input bounds
xu = x.col_join(u)
xu_min = np.concatenate((params.x_min, params.u_min))
xu_max = np.concatenate((params.x_max, params.u_max))

# discrete time dynamics in mode 1
# (ball in the air)

# set forces to zero
f_m1 = {ftf: 0., fnf: 0., ftc: 0., fnc: 0.}

# get dynamics
x_next_m1 = x_next.subs(f_m1)
S1 = AffineSystem.from_symbolic(x, u, x_next_m1)

# build domain
D1 = Polyhedron.from_bounds(xu_min, xu_max)

# - gap <= 0 with floor and ceiling
gap_floor_m1 = gap_floor.subs(f_m1)
gap_ceiling_m1 = gap_ceiling.subs(f_m1)
D1.add_symbolic_inequality(xu, sp.Matrix([- gap_floor_m1]))
D1.add_symbolic_inequality(xu, sp.Matrix([- gap_ceiling_m1]))

# check domain
assert D1.bounded
assert not D1.empty

# discrete time dynamics in mode 2
# (ball sticking with the floor, not in contact with the ceiling)

# enforce sticking
fc_m2 = {ftc: 0., fnc: 0.}
ftf_m2 = sp.solve(sp.Eq(sliding_velocity_floor.subs(fc_m2), 0), ftf)[0]
fnf_m2 = sp.solve(sp.Eq(gap_floor.subs(fc_m2), 0), fnf)[0]
f_m2 = fc_m2.copy()
f_m2.update({ftf: ftf_m2, fnf: fnf_m2})

# get dynamics
x_next_m2 = x_next.subs(f_m2)
S2 = AffineSystem.from_symbolic(x, u, x_next_m2)

# build domain
D2 = Polyhedron.from_bounds(xu_min, xu_max)

# gap <= 0 with floor
D2.add_symbolic_inequality(xu, sp.Matrix([gap_floor_m1]))

# - gap <= 0 with ceiling
D2.add_symbolic_inequality(xu, sp.Matrix([- gap_ceiling_m1]))

# ball not falling down the floor
D2.add_symbolic_inequality(xu, ball_on_floor.subs(f_m2))

# friction cone
D2.add_symbolic_inequality(xu, sp.Matrix([ftf_m2 - params.mu*fnf_m2]))
D2.add_symbolic_inequality(xu, sp.Matrix([- ftf_m2 - params.mu*fnf_m2]))

# check domain
assert D2.bounded
assert not D2.empty

# discrete time dynamics in mode 3
# (ball sliding right on the floor, not in contact with the ceiling)

# enforce sticking
f_m3 = {ftf: -params.mu*fnf_m2, fnf: fnf_m2, ftc: 0., fnc: 0.}

# get dynamics
x_next_m3 = x_next.subs(f_m3)
S3 = AffineSystem.from_symbolic(x, u, x_next_m3)

# build domain
D3 = Polyhedron.from_bounds(xu_min, xu_max)

# gap <= 0 with floor
D3.add_symbolic_inequality(xu, sp.Matrix([gap_floor_m1]))

# - gap <= 0 with ceiling
D3.add_symbolic_inequality(xu, sp.Matrix([- gap_ceiling_m1]))

# ball not falling down the floor
D3.add_symbolic_inequality(xu, ball_on_floor.subs(f_m3))

# positive relative velocity
D3.add_symbolic_inequality(xu, sp.Matrix([- sliding_velocity_floor.subs(f_m3)]))

# check domain
assert D3.bounded
assert not D3.empty

# discrete time dynamics in mode 4
# (ball sliding left on the floor, not in contact with the ceiling)

# enforce sticking
f_m4 = {ftf: params.mu*fnf_m2, fnf: fnf_m2, ftc: 0., fnc: 0.}

# get dynamics
x_next_m4 = x_next.subs(f_m4)
S4 = AffineSystem.from_symbolic(x, u, x_next_m4)

# build domain
D4 = Polyhedron.from_bounds(xu_min, xu_max)

# gap <= 0 with floor
D4.add_symbolic_inequality(xu, sp.Matrix([gap_floor_m1]))

# - gap <= 0 with ceiling
D4.add_symbolic_inequality(xu, sp.Matrix([- gap_ceiling_m1]))

# ball not falling down the floor
D4.add_symbolic_inequality(xu, ball_on_floor.subs(f_m4))

# negative relative velocity
D4.add_symbolic_inequality(xu, sp.Matrix([sliding_velocity_floor.subs(f_m4)]))

# check domain
assert D4.bounded
assert not D4.empty

# discrete time dynamics in mode 5
# (ball sticking on the ceiling, not in contact with the floor)

# enforce sticking
ff_m5 = {ftf: 0., fnf: 0.}
ftc_m5 = sp.solve(sp.Eq(sliding_velocity_ceiling.subs(ff_m5), 0), ftc)[0]
fnc_m5 = sp.solve(sp.Eq(gap_ceiling.subs(ff_m5), 0), fnc)[0]
f_m5 = ff_m5.copy()
f_m5.update({ftc: ftc_m5, fnc: fnc_m5})

# get dynamics
x_next_m5 = x_next.subs(f_m5)
S5 = AffineSystem.from_symbolic(x, u, x_next_m5)

# build domain
D5 = Polyhedron.from_bounds(xu_min, xu_max)

# - gap <= 0 with floor
D5.add_symbolic_inequality(xu, sp.Matrix([- gap_floor_m1]))

# gap <= 0 with ceiling
D5.add_symbolic_inequality(xu, sp.Matrix([gap_ceiling_m1]))

# ball in contact with the ceiling
D5.add_symbolic_inequality(xu, ball_on_ceiling.subs(f_m5))

# friction cone
D5.add_symbolic_inequality(xu, sp.Matrix([ftc_m5 - params.mu*fnc_m5]))
D5.add_symbolic_inequality(xu, sp.Matrix([- ftc_m5 - params.mu*fnc_m5]))

# check domain
assert D5.bounded
assert not D5.empty

# discrete time dynamics in mode 6
# (ball sliding right on the ceiling, not in contact with the floor)

# enforce sticking
f_m6 = {ftc: -params.mu*fnc_m5, fnc: fnc_m5, ftf: 0., fnf: 0.}

# get dynamics
x_next_m6 = x_next.subs(f_m6)
S6 = AffineSystem.from_symbolic(x, u, x_next_m6)

# build domain
D6 = Polyhedron.from_bounds(xu_min, xu_max)

# - gap <= 0 with floor
D6.add_symbolic_inequality(xu, sp.Matrix([- gap_floor_m1]))

# gap <= 0 with ceiling
D6.add_symbolic_inequality(xu, sp.Matrix([gap_ceiling_m1]))

# ball in contact with the ceiling
D6.add_symbolic_inequality(xu, ball_on_ceiling.subs(f_m6))

# positive relative velocity
D6.add_symbolic_inequality(xu, sp.Matrix([- sliding_velocity_ceiling.subs(f_m6)]))

# check domain
assert D6.bounded
assert not D6.empty

# discrete time dynamics in mode 7
# (ball sliding left on the ceiling, not in contact with the floor)

# enforce sticking
f_m7 = {ftc: params.mu*fnc_m5, fnc: fnc_m5, ftf: 0., fnf: 0.}

# get dynamics
x_next_m7 = x_next.subs(f_m7)
S7 = AffineSystem.from_symbolic(x, u, x_next_m7)

# build domain
D7 = Polyhedron.from_bounds(xu_min, xu_max)

# - gap <= 0 with floor
D7.add_symbolic_inequality(xu, sp.Matrix([- gap_floor_m1]))

# gap <= 0 with ceiling
D7.add_symbolic_inequality(xu, sp.Matrix([gap_ceiling_m1]))

# ball in contact with the ceiling
D7.add_symbolic_inequality(xu, ball_on_ceiling.subs(f_m7))

# negative relative velocity
D7.add_symbolic_inequality(xu, sp.Matrix([sliding_velocity_ceiling.subs(f_m7)]))

# check domain
assert D7.bounded
assert not D7.empty

# list of dynamics
S_list = [S1, S2, S3, S4, S5, S6, S7]

# list of domains
D_list = [D1, D2, D3, D4, D5, D6, D7]

# PWA system
S = PieceWiseAffineSystem(S_list, D_list)