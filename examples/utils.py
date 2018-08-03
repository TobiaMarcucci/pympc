import numpy as np
from copy import copy
from scipy.spatial import Voronoi
from copy import copy

from pydrake.all import (DiagramBuilder,
                         RigidBodyTree,
                         FloatingBaseType,
                         RigidBodyPlant,
                         SignalLogger,
                         Simulator,
                         FirstOrderTaylorApproximation,
                         BasicVector,
                         VectorSystem
                        )

from pympc.geometry.polyhedron import Polyhedron
from pympc.dynamics.discrete_time_systems import LinearSystem, AffineSystem, PieceWiseAffineSystem
from pympc.control.hybrid_benchmark.controllers import HybridModelPredictiveController

def voronoi_nd(points):
    vor = Voronoi(points)
    polyhedra = []
    for i, point in enumerate(points):
        A = []
        b = []
        for ridge in vor.ridge_points:
            if i in ridge:
                bottom = point
                tip = points[ridge[1 - ridge.tolist().index(i)]]
                Ai = tip - point
                center = bottom + (tip - bottom) / 2.
                bi = Ai.dot(center)
                A.append(Ai)
                b.append(bi)
        polyhedron = Polyhedron(np.vstack(A), np.vstack(b))
        polyhedron.normalize()
        polyhedra.append(polyhedron)
    return polyhedra

def voronoi_1d(points):
    points = sorted(points)
    polyhedra = []
    for i, point in enumerate(points):
        tips = []
        if i > 0:
            tips.append(points[i-1])
        if i < len(points)-1:
            tips.append(points[i+1])
        A = []
        b = []
        for tip in tips:
            bottom = point
            Ai = tip - point
            center = bottom + (tip - bottom) / 2.
            bi = Ai.dot(center)
            A.append(Ai)
            b.append(bi)
        polyhedron = Polyhedron(np.vstack(A), np.vstack(b))
        polyhedron.normalize()
        polyhedra.append(polyhedron)
    return polyhedra

def voronoi(points):
    nx = min(p.size for p in points)
    assert nx == max(p.size for p in points)
    if nx == 1:
        return voronoi_1d(points)
    else:
        return voronoi_nd(points)

def state_space_partition(X, x_linearization, indices=None):
    partition = [copy(X) for i in x_linearization]
    for i, p in enumerate(voronoi(x_linearization)):
        partition[i].add_inequality(p.A, p.b, indices)
    return partition

def pwa_from_RigidBodyPlant(plant, h, X, U, x_linearization, indices=None, method='zero_order_hold'):

    # parittion of the state space
    X_partition = state_space_partition(X, x_linearization, indices)
    domains = [Xi.cartesian_product(U) for Xi in X_partition]

    # create context
    context = plant.CreateDefaultContext()
    x = context.get_mutable_continuous_state_vector()
    u = BasicVector(np.array([0.]))
    context.FixInputPort(0, u)

    # affine systems
    affine_systems = []
    for x_nom in x_linearization:
        x_nom = np.array([x_nom[indices.index(i)] if i in indices else 0. for i in range(plant.get_num_states())])
        x.SetFromVector(x_nom)
        taylor_approx = FirstOrderTaylorApproximation(plant, context)
        affine_system = AffineSystem.from_continuous(
            taylor_approx.A(),
            taylor_approx.B(),
            np.vstack(taylor_approx.f0()),
            h,
            method
            )
        affine_systems.append(affine_system)

    return PieceWiseAffineSystem(affine_systems, domains)

class Controller(VectorSystem):

    def __init__(self, pwa, N, Q, R, P, X_N, method='Big-M'):

        # controller
        VectorSystem.__init__(self, pwa.nx, pwa.nu)
        self.controller = HybridModelPredictiveController(pwa, N, Q, R, Q, X_N, method=method)

    def _DoCalcVectorOutput(self, context, cart_pole_state, unused, cart_pole_input):
        print(str(context.get_time()) + ' - ' + str(cart_pole_state) + '.\r'),
        cart_pole_input[:] = self.controller.feedback(np.vstack(cart_pole_state)).flatten()
        # print self.controller.S.get_mode(np.vstack(cart_pole_state), np.vstack(cart_pole_input))