import numpy as np
from copy import copy
from scipy.spatial import Voronoi
from copy import copy

# drake
from pydrake.all import (FirstOrderTaylorApproximation,
    BasicVector,
    VectorSystem,
    MathematicalProgram,
    SolutionResult)
from pydrake.solvers.gurobi import GurobiSolver

# pympc
from pympc.geometry.polyhedron import Polyhedron
from pympc.dynamics.discrete_time_systems import LinearSystem, AffineSystem, PieceWiseAffineSystem
from pympc.optimization.programs import linear_program

def _voronoi_nd(points):
    """
    Given a list of n-dimensional points, returns the Voronoi partition of the space as a list of Polyhedron (pympc class).
    Uses the scipy wrapper of Voronoi from Qhull.
    Does not work in case of 1-dimensional points.

    Arguments
    ----------
    points : list of numpy.ndarray
        Points for the Voronoi partition.

    Returns
    ----------
    partition : list of Polyhedron
        Halfspace representation of all the cells of the partiotion.
    """

    # loop over points
    partition = []
    for i, point in enumerate(points):

        # h-rep of the cell for the point i
        A = []
        b = []

        # loop over the points that share a facet with point i
        for ridge in Voronoi(points).ridge_points:
            if i in ridge:

                # vector from point i to the neighbor
                bottom = point
                tip = points[ridge[1 - ridge.tolist().index(i)]]

                # hyperplane that separates point i and its neighbor
                Ai = tip - point
                center = bottom + (tip - bottom) / 2.
                bi = Ai.dot(center)
                A.append(Ai)
                b.append(bi)

        # assemble cell i and add to partition
        cell = Polyhedron(np.vstack(A), np.array(b))
        cell.normalize()
        partition.append(cell)

    return partition

def _voronoi_1d(points):
    """
    Given a list of 1-dimensional points, returns the Voronoi partition of the space as a list of Polyhedron (pympc class).

    Arguments
    ----------
    points : list of numpy.ndarray
        Points for the Voronoi partition.

    Returns
    ----------
    partition : list of Polyhedron
        Halfspace representation of all the cells of the partiotion.
    """

    # order point from smallest to biggest
    points = sorted(points)

    # loop from the smaller point
    polyhedra = []
    for i, point in enumerate(points):

        # h-rep of the cell for the point i
        A = []
        b = []

        # get previous and next point (if any)
        tips = []
        if i > 0:
            tips.append(points[i-1])
        if i < len(points)-1:
            tips.append(points[i+1])


        # vector from point i to the next/previous point
        for tip in tips:
            bottom = point
            center = bottom + (tip - bottom) / 2.

            # hyperplane that separates point i and its neighbor
            Ai = tip - point
            bi = Ai.dot(center)
            A.append(Ai)
            b.append(bi)

        # assemble cell i and add to partition
        polyhedron = Polyhedron(np.vstack(A), np.array(b))
        polyhedron.normalize()
        polyhedra.append(polyhedron)

    return polyhedra

def constrained_voronoi(points, X=None):
    """
    Given a list of n-dimensional points, returns the Voronoi partition of the Polyhedron X as a list of Polyhedron.
    If X is None, returns the partition of the whole space.

    Arguments
    ----------
    points : list of numpy.ndarray
        Points for the Voronoi partition.
    X : Polyhedron
        Set we want to partition.

    Returns
    ----------
    partition : list of Polyhedron
        Halfspace representation of all the cells of the partiotion.
    """

    # get indices of non-coincident coordinates
    nx = min(p.size for p in points)
    assert nx == max(p.size for p in points)
    indices = [i for i in range(nx) if not np.isclose(min([p[i] for p in points]), max([p[i] for p in points]))]

    # get voronoi partition without boundaries
    points_lower_dimensional = [p[indices] for p in points]
    if len(indices) == 1:
        vor = _voronoi_1d(points_lower_dimensional)
    else:
        vor = _voronoi_nd(points_lower_dimensional)

    # go back to the higher dimensional space
    partition = [Polyhedron(np.zeros((0,nx)), np.zeros(0)) for i in points]
    for i, cell in enumerate(vor):
        partition[i].add_inequality(cell.A, cell.b, indices=indices)

        # intersect with X is provided
        if X is not None:
            partition[i].add_inequality(X.A, X.b)

    return partition

def pwa_from_RigidBodyPlant(plant, linearization_points, X, U, h, method='zero_order_hold'):
    """

    Arguments
    ----------
    plant : RigidBodyPlant
        RigidBodyPlant of the robot.
    linearization_points : list of numpy.ndarray
        Points in the state space where to linearize the dynamics.
    X : Polyhedron
        Overall bounds of the state space.
    U : Polyhedron
        Overall bounds of the control space.
    h : float
        Sampling time for the time discretization of the PWA system.
    method : string
        Method used to discretize each piece of the PWA dynamics ('explicit_euler' or 'zero_order_hold').

    Returns
    ----------
    PWA : PieceWiseAffineSystem
    """

    # partition of the state space
    X_partition = constrained_voronoi(linearization_points, X)
    domains = [Xi.cartesian_product(U) for Xi in X_partition]

    # create context
    context = plant.CreateDefaultContext()
    state = context.get_mutable_continuous_state_vector()
    input = BasicVector(np.array([0.]))
    context.FixInputPort(0, input)

    # affine systems
    affine_systems = []
    for x in linearization_points:
        state.SetFromVector(x)
        taylor_approx = FirstOrderTaylorApproximation(plant, context)
        affine_system = AffineSystem.from_continuous(
            taylor_approx.A(),
            taylor_approx.B(),
            taylor_approx.f0(),
            h,
            method
            )
        affine_systems.append(affine_system)

    return PieceWiseAffineSystem(affine_systems, domains)

class Controller(VectorSystem):
    """
    Wrapper for the HybridModelPredictiveController class from pympc.
    """

    def __init__(self, S, N, Q, R, P, X_N):
        """
        Arguments
        ----------
        S : PieceWiseAffineSystem
            PWA system to be controlled.
        N : int
            Horizon of the optimal control problem.
        Q : numpy.ndarray
            Quadratic cost for the state.
        R : numpy.ndarray
            Quadratic cost for the input.
        P : numpy.ndarray
            Quadratic cost for the terminal state.
        X_N : Polyhedron
            Terminal set.
        """

        # bouild drake controller
        VectorSystem.__init__(self, S.nx, S.nu)
        self.controller = HybridModelPredictiveController(S, N, Q, R, Q, X_N)

    def _DoCalcVectorOutput(self, context, plant_state, unused, plant_input):
        print('Controller called at time ' + str(context.get_time()) + ' with state ' + str(plant_state) + '          \r'),
        plant_input[:] = self.controller.feedback(plant_state)

class HybridModelPredictiveController(object):

    def __init__(self, S, N, Q, R, P, X_N):

        # store inputs
        self.S = S
        self.N = N
        self.Q = Q
        self.R = R
        self.P = P
        self.X_N = X_N

        # mpMIQP
        self.build_mpmiqp()

    def build_mpmiqp(self):

        # express the constrained dynamics as a list of polytopes in the (x,u,x+)-space
        P = graph_representation(self.S)
        m = big_m(P)

        # initialize program
        self.prog = MathematicalProgram()
        self.x = []
        self.u = []
        self.d = []
        obj = 0.
        self.binaries_lower_bound = []

        # initial conditions (set arbitrarily to zero in the building phase)
        self.x.append(self.prog.NewContinuousVariables(self.S.nx))
        self.initial_condition = []
        for k in range(self.S.nx):
            self.initial_condition.append(self.prog.AddLinearConstraint(self.x[0][k] == 0.).evaluator())

        # loop over time
        for t in range(self.N):

            # create input, mode and next state variables
            self.u.append(self.prog.NewContinuousVariables(self.S.nu))
            self.d.append(self.prog.NewBinaryVariables(self.S.nm))
            self.x.append(self.prog.NewContinuousVariables(self.S.nx))
            
            # enforce constrained dynamics (big-m methods)
            xux = np.concatenate((self.x[t], self.u[t], self.x[t+1]))
            for i in range(self.S.nm):
                mi_sum = np.sum([m[i][j] * self.d[t][j] for j in range(self.S.nm) if j != i], axis=0)
                for k in range(P[i].A.shape[0]):
                    self.prog.AddLinearConstraint(P[i].A[k].dot(xux) <= P[i].b[k] + mi_sum[k])

            # SOS1 on the binaries
            self.prog.AddLinearConstraint(sum(self.d[t]) == 1.)

            # stage cost to the objective
            obj += .5 * self.u[t].dot(self.R).dot(self.u[t])
            obj += .5 * self.x[t].dot(self.Q).dot(self.x[t])

        # terminal constraint
        for k in range(self.X_N.A.shape[0]):
            self.prog.AddLinearConstraint(self.X_N.A[k].dot(self.x[self.N]) <= self.X_N.b[k])

        # terminal cost
        obj += .5 * self.x[self.N].dot(self.P).dot(self.x[self.N])
        self.objective = self.prog.AddQuadraticCost(obj)

        # set solver
        self.solver = GurobiSolver()
        self.prog.SetSolverOption(self.solver.solver_type(), 'OutputFlag', 1)


    def set_initial_condition(self, x0):
        for k, c in enumerate(self.initial_condition):
            c.UpdateLowerBound(x0[k:k+1])
            c.UpdateUpperBound(x0[k:k+1])

    def feedforward(self, x0):

        # overwrite initial condition
        self.set_initial_condition(x0)

        # solve MIQP
        result = self.solver.Solve(self.prog)

        # check feasibility
        if result != SolutionResult.kSolutionFound:
            return None, None, None, None

        # get cost
        obj = self.prog.EvalBindingAtSolution(self.objective)[0]

        # store argmin in list of vectors
        u = [self.prog.GetSolution(ut) for ut in self.u]
        x = [self.prog.GetSolution(xt) for xt in self.x]
        d = [self.prog.GetSolution(dt) for dt in self.d]

        # retrieve mode sequence and check integer feasibility
        ms = [np.argmax(dt) for dt in d]

        return u, x, ms, obj


    def feedback(self, x0):

        # get feedforward and extract first input
        u_feedforward = self.feedforward(x0)[0]
        if u_feedforward is None:
            return None

        return u_feedforward[0]

def graph_representation(S):
    '''
    For the PWA system S
    x+ = Ai x + Bi u + ci if Fi x + Gi u <= hi,
    returns the graphs of the dynamics (list of Polyhedron)
    [ Fi  Gi  0] [ x]    [ hi]
    [ Ai  Bi -I] [ u] <= [-ci]
    [-Ai -Bi  I] [x+]    [ ci]
    '''
    P = []
    for i in range(S.nm):
        Di = S.domains[i]
        Si = S.affine_systems[i]
        Ai = np.vstack((
            np.hstack((Di.A, np.zeros((Di.A.shape[0], S.nx)))),
            np.hstack((Si.A, Si.B, -np.eye(S.nx))),
            np.hstack((-Si.A, -Si.B, np.eye(S.nx))),
            ))
        bi = np.concatenate((Di.b, -Si.c, Si.c))
        P.append(Polyhedron(Ai, bi))
    return P

def big_m(P_list, tol=1.e-6):
    '''
    For the list of Polyhedron P_list in the from Pi = {x | Ai x <= bi} returns a list of lists of numpy arrays, where m[i][j] := max_{x in Pj} Ai x - bi.

    '''
    m = []
    for i, Pi in enumerate(P_list):
        mi = []
        for j, Pj in enumerate(P_list):
            mij = []
            for k in range(Pi.A.shape[0]):
                sol = linear_program(-Pi.A[k], Pj.A, Pj.b)
                mijk = - sol['min'] - Pi.b[k]
                if np.abs(mijk) < tol:
                    mijk = 0.
                mij.append(mijk)
            mi.append(np.array(mij))
        m.append(mi)
    return m