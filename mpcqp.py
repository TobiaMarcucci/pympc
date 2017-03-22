import itertools
import sympy
import numpy as np
import scipy.linalg as linalg
import pydrake.solvers.mathematicalprogram as mp
from mpc_tools import Polytope
from optimization import linear_program
import cdd


def extract_linear_equalities(prog):
    bindings = prog.linear_equality_constraints()
    C = np.zeros((len(bindings), prog.num_vars()))
    d = np.zeros(len(bindings))
    for (i, binding) in enumerate(bindings):
        constraint = binding.constraint()
        ci = np.zeros(prog.num_vars())
        assert constraint.upper_bound() == constraint.lower_bound()
        d[i] = constraint.upper_bound()
        ai = constraint.A()
        assert ai.shape[0] == 1
        ai = ai[0, :]
        for (j, var) in enumerate(binding.variables()):
            ci[prog.FindDecisionVariableIndex(var)] = ai[j]
        C[i, :] = ci
    return C, d


def extract_linear_inequalities(prog):
    bindings = itertools.chain(prog.linear_constraints(), prog.bounding_box_constraints())
    if not bindings:
        return np.zeros((0, prog.num_vars())), np.zeros(0)
    A = []
    b = []
    for (i, binding) in enumerate(bindings):
        constraint = binding.constraint()
        A_row = np.zeros(prog.num_vars())
        ai = constraint.A()
        assert ai.shape[0] == 1
        ai = ai[0, :]
        for (j, var) in enumerate(binding.variables()):
            A_row[prog.FindDecisionVariableIndex(var)] = ai[j]

        if constraint.upper_bound() != np.inf:
            A.append(A_row)
            b.append(constraint.upper_bound())
        if constraint.lower_bound() != -np.inf:
            A.append(-A_row)
            b.append(-constraint.lower_bound())
    return np.vstack(A), np.hstack(b)


def mpc_order(prog, u, x0):
    order = np.zeros(prog.num_vars())
    for (i, var) in enumerate(itertools.chain(u.flat, x0.flatten(order='F'))):
        order[prog.FindDecisionVariableIndex(var)] = i
    return np.argsort(order)


def eliminate_equality_constrained_variables(C, d):
    """
    Given C and d defining a set of linear equality constraints:

        C x == d

    find A, b such that x = A z + b implies C x == d implies for any z \subset x

    This is just a matter of taking an appropriate null basis of C.

    This allows us to rewrite a QP with equality constraints into a QP over
    fewer variables with no equality constraints.
    """
    if C.shape[0] == 0:
        assert d.size == 0
        A = np.eye(C.shape[1])
        b = np.zeros(C.shape[1])
        preserved = np.ones(C.shape[1], dtype=np.bool)
        return Affine(A, b), preserved

    c_rational = sympy.Matrix([[sympy.Rational(x) for x in row] for row in C])
    W = sympy.Matrix.hstack(*c_rational.nullspace())

    # Convert to column-echelon form. We do this in order to ensure that each
    # variable in the new optimization corresponds exactly to one variable in
    # the original optimization (rather than corresponding to some constant
    # times that variable). This makes it easier to verify that our model is
    # correct.
    W = W.T.rref()[0].T

    # Check which variables were preserved by looking for the leading ones
    preserved = np.zeros(C.shape[1], dtype=np.bool)
    for j in range(W.shape[1]):
        for i in range(W.shape[0]):
            if W[i, j] == 1:
                preserved[i] = True
                break

    A = np.asarray(W).astype(np.float64)
    b = np.linalg.lstsq(C, d)[0]
    assert b.size == A.shape[0]
    return Affine(A, b), preserved


def extract_objective(prog):
    num_vars = prog.num_vars()
    Q = np.zeros((num_vars, num_vars))
    q = np.zeros(num_vars)

    for binding in prog.linear_costs():
        var_order = [prog.FindDecisionVariableIndex(v) for v in binding.variables()]
        ai = binding.constraint().A()
        assert ai.shape[0] == 1
        ai = ai[0, :]
        for i in range(ai.size):
            q[var_order[i]] += ai[i]

    for binding in prog.quadratic_costs():
        var_order = [prog.FindDecisionVariableIndex(v) for v in binding.variables()]
        Qi = binding.constraint().Q()
        bi = binding.constraint().b()
        for i in range(bi.size):
            q[var_order[i]] += bi[i]
            for j in range(bi.size):
                Q[var_order[i], var_order[j]] += Qi[i, j]
    Q = 0.5 * (Q + Q.T)
    return Q, q

def permutation_matrix(order):
    """
    Returns a matrix P such that P * y = y[order]
    """
    P = np.zeros((len(order), len(order)))
    for i in range(len(order)):
        P[i, order[i]] = 1

    return P


class Affine(object):
    __slots__ = ["A", "b"]
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def copy(self):
        return Affine(self.A.copy(), self.b.copy())


class SimpleQuadraticProgram(object):
    """
    Represents a quadratic program of the form:

    minimize 0.5 y' H y + f' y
       y
    such that A y <= b
              C y == d

    Also stores an affine transform T to allow us to perform affine variable
    substitutions on the internal model without affecting the result. Calling
    solve() solves the inner optimization for y, then returns:
    x = T.A * y + T.b

    This class is meant to be used as a temporary container while performing
    variable substitutions and other transformations on the optimization
    problem.
    """
    def __init__(self, H, f, A, b, C=None, d=None, T=None):
        self.H = H
        self.f = f
        self.A = A
        self.b = b
        if C is None or d is None:
            assert C is None and d is None, "C and d must both be provided together"
            C = np.zeros((0, A.shape[1]))
            d = np.zeros(0)
        if T is None:
            T = Affine(np.eye(A.shape[1]), np.zeros(A.shape[1]))
        self.T = T
        self.C = C
        self.d = d

    @property
    def num_vars(self):
        return self.A.shape[1]

    @staticmethod
    def from_mathematicalprogram(prog):
        """
        Construct a simple quadratic program representation from a symbolic
        MathematicalProgram. Note that this destroys the sparsity pattern of
        the MathematicalProgram's constraints and costs.
        """
        assert SimpleQuadraticProgram.check_form(prog)
        C, d = extract_linear_equalities(prog)
        A, b = extract_linear_inequalities(prog)
        Q, q = extract_objective(prog)
        return SimpleQuadraticProgram(Q, q, A, b, C, d)

    @staticmethod
    def check_form(prog):
        # TODO:
        # * verify that all variables are continuous
        # * verify that all constraints are linear
        # * verify that all costs are linear or quadratic
        return True

    def to_mathematicalprogram(self):
        prog = mp.MathematicalProgram()
        y = prog.NewContinuousVariables(self.T.A.shape[1], "y")
        for i in range(self.A.shape[0]):
            prog.AddLinearConstraint(self.A[i, :].dot(y) <= self.b[i])
        for i in range(self.C.shape[0]):
            prog.AddLinearConstraint(self.C[i, :].dot(y) == self.d[i])
        prog.AddQuadraticCost(0.5 * y.dot(self.H).dot(y) + self.f.dot(y))
        return prog, y, self.T

    def solve(self):
        prog, y, T = self.to_mathematicalprogram()
        result = prog.Solve()
        assert result == mp.SolutionResult.kSolutionFound, "Optimization failed with: {}".format(result)
        return T.A.dot(prog.GetSolution(y)) + T.b

    def affine_variable_substitution(self, U, v):
        """
        Given an optimization of the form:
        minimize 0.5 z' H z + f' z
           z
        such that A z <= b
                  C z == d

        perform a variable substitution defined by:
            z = U y + v

        and return an equivalent optimization over y.

        Since we have x = T.A * z + T.b and z = U * y + v, we have
        x = T.A * U * y + T.A * v + T.b

        Note that the new optimization will have its internal affine transform
        set up to ensure that calling solve() still returns x rather than y.
        Thus, we should have:

            qp2 = qp1.affine_variable_substitution(U, v)
            qp2.solve() == qp1.solve()

        for any U and v, even though the internal matrices of qp2 will be
        different than qp1.
        """

        # 0.5 (U y + v)' H (U y + v) + f' (U y + v)
        # 0.5 ( y' U' H U y + y' U' H v + v' H U y + v' H v) + f' U y + f' v
        # 0.5 y' U' H U y + v' H U y + f' U y + f' v + 0.5 v' H v
        # eliminate constants:
        # 0.5 y' U' H U y + (v' H U + f' U) y
        #
        # A x <= b
        # A (U y + v) <= b
        # A U y <= b - A v
        #
        # C x == d
        # C (U y + v) == d
        # C U y == d - C v

        H = U.T.dot(self.H).dot(U)
        f = v.dot(self.H).dot(U) + self.f.dot(U)
        A = self.A.dot(U)
        b = self.b - self.A.dot(v)
        C = self.C.dot(U)
        d = self.d - self.C.dot(v)
        T = Affine(self.T.A.dot(U), self.T.A.dot(v) + self.T.b)
        return SimpleQuadraticProgram(H, f, A, b, C, d, T)

    def transform_goal_to_origin(self):
        """
        Given an optimization of the form:
        minimize 0.5 z' H z + f' z
           z
        such that A z <= b
                  C z == d

        Perform a variable substitution defined by:

            z = U y + v

        such that we can write an equivalent optimization over y with f' = 0
        (in other words, such that the optimal cost occurs at y = 0).
        """

        """
        0.5 z' H z + f' z
        0.5 ( y' U' H U y + y' U' H v + v' H U y + v' H v) + f' U y + f' v
        0.5 y' U' H U y + v' H U y + f' U y + f' v + 0.5 v' H v
        eliminate constants:
        0.5 y' U' H U y + (v' H U + f' U) y

        So we need: v' H U + f' U == 0
        U' H' v + U' f == 0
        H' v + f == 0
        v == -H^-T f
        """
        U = np.eye(self.num_vars)
        assert np.allclose(self.H, self.H.T)
        zero_mask = [np.allclose(self.H[i, :], 0) for i in range(self.H.shape[0])]
        nonzero_mask = np.logical_not(zero_mask)
        assert np.allclose(self.f[np.logical_not(nonzero_mask)], 0)
        Hhat = self.H[nonzero_mask, :][:, nonzero_mask]
        fhat = self.f[nonzero_mask]
        vhat = -np.linalg.inv(Hhat).T.dot(fhat)
        v = np.zeros(self.f.shape)
        v[nonzero_mask] = vhat
        assert np.allclose(self.H.T.dot(v) + self.f, 0, atol=1e-6)
        # v = -np.linalg.inv(self.H).T.dot(self.f)
        return self.affine_variable_substitution(U, v)

    def eliminate_equality_constrained_variables(self):
        """
        Given:
            - self: an optimization program over variables x
        Returns:
            - new_program: a new optimization program over variables
                           z \subset x with all equality-constrained variables
                           eliminated by solving C x == d

        Note: new_program will have its internal affine transform set to account
        for the elimination, so calling new_program.solve() will still return
        values for x
        """

        T, preserved = eliminate_equality_constrained_variables(self.C, self.d)
        # x = T.A z + T.b
        new_program = self.affine_variable_substitution(T.A, T.b)
        mask = np.ones(new_program.C.shape[0], dtype=np.bool)
        for i in range(new_program.C.shape[0]):
            if np.allclose(new_program.C[i, :], 0):
                assert np.isclose(new_program.d[i], 0)
                mask[i] = False
        new_program.C = new_program.C[mask, :]
        new_program.d = new_program.d[mask]
        return new_program, preserved

    def convert_double_inequalities_to_equalities(self):
        """
        Returns a new quadratic program with all double-sided inequalities:
            a'x <= b AND a'x >= b
        converted to equalities:
            a'x == b
        """

        qp = self.copy()
        to_delete = np.zeros(qp.A.shape[0], dtype=np.bool)
        C_new = []
        d_new = []
        for i in range(qp.A.shape[0]):
            vi = np.hstack((qp.A[i, :], qp.b[i]))
            if np.linalg.norm(vi) == 0:
                continue
            vi /= np.linalg.norm(vi)
            for j in range(i + 1, qp.A.shape[0]):
                vj = np.hstack((qp.A[j, :], qp.b[j]))
                if np.linalg.norm(vj) == 0:
                    continue
                vj /= np.linalg.norm(vj)
                if np.allclose(vi, -vj):
                    to_delete[i] = True
                    to_delete[j] = True
                    C_new.append(vi[:-1])
                    d_new.append(vi[-1])
        qp.A = qp.A[np.logical_not(to_delete), :]
        qp.b = qp.b[np.logical_not(to_delete)]
        if C_new:
            qp.C = np.vstack((qp.C, np.vstack(C_new)))
            qp.d = np.hstack((qp.d, np.hstack(d_new)))
        return qp

    def eliminate_trivial_inequalities(self, tol=1e-7):
        """
        Returns a new quadratic program with all trivial inequality
        constraints (of the form: a'x <= b where a == 0 and b == 0)
        removed.
        """
        qp = self.copy()
        to_delete = np.zeros(qp.A.shape[0], dtype=np.bool)
        for i in range(qp.A.shape[0]):
            if np.linalg.norm(qp.A[i, :]) <= tol:
                assert qp.b[i] >= -10 * tol, "Constraint appears to be trivially infeasible: a: {}, b: {}".format(qp.A[i, :], qp.b[i])
                to_delete[i] = True
        qp.A = qp.A[np.logical_not(to_delete), :]
        qp.b = qp.b[np.logical_not(to_delete)]
        return qp

    def eliminate_redundant_inequalities(self):
        """
        Returns a new quadratic program with all redundant inequality
        constraints removed.
        """
        p = Polytope(self.A.copy(), self.b.copy().reshape((-1, 1)))
        p.assemble()
        A = p.lhs_min
        b = p.rhs_min.reshape((-1))
        assert A.shape[1] == self.A.shape[1]
        assert A.shape[0] == b.size
        new_program = SimpleQuadraticProgram(self.H, self.f,
                                             A, b,
                                             self.C, self.d,
                                             self.T)
        return new_program

    def copy(self):
        return SimpleQuadraticProgram(self.H.copy(), self.f.copy(),
                                      self.A.copy(), self.b.copy(),
                                      self.C.copy(), self.d.copy(),
                                      T=self.T.copy())

    def permute_variables(self, new_order):
        """
        Given:
            - self: an optimization program over variables x
            - new_order: a new ordering of variables
        Returns:
            - new_program: an optimization program over variables z such that
                           z = x[new_order]

        Note: new_program will have its internal affine transform set to account
        for the permutation, so calling new_program.solve() will still return
        values in the original order of x.
        """
        assert len(new_order) == self.num_vars
        P = np.linalg.inv(permutation_matrix(new_order))
        # x = P x[new_order]
        return self.affine_variable_substitution(P, np.zeros(self.num_vars))


class CanonicalMPCQP(object):
    """
    Represents a model-predictive control quadratic program of the form:

    minimize 0.5 u' H u + x' F u + 0.5 * x' Q x
      u, x
    such that G u <= W + E x

    Also stores an affine transform T such that calling solve() returns:

        y = T.A * [u; x] + T.b
    """
    def __init__(self, H, F, Q, G, W, E, T=None):
        self.H = H
        self.F = F
        self.Q = Q
        self.G = G
        self.W = W
        self.E = E
        if T is None:
            nv = G.shape[1] + E.shape[1]
            T = Affine(np.eye(nv), np.zeros(nv))
        self.T = T
        self._H_inv = None
        self._S = None

    def save(self, file):
        np.lib.npyio._savez(file, args=[], kwds={
            "H": self.H,
            "F": self.F,
            "Q": self.Q,
            "G": self.G,
            "W": self.W,
            "E": self.E,
            "TA": self.T.A,
            "Tb": self.T.b
        }, compress=False, allow_pickle=False)

    @staticmethod
    def load(file):
        data = np.load(file)
        qp = CanonicalMPCQP(H=data["H"],
                            F=data["F"],
                            Q=data["Q"],
                            G=data["G"],
                            W=data["W"],
                            E=data["E"],
                            T=Affine(data["TA"],
                                     data["Tb"]))
        data.close()
        return qp

    @staticmethod
    def from_mathematicalprogram(prog, u, x):
        u = np.asarray(u)
        nu = u.size
        x = np.asarray(x)
        nvars = u.size + x.size
        u_mask = np.zeros(nvars, dtype=np.bool)
        u_mask[:nu] = True
        x_mask = np.zeros(nvars, dtype=np.bool)
        x_mask[nu:] = True

        qp = SimpleQuadraticProgram.from_mathematicalprogram(prog)
        qp = qp.convert_double_inequalities_to_equalities()

        order = mpc_order(prog, u, x)
        qp = qp.permute_variables(order)

        qp, preserved = qp.eliminate_equality_constrained_variables()
        u_mask = u_mask[preserved]
        x_mask = x_mask[preserved]
        assert sum(u_mask) + sum(x_mask) == qp.A.shape[1]

        qp = qp.transform_goal_to_origin()

        # print "A"
        # print qp.A
        # print "b"
        # print qp.b

        qp = qp.eliminate_trivial_inequalities()

        qp = qp.eliminate_redundant_inequalities()
        assert np.allclose(qp.f, 0, atol=1e-6)
        assert np.allclose(qp.C, 0, atol=1e-6)

        H = qp.H[u_mask, :][:, u_mask]
        F = qp.H[x_mask, :][:, u_mask]
        Q = qp.H[x_mask, :][:, x_mask]

        G = qp.A[:, u_mask]
        W = qp.b.reshape((-1, 1))
        E = -qp.A[:, x_mask]
        return CanonicalMPCQP(H, F, Q, G, W, E, qp.T)

    def eliminate_state_constraints(self, tol=1e-7):
        to_delete = np.zeros(self.G.shape[0], dtype=np.bool)
        for i in range(self.G.shape[0]):
            if np.linalg.norm(self.G[i, :]) <= tol:
                to_delete[i] = True
        self.G = self.G[np.logical_not(to_delete), :]
        self.W = self.W[np.logical_not(to_delete), :]
        self.E = self.E[np.logical_not(to_delete), :]

    def to_simple_qp(self):
        nu = self.G.shape[1]
        nx = self.E.shape[1]
        H = np.vstack((np.hstack((self.H, self.F.T)),
                       np.hstack((self.F, self.Q))))
        f = np.zeros(nu + nx)
        A = np.hstack((self.G, -self.E))
        b = self.W
        return SimpleQuadraticProgram(H, f, A, b, C=None, d=None, T=self.T)

    def solve(self):
        return self.to_simple_qp().solve()

    @property
    def H_inv(self):
        if self._H_inv is None:
            self._H_inv = np.linalg.inv(self.H)
        return self._H_inv

    @property
    def S(self):
        # change of variables for exeplicit MPC (z := u_seq + H^-1 F^T x0)
        if self._S is None:
            self._S = self.E + self.G.dot(self.H_inv.dot(self.F.T))
        return self._S


class MPCQPFactory(object):
    """
    VARIABLES:
        sys:
        N:
        Q:
        R:
        P:
        terminal_cost:
        terminal_constraint:
        state_constraints:
        input_constraints:
        H:
        F:
        G:
        W:
        E:
        S:
    """
    def __init__(self, sys, N, Q, R, terminal_cost=None, terminal_constraint=None, state_constraints=None, input_constraints=None):
        self.sys = sys
        self.N = N
        self.Q = Q
        self.R = R
        self.terminal_cost = terminal_cost
        self.terminal_constraint = terminal_constraint
        self.state_constraints = state_constraints
        self.input_constraints = input_constraints
        return

    def add_state_constraint(self, lhs, rhs):
        if self.state_constraints is None:
            self.state_constraints = Polytope(lhs, rhs)
        else:
            self.state_constraints.add_facets(lhs, rhs)
        return

    def add_input_constraint(self, lhs, rhs):
        if self.input_constraints is None:
            self.input_constraints = Polytope(lhs, rhs)
        else:
            self.input_constraints.add_facets(lhs, rhs)
        return

    def add_state_bound(self, x_max, x_min):
        if self.state_constraints is None:
            self.state_constraints = Polytope.from_bounds(x_max, x_min)
        else:
            self.state_constraints.add_bounds(x_max, x_min)
        return

    def add_input_bound(self, u_max, u_min):
        if self.input_constraints is None:
            self.input_constraints = Polytope.from_bounds(u_max, u_min)
        else:
            self.input_constraints.add_bounds(u_max, u_min)
        return

    def set_terminal_constraint(self, terminal_constraint):
        self. terminal_constraint = terminal_constraint
        return

    def assemble(self):
        if self.state_constraints is not None:
            self.state_constraints.assemble()
        if self.input_constraints is not None:
            self.input_constraints.assemble()
        self.terminal_cost_matrix()
        self.constraint_blocks()
        self.cost_blocks()
        self.critical_regions = None
        return CanonicalMPCQP(H=self.H,
                              F=self.F,
                              Q=np.eye(self.F.shape[0]),
                              G=self.G,
                              W=self.W,
                              E=self.E)

    def terminal_cost_matrix(self):
        if self.terminal_cost is None:
            self.P = self.Q
        elif self.terminal_cost == 'dare':
            self.P = self.dare()[0]
        else:
            raise ValueError('Unknown terminal cost!')
        return

    def dare(self):
        # DARE solution
        P = linalg.solve_discrete_are(self.sys.A, self.sys.B, self.Q, self.R)
        # optimal gain
        K = - linalg.inv(self.sys.B.T.dot(P).dot(self.sys.B)+self.R).dot(self.sys.B.T).dot(P).dot(self.sys.A)
        return [P, K]

    def constraint_blocks(self):
        # compute each constraint
        [G_u, W_u, E_u] = self.input_constraint_blocks()
        [G_x, W_x, E_x] = self.state_constraint_blocks()
        [G_xN, W_xN, E_xN] = self.terminal_constraint_blocks()
        # gather constraints
        G = np.vstack((G_u, G_x, G_xN))
        W = np.vstack((W_u, W_x, W_xN))
        E = np.vstack((E_u, E_x, E_xN))
        # remove redundant constraints
        constraint_polytope = Polytope(np.hstack((G, -E)), W)
        constraint_polytope.assemble()
        self.G = constraint_polytope.lhs_min[:,:self.sys.n_u*self.N]
        self.E = - constraint_polytope.lhs_min[:,self.sys.n_u*self.N:]
        self.W = constraint_polytope.rhs_min
        return

    def input_constraint_blocks(self):
        if self.input_constraints is None:
            G_u = np.array([]).reshape((0, self.sys.n_u*self.N))
            W_u = np.array([]).reshape((0, 1))
            E_u = np.array([]).reshape((0, self.sys.n_x))
        else:
            G_u = linalg.block_diag(*[self.input_constraints.lhs_min for i in range(0, self.N)])
            W_u = np.vstack([self.input_constraints.rhs_min for i in range(0, self.N)])
            E_u = np.zeros((W_u.shape[0], self.sys.n_x))
        return [G_u, W_u, E_u]

    def state_constraint_blocks(self):
        if self.state_constraints is None:
            G_x = np.array([]).reshape((0, self.sys.n_u*self.N))
            W_x = np.array([]).reshape((0, 1))
            E_x = np.array([]).reshape((0, self.sys.n_x))
        else:
            [free_evolution, forced_evolution] = self.sys.evolution_matrices(self.N)
            lhs_x_diag = linalg.block_diag(*[self.state_constraints.lhs_min for i in range(0, self.N)])
            G_x = lhs_x_diag.dot(forced_evolution)
            W_x = np.vstack([self.state_constraints.rhs_min for i in range(0, self.N)])
            E_x = - lhs_x_diag.dot(free_evolution)
        return [G_x, W_x, E_x]

    def terminal_constraint_blocks(self):
        if self.terminal_constraint is None:
            G_xN = np.array([]).reshape((0, self.sys.n_u*self.N))
            W_xN = np.array([]).reshape((0, 1))
            E_xN = np.array([]).reshape((0, self.sys.n_x))
        else:
            if self.terminal_constraint == 'moas':
                # solve dare
                K = self.dare()[1]
                # closed loop dynamics
                A_cl = self.sys.A + self.sys.B.dot(K)
                # constraints for the maximum output admissible set
                lhs_cl = np.vstack((self.state_constraints.lhs_min, self.input_constraints.lhs_min.dot(K)))
                rhs_cl = np.vstack((self.state_constraints.rhs_min, self.input_constraints.rhs_min))
                # compute maximum output admissible set
                moas = self.maximum_output_admissible_set(A_cl, lhs_cl, rhs_cl)[0]
                lhs_xN = moas.lhs_min
                rhs_xN = moas.rhs_min
            elif self.terminal_constraint == 'origin':
                lhs_xN = np.vstack((np.eye(self.sys.n_x), - np.eye(self.sys.n_x)))
                rhs_xN = np.zeros((2*self.sys.n_x,1))
            else:
                raise ValueError('Unknown terminal constraint!')
            forced_evolution = self.sys.evolution_matrices(self.N)[1]
            G_xN = lhs_xN.dot(forced_evolution[-self.sys.n_x:,:])
            W_xN = rhs_xN
            E_xN = - lhs_xN.dot(np.linalg.matrix_power(self.sys.A, self.N))
        return [G_xN, W_xN, E_xN]

    def cost_blocks(self):
        # quadratic term in the state sequence

        # On my mac, block_diag produces the wrong result when given an empty
        # input (i.e. when self.N = 1). Instead of an empty 0 x 0 matrix, it
        # gives an empty 1 x 0 matrix, which results in H being non-square. So
        # we need a special case to work around that issue:
        if self.N == 1:
            H_x = np.zeros((0, 0))
        else:
            H_x = linalg.block_diag(*[self.Q for i in range(0, self.N-1)])
        H_x = linalg.block_diag(H_x, self.P)
        # quadratic term in the input sequence
        H_u = linalg.block_diag(*[self.R for i in range(0, self.N)])
        # evolution of the system
        [free_evolution, forced_evolution] = self.sys.evolution_matrices(self.N)
        # quadratic term
        self.H = 2*(H_u+forced_evolution.T.dot(H_x.dot(forced_evolution)))
        # linear term
        F = 2*forced_evolution.T.dot(H_x.T).dot(free_evolution)
        self.F = F.T
        return

    @staticmethod
    def maximum_output_admissible_set(A, lhs, rhs):
        """
        Returns the maximum output admissible set (see Gilbert, Tan - Linear Systems with State and
        Control Constraints, The Theory and Application of Maximal Output Admissible Sets) for a
        non-actuated linear system with state constraints (the output vector is supposed to be the
        entire state of the system , i.e. y=x and C=I).

        INPUTS:
            A: state transition matrix
            lhs: left-hand side of the constraints lhs * x <= rhs
            rhs: right-hand side of the constraints lhs * x <= rhs

        OUTPUTS:
            moas: maximum output admissible set (instatiated as a polytope)
            t: minimum number of steps in the future that define the moas
        """

        # ensure that the system is stable (otherwise the algorithm doesn't converge)
        eig_max = np.max(np.absolute(np.linalg.eig(A)[0]))
        if eig_max > 1:
            raise ValueError('Cannot compute MOAS for unstable systems')

        # Gilber and Tan algorithm
        [n_constraints, n_variables] = lhs.shape
        t = 0
        convergence = False
        while convergence == False:

            # cost function gradients for all i
            J = lhs.dot(np.linalg.matrix_power(A,t+1))

            # constraints to each LP
            cons_lhs = np.vstack([lhs.dot(np.linalg.matrix_power(A,k)) for k in range(0,t+1)])
            cons_rhs = np.vstack([rhs for k in range(0,t+1)])

            # list of all minima
            J_sol = []
            for i in range(0, n_constraints):
                J_sol_i = linear_program(np.reshape(-J[i,:], (n_variables,1)), cons_lhs, cons_rhs)[1]
                J_sol.append(-J_sol_i - rhs[i])

            # convergence check
            if np.max(J_sol) < 0:
                convergence = True
            else:
                t += 1

        # define polytope
        moas = Polytope(cons_lhs, cons_rhs)
        moas.assemble()

        return [moas, t]
