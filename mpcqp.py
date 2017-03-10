import itertools
import numpy as np
import pydrake.solvers.mathematicalprogram as mp
from mpc_tools import Polytope


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


def eliminate_equality_constrained_variables(C, d, preserve=None):
    """
    Given C and d defining a set of linear equality constraints:

        C x == d

    find a matrix W such that C x == d implies x = W z for some z \subset x

    This allows us to rewrite a QP with equality constraints into a QP over
    fewer variables with no equality constraints.
    """
    if preserve is None:
        preserve = np.zeros(C.shape[1], dtype=np.bool)
    C = C.copy()
    num_vars = C.shape[1]
    W = np.eye(num_vars)
    for j in range(C.shape[1] - 1, C.shape[1] - C.shape[0] - 1, -1):
        if preserve[j]:
            continue
        nonzeros = np.nonzero(C[:, j])[0]
        if len(nonzeros) != 1:
            raise ValueError("C must be triangular (up to permutation). Try permuting the problem to mpc_order()")
        i = nonzeros[0]
        assert d[i] == 0, "Right-hand side of the equality constraints must be zero"
        v = C[i, :j] / -C[i, j]
        W = W.dot(np.vstack([np.eye(j), v]))
        C = C[[k for k in range(C.shape[0]) if k != i], :]
    return W


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
        v = -np.linalg.inv(self.H).T.dot(self.f)
        return self.affine_variable_substitution(U, v)

    def eliminate_equality_constrained_variables(self, preserve=None):
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

        W = eliminate_equality_constrained_variables(self.C, self.d, preserve)
        # x = W z
        new_program = self.affine_variable_substitution(W, np.zeros(self.num_vars))
        mask = np.ones(new_program.C.shape[0], dtype=np.bool)
        for i in range(new_program.C.shape[0]):
            if np.allclose(new_program.C[i, :], 0):
                assert np.isclose(new_program.d[i], 0)
                mask[i] = False
        new_program.C = new_program.C[mask, :]
        new_program.d = new_program.d[mask]
        return new_program

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

    @staticmethod
    def from_mathematicalprogram(prog, u, x):
        u = np.asarray(u)
        nu = u.size
        x = np.asarray(x)
        nvars = u.size + x.size
        qp = SimpleQuadraticProgram.from_mathematicalprogram(prog)
        order = mpc_order(prog, u, x)
        qp = qp.permute_variables(order)

        preserve = np.zeros(nvars, dtype=np.bool)
        preserve[:(nu + x.shape[0])] = True
        qp = qp.eliminate_equality_constrained_variables(preserve)
        qp = qp.transform_goal_to_origin()
        qp = qp.eliminate_redundant_inequalities()

        assert np.allclose(qp.f, 0)
        assert np.allclose(qp.C, 0)
        H = qp.H[:nu, :nu]
        F = qp.H[nu:, :nu]
        Q = qp.H[nu:, nu:]

        G = qp.A[:, :nu]
        W = qp.b
        E = -qp.A[:, nu:]
        return CanonicalMPCQP(H, F, Q, G, W, E, qp.T)

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


def generate_mpc_system(prog, u, x0):
    C, d = extract_linear_equalities(prog)
    A, b = extract_linear_inequalities(prog)
    Q, q = extract_objective(prog)
    assert np.allclose(q, 0), "linear objective terms are not yet implemented"
    return Q, q, A, b, C, d

    order = mpc_order(prog, u, x0)
    P = permutation_matrix(order)
    # yhat = P * y
    # y = P^-1 * yhat

    Pinv = np.linalg.inv(P)

    Qhat = Pinv.T.dot(Q).dot(Pinv)
    qhat = q.dot(Pinv)
    Ahat = A.dot(Pinv)
    bhat = b
    Chat = C.dot(Pinv)
    dhat = d

    for i in range(10):
        f = rand(q.size)
        fhat = f.dot(Pinv)
        y, cost = mpc.quadratic_program(Q, f, A, b, C, d)
        yhat, cost_hat = mpc.quadratic_program(Qhat, fhat, Ahat, bhat, Chat, dhat)
        assert np.isclose(cost, cost_hat)
        assert np.allclose(yhat, P.dot(y))

    W = simplify(Chat)
    # yhat = W * z

    Qtilde = W.T.dot(Qhat).dot(W)
    qtilde = qhat.dot(W)
    Atilde = Ahat.dot(W)
    btilde = bhat
    Ctilde = Chat.dot(W)
    dtilde = dhat
    assert np.allclose(Ctilde, 0)

    for i in range(100):
        f = rand(q.size)
        fhat = f.dot(Pinv)
        ftilde = fhat.dot(W)
        y, cost = mpc.quadratic_program(Q, f, A, b, C, d)
        yhat, cost_hat = mpc.quadratic_program(Qhat, fhat, Ahat, bhat, Chat, dhat)
        ytilde, cost_tilde = mpc.quadratic_program(Qtilde, ftilde, Atilde, btilde)
        assert np.isclose(cost, cost_hat)
        assert np.isclose(cost, cost_tilde)
        assert np.allclose(yhat, P.dot(y))
        assert np.allclose(yhat, W.dot(ytilde))

    H = Qtilde[:u.size, :u.size]
    F = Qtilde[u.size:, :u.size]
    G = Atilde[:, :u.size]
    E = -Atilde[:, u.size:]
    W = btilde

    return H, F, G, W, E