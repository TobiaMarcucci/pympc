import unittest
import numpy as np

import mpc_tools.mpcqp as mqp
import pydrake.solvers.mathematicalprogram as mp
from mpc_tools.optimization.mpqpsolver import MPQPSolver
from boxatlas.contactstabilization import MixedIntegerTrajectoryOptimization


class TestSymbolicExplicitMPC(unittest.TestCase):
    def test_redundant_inputs(self):
        """
        Test a system with redundant input variables, which violates the
        assumption that input variables are never eliminated when removing
        equality-constrained variables.
        """

        N = 2
        dt = 0.05
        ts = np.linspace(0, N * dt, N + 1)
        mass = 1
        dim = 2
        q_max = np.ones(dim)
        q_min = -q_max
        v_max = 10 * np.ones(dim)
        v_min = -v_max
        a_max = 10 * np.ones(dim)
        a_min = -a_max
        f_max = 10 * np.ones(dim)
        f_min = -f_max

        prog = MixedIntegerTrajectoryOptimization()
        qcom = prog.new_piecewise_polynomial_variable(ts, dimension=dim, degree=2)
        prog.add_continuity_constraints(qcom)
        vcom = qcom.derivative()
        prog.add_continuity_constraints(vcom)
        acom = vcom.derivative()

        force = prog.new_piecewise_polynomial_variable(ts, dimension=dim, degree=0)

        for j in range(len(ts) - 1):
            for i in range(dim):
                prog.AddLinearConstraint(mass * acom(ts[j])[i] == force(ts[j])[i])

        for j in range(len(ts) - 1):
            q = qcom.from_below(ts[j + 1])
            v = vcom.from_below(ts[j + 1])
            for i in range(dim):
                prog.AddLinearConstraint(q[i] <= q_max[i])
                prog.AddLinearConstraint(q[i] >= q_min[i])
                prog.AddLinearConstraint(v[i] <= v_max[i])
                prog.AddLinearConstraint(v[i] >= v_min[i])

                prog.AddLinearConstraint(acom(ts[j])[i] <= a_max[i])
                prog.AddLinearConstraint(acom(ts[j])[i] >= a_min[i])
                prog.AddLinearConstraint(force(ts[j])[i] <= f_max[i])
                prog.AddLinearConstraint(force(ts[j])[i] >= f_min[i])

            prog.AddQuadraticCost(10 * np.sum(np.power(q, 2)))
            prog.AddQuadraticCost(0.01 * np.sum(np.power(v, 2)))
            prog.AddQuadraticCost(0.001 * np.sum(np.power(acom(ts[j]), 2)))
            prog.AddQuadraticCost(0.001 * np.sum(np.power(force(ts[j]), 2)))


        x = []
        for j in range(len(ts) - 1):
            x.append(np.hstack([np.hstack(qcom.functions[j].coeffs[:-1])]))
        x = np.vstack(x).T

        u = []
        for j in range(len(ts) - 1):
            u.append(np.hstack([np.hstack(qcom.functions[j].coeffs[-1:])] +
                               [np.hstack(force.functions[j].coeffs)]))
        u = np.vstack(u).T

        qp = mqp.CanonicalMPCQP.from_mathematicalprogram(prog, u, x)
        MPQPSolver(qp)


if __name__ == '__main__':
    unittest.main()
