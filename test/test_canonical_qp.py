import unittest
import numpy as np
import mpc_tools.mpcqp as mqp
import tempfile


class TestCanonicalQP(unittest.TestCase):
    def test_save_load(self):
        np.random.seed(100)
        for i in range(100):
            nx = np.random.randint(0, 5)
            nu = np.random.randint(0, 5)
            H = np.random.randn(nu, nu)
            F = np.random.randn(nx, nu)
            Q = np.random.randn(nx, nx)
            nconstr = np.random.randint(0, 5)
            G = np.random.randn(nconstr, nu)
            W = np.random.randn(nconstr)
            E = np.random.randn(nconstr, nx)
            T = mqp.Affine(np.random.randn(nx + nu, nx + nu),
                           np.random.randn(nx + nu))
            qp = mqp.CanonicalMPCQP(H, F, Q, G, W, E, T=T)
            fname = tempfile.mkstemp(suffix=".npz")[1]
            qp.save(fname)
            qp2 = mqp.CanonicalMPCQP.load(fname)

            self.assertTrue(np.allclose(qp.H, qp2.H))
            self.assertTrue(np.allclose(qp.F, qp2.F))
            self.assertTrue(np.allclose(qp.Q, qp2.Q))
            self.assertTrue(np.allclose(qp.G, qp2.G))
            self.assertTrue(np.allclose(qp.W, qp2.W))
            self.assertTrue(np.allclose(qp.E, qp2.E))
            self.assertTrue(np.allclose(qp.T.A, qp2.T.A))
            self.assertTrue(np.allclose(qp.T.b, qp2.T.b))


if __name__ == '__main__':
    unittest.main()
