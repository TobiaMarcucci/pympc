# external imports
import unittest
import numpy as np
from itertools import product

# internal inputs
from pympc.geometry.polyhedron import Polyhedron
from pympc.geometry.orthogonal_projection import convex_hull_method
from pympc.test.test_geometry.utils import same_rows, same_vectors

class TestOrthogonalProjection(unittest.TestCase):

    def test_convex_hull_method(self):
        np.random.seed(3)

        # cube from n-dimensions to n-1-dimensions
        for n in range(2, 6):
            x_min = - np.ones((n, 1))
            x_max = - x_min
            p = Polyhedron.from_bounds(x_min, x_max)
            E = np.vstack((
                np.eye(n-1),
                - np.eye(n-1)
                ))
            f = np.ones((2*(n-1), 1))
            vertices = [np.array(v).reshape(n-1,1) for v in product([1., -1.], repeat=n-1)]
            E_chm, f_chm, vertices_chm = convex_hull_method(p.A, p.b, range(n-1))
            p_chm = Polyhedron(E_chm, f_chm)
            p_chm.remove_redundant_inequalities()
            self.assertTrue(same_rows(
                np.hstack((E, f)),
                np.hstack((p_chm.A, p_chm.b))
                ))
            self.assertTrue(same_vectors(vertices, vertices_chm))

        # test with random polytopes wiith m facets
        m = 10

        # dimension of the higher dimensional polytope
        for n in range(3, 6):

            # dimension of the lower dimensional polytope
            for n_proj in range(2, n):

                # higher dimensional polytope
                A = np.random.randn(m, n)
                b = np.random.rand(m, 1)
                p = Polyhedron(A, b)

                # if not empty or unbounded
                if p.vertices is not None:
                    points = [v[:n_proj, :] for v in p.vertices]
                    p = Polyhedron.from_convex_hull(points)

                    # check half spaces
                    p.remove_redundant_inequalities()
                    E_chm, f_chm, vertices_chm = convex_hull_method(A, b, range(n_proj))
                    p_chm = Polyhedron(E_chm, f_chm)
                    p_chm.remove_redundant_inequalities()
                    self.assertTrue(same_rows(
                        np.hstack((p.A, p.b)),
                        np.hstack((p_chm.A, p_chm.b))
                    ))

                    # check vertices
                    self.assertTrue(same_vectors(p.vertices, vertices_chm))

if __name__ == '__main__':
    unittest.main()