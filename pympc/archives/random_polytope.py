import numpy as np
from pympc.geometry.polytope import Polytope
import matplotlib.pyplot as plt

from pympc.geometry.convex_hull import PolytopeProjectionInnerApproximation


n_var = 5
n_cons = 100
# A = np.random.randn(n_cons, n_var)
# b = np.random.rand(n_cons, 1)
# np.save('A_random', A)
# np.save('b_random', b)
A = np.load('A_random.npy')
b = np.load('b_random.npy')
p0 = Polytope(A, b)
p0.assemble()


app = PolytopeProjectionInnerApproximation(A, b, [0,1])


p1 = Polytope(app.A, app.b)
p1.assemble()

p0.plot()
p1.plot()

plt.show()

for point in [np.array([[0.],[.1]]), np.array([[-.13],[.0]])]:
	print 'aaa'
	p0.plot()
	app.include_point(point)
	p2 = Polytope(app.A, app.b)
	p2.assemble()
	p2.plot()
	p1.plot()

	plt.show()

