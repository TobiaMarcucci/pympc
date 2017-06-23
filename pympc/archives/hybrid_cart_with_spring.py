import numpy as np
import matplotlib.pyplot as plt
from geometry import Polytope

# numerical parameters
m = 1.
k = 1.
o = np.sqrt(k/m)
x_max = np.array([[1.],[1.]])
x_min = -x_max

for i, h in enumerate([.1,1.,2.,3.5,5.,10.]):

    plt.figure()

    # mode sequence 1
    A = np.array([[-1., 0.],[-1., -h]])
    b = np.zeros((2,1))
    D1 = Polytope(A,b)
    D1.add_bounds(x_min, x_max)
    D1.assemble()
    D1.plot(facecolor=np.array([1.,.5,.5]))
    plt.text(D1.center[0], D1.center[1], '{1}')

    # mode sequence 2
    if h <= np.pi/o:
        A = np.array([[1., 0.],[np.cos(o*h), np.sin(o*h)/o]])
        b = np.zeros((2,1))
        D2 = Polytope(A,b)
        D2.add_bounds(x_min, x_max)
        D2.assemble()
        D2.plot(facecolor=np.array([.5,1.,.5]))
        plt.text(D2.center[0], D2.center[1], '{2}')

    # mode sequence 12
    A = np.array([[-1., 0.],[1., h],[-1., np.pi/o - h]])
    b = np.zeros((3,1))
    D12 = Polytope(A,b)
    D12.add_bounds(x_min, x_max)
    D12.assemble()
    D12.plot(facecolor=np.array([.5,.5,1.]))
    plt.text(D12.center[0], D12.center[1], '{1,2}')

    # mode sequence 121
    if h >= np.pi/o:
        A = np.array([[-1., 0.],[1., h],[1., h - np.pi/o]])
        b = np.zeros((3,1))
        D121 = Polytope(A,b)
        D121.add_bounds(x_min, x_max)
        D121.assemble()
        D121.plot(facecolor=np.array([1.,1.,.5]))
        plt.text(D121.center[0], D121.center[1], '{1,2,1}')

    # mode sequence 21
    if h <= np.pi/o:
        A = np.array([[1., 0.],[- np.cos(o*h), - np.sin(o*h)/o]])
        b = np.zeros((2,1))
    else:
        A = np.array([[1., 0.]])
        b = np.zeros((1,1))
    D21 = Polytope(A,b)
    D21.add_bounds(x_min, x_max)
    D21.assemble()
    D21.plot(facecolor=np.array([1.,.5,1.]))
    plt.text(D21.center[0], D21.center[1], '{2,1}')

    def q(q0, qd0):
        x = np.array([[q0],[qd0]])
        if D1.applies_to(x):
            return q0 + h*qd0
        if h <= np.pi/o and D2.applies_to(x):
            return q0*np.cos(o*h) + qd0*np.sin(o*h)/o
        if D12.applies_to(x):
            return qd0*np.sin(o*(h+q0/qd0))/o
        if h >= np.pi/o and D121.applies_to(x):
            return qd0*(np.pi/o-h)-q0
        if D21.applies_to(x):
            if o*q0/qd0 > 0.:
                return np.sqrt((q0*o)**2+qd0**2)*(h+1/o*(np.arctan(o*q0/qd0)-np.pi))
            else:
                return np.sqrt((q0*o)**2+qd0**2)*(h+1/o*np.arctan(o*q0/qd0))
            
    x = np.arange(x_min[0], x_max[0], (x_max[0]-x_min[0])/100.)
    y = np.arange(x_min[1], x_max[1], (x_max[1]-x_min[1])/100.)
    X, Y = np.meshgrid(x, y)
    z = np.array([q(x, y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = z.reshape(X.shape)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)

    plt.xlabel(r'$q_0$')
    plt.ylabel(r'$\dot q_0$')
    plt.xlim(x_min[0], x_max[0])
    plt.ylim(x_min[1], x_max[1])
    plt.savefig('/Users/tobia/Google Drive/UNIPI/PhD/box_atlas/notes/figures/state_partition_' + str(i) + '.pdf')