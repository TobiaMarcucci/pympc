import numpy as np
import matplotlib.pyplot as plt

def input_sequence(u_sequence, t_s, N, u_bounds=None):
    """
    Plots the input sequence and its bounds as functions of time.

    INPUTS:
        u_sequence: list with N inputs (2D numpy vectors) of dimension (n_u,1) each
        t_s: sampling time
        N: number of steps
        u_bounds: list of bound on the input (2D numpy vectors of dimension (n_u,1))
    """

    # dimension of the input
    n_u = u_sequence[0].shape[0]

    # time axis
    t = np.linspace(0,N*t_s,N+1)

    # plot each input element separately
    for i in range(n_u):
        plt.subplot(n_u, 1, i+1)

        # plot input sequence
        u_i_sequence = [u_sequence[j][i] for j in range(0,N)]
        input_plot, = plt.step(t, [u_i_sequence[0]] + u_i_sequence, 'b')

        # plot bounds if provided
        if u_bounds is not None:
            for bound in u_bounds:
                bound_plot, = plt.step(t, bound[i,0]*np.ones(t.shape), 'r')

        # miscellaneous options
        plt.ylabel(r'$u_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            if u_bounds is not None:
                plt.legend([input_plot, bound_plot], ['Optimal control', 'Control bounds'], loc=1)
            else:
                plt.legend([input_plot], ['Optimal control'], loc=1)
    plt.xlabel(r'$t$')

    return

def state_trajectory(x_trajectory, t_s, N, x_bounds=None):
    """
    Plots the state trajectory and its bounds as functions of time.

    INPUTS:
        x_trajectory: list with N+1 states (2D numpy vectors) of dimension (n_x,1) each
        t_s: sampling time
        N: number of steps
        x_bounds: list of bound on the state (2D numpy vectors of dimension (n_x,1))
    """

    # dimension of the state
    n_x = x_trajectory[0].shape[0]

    # time axis
    t = np.linspace(0,N*t_s,N+1)

    # plot each state element separately
    for i in range(n_x):
        plt.subplot(n_x, 1, i+1)

        # plot state trajectory
        x_i_trajectory = [x_trajectory[j][i] for j in range(0,N+1)]
        state_plot, = plt.plot(t, x_i_trajectory, 'b')

        # plot bounds if provided
        if x_bounds is not None:
            for bound in x_bounds:
                bound_plot, = plt.step(t, bound[i,0]*np.ones(t.shape),'r')

        # miscellaneous options
        plt.ylabel(r'$x_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            if x_bounds is not None:
                plt.legend([state_plot, bound_plot], ['Optimal trajectory', 'State bounds'], loc=1)
            else:
                plt.legend([state_plot], ['Optimal trajectory'], loc=1)
    plt.xlabel(r'$t$')

    return


def state_space_trajectory(x_trajectory, state_components=[0,1], **kwargs):
    """
    Plots the state trajectories as functions of time (2d plot).

    INPUTS:
        x_trajectory: state trajectory \in R^((N+1)*n_x)
        N: time steps
        state_components: components of the state vector to be plotted.
    """
    for k in range(len(x_trajectory)-1):
        plt.plot([x_trajectory[k][state_components[0]], x_trajectory[k+1][state_components[0]]], [x_trajectory[k][state_components[1]], x_trajectory[k+1][state_components[1]]], **kwargs)
        # plt.text(x_trajectory[k][0], x_trajectory[k][1], r'$x('+str(k)+')$')
    # ax = plt.axes()
    x_0 = (x_trajectory[0][state_components[0]][0], x_trajectory[0][state_components[1]][0])
    plt.scatter(x_0[0], x_0[1], color='b')
    plt.text(x_0[0], x_0[1], r'$x(0)$')
    plt.xlabel(r'$x_{' + str(state_components[0]+1) + '}$')
    plt.ylabel(r'$x_{' + str(state_components[1]+1) + '}$')
    return


def state_partition(explicit_controller, active_set=False, facet_index=False, **kwargs):

    if explicit_controller.critical_regions is None:
        raise ValueError('Explicit solution not computed yet! First run .compute_explicit_solution().')

    fig, ax = plt.subplots()
    for cr in explicit_controller.critical_regions:
        cr.polytope.plot(facecolor=np.random.rand(3,1), **kwargs)
        ax.autoscale_view()
        if active_set:
            plt.text(cr.polytope.center[0], cr.polytope.center[1], str(cr.active_set))
        if facet_index:
            for j in range(0, len(cr.polytope.minimal_facets)):
                plt.text(cr.polytope.facet_centers(j)[0], cr.polytope.facet_centers(j)[1], str(cr.polytope.minimal_facets[j]))
    return

def grouped_state_partition(explicit_controller, active_set=False, first_input=False, facet_index=False, **kwargs):
    explicit_controller.group_critical_regions()
    fig, ax = plt.subplots()
    for i, family in enumerate(explicit_controller.cr_families):
        color = np.random.rand(3,1)
        for cr in family:
            cr.polytope.plot(facecolor=color, **kwargs)
            ax.autoscale_view()
            if active_set:
                plt.text(cr.polytope.center[0], cr.polytope.center[1], str(cr.active_set))
            if first_input:
                plt.text(cr.polytope.center[0], cr.polytope.center[1], str(cr.u_linear[0,:])+str(cr.u_offset[0]))
            if facet_index:
                for j in range(0, len(cr.polytope.minimal_facets)):
                    plt.text(cr.polytope.facet_centers(j)[0], cr.polytope.facet_centers(j)[1], str(cr.polytope.minimal_facets[j]))

def optimal_value_function(explicit_controller):
    if explicit_controller.critical_regions is None:
        raise ValueError('Explicit solution not computed yet! First run .compute_explicit_solution().')
    vertices = np.zeros((0,2))
    for cr in explicit_controller.critical_regions:
        vertices = np.vstack((vertices, cr.polytope.vertices))
    x_max = max([vertex[0] for vertex in vertices])
    x_min = min([vertex[0] for vertex in vertices])
    y_max = max([vertex[1] for vertex in vertices])
    y_min = min([vertex[1] for vertex in vertices])
    x = np.arange(x_min, x_max, (x_max-x_min)/100.)
    y = np.arange(y_min, y_max, (y_max-y_min)/100.)
    X, Y = np.meshgrid(x, y)
    zs = np.array([explicit_controller.optimal_value_function(np.array([[x],[y]])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    cp = plt.contour(X, Y, Z)
    plt.colorbar(cp)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(r'$V^*(x)$')
    return