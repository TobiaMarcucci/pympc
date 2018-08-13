from numpy import inf
from time import time


class Node(object):
    '''
    Node of the branch and bound tree.
    '''

    def __init__(self, parent, branch):
        '''
        A node is uniquely identified by its identifier.
        The identifier is a dictionary (typically containing the binary assigment).
        The identifier of the node is given by the union of the identifier of its partent and the branch dictionary.

        Arguments
        ----------
        parent : Node or None
            Parent node in the branch and bound tree.
            The root node in the tree should have None here.
        branch : dict
            Subpart of identifier that merged with the identifier of the parent gives the identifier of the child.
        '''

        # initialize node
        self.parent = parent
        self.solution = None
        self.feasible = None
        self.objective = None
        self.integer_feasible = None
        self.num_children = None
        self.num_solved_children = 0

        # build identifier of the node
        self.identifier = branch
        if parent is not None:
            self.identifier.update(parent.identifier)
        
    def solve(self, solver):
        '''
        Solves the subproblem for this node.

        Arguments
        ----------
        solver : function
            Function that given the identifier of the node solves its subproblem.
            solver should return:
            - feasible (bool), True if the subproblem is feasible, False otherwise.
            - objective (float or None), cost of the subproblem (None if infeasible).
            - integer_feasible (bool), True if the subproblem is a feasible solution for the original problem, False otherwise.
            - solution, container for any other info we want to keep from the solution of the subproblem.
        '''

        # solve subproblem
        self.feasible, self.objective, self.integer_feasible, self.solution = solver(self.identifier)

        # update number of solved children for the parent
        if self.parent is not None:
            self.parent.num_solved_children += 1

    def branch(self, branching_rule):
        '''
        Given the (feasible) solution of the subproblem for the node, generates the children nodes.

        Arguments
        ----------
        branching_rule : function
            Function that given the identifier and the solution of the subproblem for this node, returns a branch (dict) for each children.

        Returns
        ----------
        children : list of Node
            Children nodes.
        '''

        # check that the subproblem has been solved
        assert self.feasible

        # branch
        children_branches = branching_rule(self.identifier, self.solution)
        children = [Node(self, branch) for branch in children_branches]

        # store number of children (needed with num_solved_children to uderstand if this node is a leaf of the tree).
        self.num_children = len(children_branches)

        return children

    def is_leaf(self):
        '''
        Checks if this node is a leaf of the branch and bound tree.
        '''

        return self.num_solved_children < self.num_children


class Printer(object):
    '''
    Printer for the branch and bound algorithm.
    '''

    def __init__(self, printing_period, column_width=15):
        '''
        Arguments
        ----------
        printing_period : float or None
            Maximum amount of time in seconds without printing.
        column_width : int
            Number of characters of the columns of the table printed during the solution.
        '''

        # store parameters
        self.printing_period = printing_period
        self.column_width = column_width

        # initialize variables that will change as time goes on
        self.tic = time()
        self.last_print_time = time()
        self.explored_nodes = 0
        self.upper_bound = inf
        self.lower_bound = -inf

    def add_one_node(self):
        '''
        Adds one node to the number of explore nodes.
        '''

        self.explored_nodes += 1

    def print_fields(self):
        '''
        Prints the first row of the table, the one with the titles of the columns.
        '''

        print '|',
        print 'Updates'.center(self.column_width) + '|',
        print 'Time (s)'.center(self.column_width) + '|',
        print 'Nodes (#)'.center(self.column_width) + '|',
        print 'Lower bound'.center(self.column_width) + '|',
        print 'Upper bound'.center(self.column_width) + '|'
        print (' ' + '-' * (self.column_width + 1)) * 5

    def print_status(self, lower_bound, upper_bound):
        '''
        Prints the status of the algorithm.
        It prints in case:
        - a new (upper or lower) bound has been found after the last call of the function,
        - nothing has been printed in the last printing_period seconds.

        Arguments
        ----------
        lower_bound : float
            Best lower bound in the branch and bound algorithm at the moment of the call of this method.
        upper_bound : float
            Best upper bound in the branch and bound algorithm at the moment of the call of this method.
        '''

        # check if a print is required
        time_now = time()
        tp = (time_now - self.last_print_time) > self.printing_period
        lb = lower_bound > self.lower_bound
        ub = upper_bound < self.upper_bound
        if not any([tp, lb, ub]):
            return

        # write updates if new bounds has been found in this loop
        updates = ''
        if lb:
            updates += 'New LB'
        if ub:
            if lb:
                updates += ', '
            updates += 'New UB'

        # print updates, time, number of nodes, and bounds
        print ' ',
        print updates.ljust(self.column_width+1),
        print ('%.3f' % (time_now - self.tic)).ljust(self.column_width+1),
        print str(self.explored_nodes).ljust(self.column_width+1),
        print ('%.3f' % lower_bound).ljust(self.column_width+1),
        print ('%.3f' % upper_bound).ljust(self.column_width+1)
        
        # update variables
        self.last_print_time = time_now
        if lb:
            self.lower_bound = lower_bound
        if ub:
            self.upper_bound = upper_bound
            

def branch_and_bound(solver, candidate_selection, branching_rule, tol=0., printing_period=5.):
    '''
    Branch and bound solver for combinatorial optimization problems.

    Arguments
    ----------
    solver : function
        Function that given the identifier of the node solves its subproblem.
        (See the docs in the solve method of the Node class.)
    candidate_selection : function
        Function that given a list of nodes and the current incumbent node picks the subproblem (node) to solve next.
        The current incumbent is provided to enable selection strategies that vary depending on the progress of the algorithm.
    branching_rule : function
        Function that given the identifier and the solution of the subproblem for this node, returns a branch (dict) for each children.
    tol : float
        Positive convergence tolerance on the different between the best lower bound and the best upper bound.
    printing_period : float or None
        Period in seconds for printing the status of the solver.

    Returns
    ----------
    solution : unpecified
        Generic container of the info to keep from the solution of the incumbent node.
        Is the solution output provided by solver function when applied to the incumbent node.
    solution_time : float
        Overall time spent to solve the combinatorial program.
    explored_nodes : int
        Number of nodes at convergence in the tree.
    '''

    # initialization (leaves are only the nodes candidate to be the lower bound)
    candidate_nodes = [Node(None, {})]
    incumbent = None
    upper_bound = inf
    lower_bound = - inf
    leaves = []

    # initialize printing
    if printing_period is not None:
        printer = Printer(printing_period)
        printer.print_fields()

    # termination check
    while upper_bound - lower_bound > tol:

        # selection of candidate node
        candidate_node = candidate_selection(candidate_nodes, incumbent)
        candidate_nodes.remove(candidate_node)

        # solution of candidate node
        candidate_node.solve(solver)

        # fathoming for infeasibility
        # (trivially not a lower bound)
        if not candidate_node.feasible:
            pass

        # fathoming for cost
        # (not in leaves since incumbent is always a better lower bound)
        elif candidate_node.objective >= upper_bound:
            pass

        # fathoming for new incumbent
        # (the incumbent can be the best lower bound at convergence with zero tolerance)
        elif candidate_node.integer_feasible:
            if candidate_node.objective < upper_bound:

                # update leaves
                if incumbent is not None:
                    leaves.remove(incumbent)
                leaves.append(candidate_node)

                # set new result
                upper_bound = candidate_node.objective
                incumbent = candidate_node

        # branching
        else:
            leaves.append(candidate_node)
            candidate_nodes += candidate_node.branch(branching_rule)

        # remove parent node from leaves if all the children have been solved
        if candidate_node.parent is not None and not candidate_node.parent.is_leaf():
            leaves.remove(candidate_node.parent)

        # compute new lower bound
        lower_bound = min(leaf.objective for leaf in leaves)

        # print status
        if printing_period is not None:
            printer.add_one_node()
            printer.print_status(lower_bound, upper_bound)

    # return solution
    if incumbent is None:
        return None
    else:
        return incumbent.solution


def breadth_first(candidate_nodes, incumbent):
    '''
    candidate_selection function for the branch and bound algorithm.
    FIFO selection of the nodes.
    Good for proving optimality,bad for finding feasible solutions.

    Arguments
    ----------
    candidate_nodes : list of Node
        List of the nodes among which we need to select the next subproblem to solve.
    incumbent : Node
        Incumbent node in the branch and bound algorithm.

    Returns
    ----------
    candidate_node : Node
        Node whose subproblem is the next to be solved.
    '''
    
    return candidate_nodes[0]


def depth_first(candidate_nodes, incumbent):
    '''
    candidate_selection function for the branch and bound algorithm.
    LIFO selection of the nodes.
    Good for finding feasible solutions, bad for proving optimality.

    Arguments
    ----------
    candidate_nodes : list of Node
        List of the nodes among which we need to select the next subproblem to solve.
    incumbent : Node
        Incumbent node in the branch and bound algorithm.

    Returns
    ----------
    candidate_node : Node
        Node whose subproblem is the next to be solved.
    '''

    return candidate_nodes[-1]


def best_first(candidate_nodes, incumbent):
    '''
    candidate_selection function for the branch and bound algorithm.
    Gets the node whose parent has the lowest cost (in case there are siblings picks the first in the list).
    Good for finding feasible solutions, bad for proving optimality.

    Arguments
    ----------
    candidate_nodes : list of Node
        List of the nodes among which we need to select the next subproblem to solve.
    incumbent : Node
        Incumbent node in the branch and bound algorithm.

    Returns
    ----------
    candidate_node : Node
        Node whose subproblem is the next to be solved.
    '''

    # best node
    index_best = 0
    objective_best = inf

    # loop over all possible candidates
    for i, node in enumerate(candidate_nodes):
        if node.parent is not None and node.parent.objective < objective_best:
            index_best = i

    return candidate_nodes[index_best]

def first_depth_then_breadth(candidate_nodes, incumbent):
    '''
    candidate_selection function for the branch and bound algorithm.
    Uses the depth_first approach until a feasible solution is found, the continues with the breadth first.
    Should get the best of the two approaches.

    Arguments
    ----------
    candidate_nodes : list of Node
        List of the nodes among which we need to select the next subproblem to solve.
    incumbent : Node
        Incumbent node in the branch and bound algorithm.

    Returns
    ----------
    candidate_node : Node
        Node whose subproblem is the next to be solved.
    '''

    if incumbent is None:
        return depth_first(candidate_nodes, incumbent)
    else:
        return breadth_first(candidate_nodes, incumbent)