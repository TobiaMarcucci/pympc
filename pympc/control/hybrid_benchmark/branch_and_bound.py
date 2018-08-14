from numpy import inf, argmin
from time import time


class Node(object):
    '''
    Node of the branch and bound tree.
    '''

    def __init__(self, parent, branch):
        '''
        A node is uniquely identified by its identifier.
        The identifier is a dictionary (typically containing the binary assigment).
        The identifier of the node is given by the union of the identifier of
        its partent and the branch dictionary.

        Arguments
        ----------
        parent : Node or None
            Parent node in the branch and bound tree.
            The root node in the tree should have None here.
        branch : dict
            Subpart of identifier that merged with the identifier of the parent
            gives the identifier of the child.
        '''

        # initialize node
        self.parent = parent
        self.solution = None
        self.feasible = None
        self.objective = None
        self.integer_feasible = None

        # build identifier of the node
        self.identifier = branch
        if parent is not None:
            self.identifier.update(parent.identifier)
        
    def solve(self, solver, objective_cutoff):
        '''
        Solves the subproblem for this node.

        Arguments
        ----------
        solver : function
            Function that given the identifier of the node (and the cutoff for the
            objective) solves its subproblem.
            The solver must return:
            - feasible (bool), True if the subproblem is feasible, False otherwise.
            - objective (float or None), cost of the subproblem (None if infeasible).
            - integer_feasible (bool), True if the subproblem is a feasible
            solution for the original problem, False otherwise.
            - solution, container for any other info we want to keep from the
            solution of the subproblem.
        objective_cutoff : float
            Cutoff value (float) for the objective, if the objective found is
            higher then the cutoff the subproblem can be considered unfeasible.
        '''

        # solve subproblem
        self.feasible, self.objective, self.integer_feasible, self.solution = solver(self.identifier, objective_cutoff)


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
        bound_tol : float
            Tolerance on the check if a new bound has been found.
        '''

        # store parameters
        self.printing_period = printing_period
        self.column_width = column_width

        # initialize variables that will change as time goes on
        self.tic = time()
        self.last_print_time = time()
        self.explored_nodes = 0
        self.lower_bound = -inf
        self.upper_bound = inf

    def add_one_node(self):
        '''
        Adds one node to the number of explore nodes.
        '''

        self.explored_nodes += 1

    def print_first_row(self):
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

    def print_new_row(self, updates):
        '''
        Prints a new row of the table.

        Arguments
        ----------
        updates : string
            Updates to write in the first column of the table.
        '''

        print ' ',
        print updates.ljust(self.column_width+1),
        print ('%.3f' % (time() - self.tic)).ljust(self.column_width+1),
        print str(self.explored_nodes).ljust(self.column_width+1),
        print ('%.3f' % self.lower_bound).ljust(self.column_width+1),
        print ('%.3f' % self.upper_bound).ljust(self.column_width+1)


    def print_status(self, lower_bound, upper_bound):
        '''
        Prints the status of the algorithm.
        It prints in case:
        - the root node is solved,
        - a new upper bound has been found after the last call of the function,
        - nothing has been printed in the last printing_period seconds.

        Arguments
        ----------
        lower_bound : float
            Best lower bound in the branch and bound algorithm at the moment of
            the call of this method.
        upper_bound : float
            Best upper bound in the branch and bound algorithm at the moment of
            the call of this method.
        '''

        # check if a print is required (self.lower_bound = -inf only at the beginning)
        root_node_solve = self.lower_bound == -inf
        print_time = (time() - self.last_print_time) > self.printing_period
        new_incumbent = upper_bound < self.upper_bound

        # update bounds
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # continue only if print is required
        if any([root_node_solve, print_time, new_incumbent]):

            # write updates if new bounds has been found in this loop
            updates = ''
            if new_incumbent:
                updates += 'New incumbent'
            elif root_node_solve:
                updates += 'Root node'

            # print
            self.print_new_row(updates)
            self.last_print_time = time()

    def print_solution(self, tol):
        '''
        Prints the final massage.
        Arguments
        ----------
        tol : float
            Positive convergence tolerance on the different between the best lower
            bound and the best upper bound.
        '''

        # infeasible problem
        if self.upper_bound == inf:
            self.print_new_row('Infeasible')
            print '\nExplored %d nodes in %.3f seconds:' % (self.explored_nodes, time() - self.tic),
            print 'problem is infeasible.'

        # optimal solution found
        else:
            self.print_new_row('Solution found')
            print '\nExplored %d nodes in %.3f seconds:' % (self.explored_nodes, time() - self.tic),
            print 'solution found with objective %.3f.' % self.upper_bound
            print 'Best lower bound is %.3f and lies within the tolerance of %.3f.' % (self.lower_bound, tol)
        

def branch_and_bound(
        solver,
        candidate_selection,
        branching_rule,
        tol=0.,
        printing_period=5.,
        **kwargs
        ):
    '''
    Branch and bound solver for combinatorial optimization problems.

    Arguments
    ----------
    solver : function
        Function that given the identifier of the node solves its subproblem.
        (See the docs in the solve method of the Node class.)
    candidate_selection : function
        Function that given a list of nodes and the current incumbent node
        picks the subproblem (node) to solve next.
        The current incumbent is provided to enable selection strategies that
        vary depending on the progress of the algorithm.
    branching_rule : function
        Function that given the identifier and the solution of the subproblem
        for this node, returns a branch (dict) for each children.
    tol : float
        Positive convergence tolerance on the different between the best lower
        bound and the best upper bound.
    printing_period : float or None
        Period in seconds for printing the status of the solver.

    Returns
    ----------
    solution : unpecified
        Generic container of the info to keep from the solution of the incumbent node.
        Is the solution output provided by solver function when applied to
        the incumbent node.
    solution_time : float
        Overall time spent to solve the combinatorial program.
    explored_nodes : int
        Number of nodes at convergence in the tree.
    '''

    # initialization
    candidate_nodes = [Node(None, {})]
    incumbent = None
    upper_bound = inf
    lower_bound = - inf

    # initialize printing
    if printing_period is not None:
        printer = Printer(printing_period, **kwargs)
        printer.print_first_row()

    # termination check (infeasibility also breaks the loop: upper_bound = lower_bound = inf)
    while upper_bound - lower_bound > tol:

        # selection of candidate node
        candidate_node = candidate_selection(candidate_nodes, incumbent)
        candidate_nodes.remove(candidate_node)

        # solution of candidate node
        candidate_node.solve(solver, upper_bound)

        # fathoming for infeasibility
        if not candidate_node.feasible:
            pass

        # fathoming for cost
        elif candidate_node.objective >= upper_bound:
            pass

        # fathoming for new incumbent
        elif candidate_node.integer_feasible:
            if candidate_node.objective < upper_bound:

                # set new result
                upper_bound = candidate_node.objective
                incumbent = candidate_node

        # branching
        else:
            for branch in branching_rule(candidate_node.identifier, candidate_node.solution):
                candidate_nodes.append(Node(candidate_node, branch))

        # compute new lower bound (returns inf if candisate_nodes = [] and upper_bound = inf)
        lower_bound = min([node.parent.objective for node in candidate_nodes if node.parent is not None] + [upper_bound])

        # print status
        if printing_period is not None:
            printer.add_one_node()
            printer.print_status(lower_bound, upper_bound)

    # return solution
    if printing_period is not None:
        printer.print_solution(tol)
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
    Gets the node whose parent has the lowest cost (in case there are siblings
    picks the first in the list, because argmin returns the first).
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

    # best node
    index_best_node = argmin([node.parent.objective if node.parent is not None else inf for node in candidate_nodes])
    return candidate_nodes[index_best_node]

def first_depth_then_breadth(candidate_nodes, incumbent):
    '''
    candidate_selection function for the branch and bound algorithm.
    Uses the depth_first approach until a feasible solution is found, then
    continues with the breadth first.
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

def first_depth_then_best(candidate_nodes, incumbent):
    '''
    candidate_selection function for the branch and bound algorithm.
    Uses the depth_first approach until a feasible solution is found, then
    continues with the best first.
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
        return best_first(candidate_nodes, incumbent)