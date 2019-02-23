from numpy import isinf, inf, argmin, array
from time import time
from pygraphviz import AGraph
from subprocess import call
from os import getcwd
from copy import deepcopy

class Node(object):
    '''
    Node of the branch and bound tree.
    '''

    def __init__(self, parent, branch, feasible=None, objective=None, extra_data=None):
        '''
        A node is uniquely identified by its identifier.
        The identifier is a dictionary (typically containing the binary assigment).
        The identifier of the node is given by the union of the identifier of
        its partent and the branch dictionary.

        Arguments
        ----------
        parent : Node or None
            Parent node in the branch and bound tree.
            The parent of the root node is assumed to be None.
        branch : dict
            Sub-identifier that merged with the identifier of the parent
            gives the identifier of the child.
        lower_bound : float
            Lower bound to the objective of the node provided by the user
            (useful in case of warm start).
        extra_data : dict
            ???
        '''

        # initialize node
        self.parent = parent
        self.branch = branch
        self.feasible = feasible
        self.objective = objective
        self.extra_data = extra_data
        self.integer_feasible = None

        # build identifier of the node
        if parent is None:
            self.identifier = branch
        else:
            self.identifier = parent.identifier.copy()
            self.identifier.update(branch)
        
    def solve(self, solver):
        '''
        Solves the subproblem for this node.

        Arguments
        ----------
        solver : function
            Function that given the identifier of the node solves its subproblem.
            The solver must return:
            - feasible (bool): True if the subproblem is feasible, False otherwise.
            - objective (float or None): cost of the subproblem (None if infeasible).
            - integer_feasible (bool): True if the subproblem is a feasible
            solution for the original problem, False otherwise.
            - extra_data: container for all data we want to retrieve after the solution.
        '''

        # solve subproblem (overwrites self.objective and self.extra_data)
        [self.feasible, self.integer_feasible, self.objective, self.extra_data] = solver(self.identifier)

        # correction in case of numerical errors
        if self.feasible and self.parent is not None:
            self.objective = max(self.objective, self.parent.objective)

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
        self.lower_bound = -inf
        self.upper_bound = inf

    def add_one_node(self):
        '''
        Adds one node to the number of explore nodes.
        '''

        self.explored_nodes += 1

    def print_warm_start(self, warm_start):

        print 'Loaded warm start with %d nodes.' % (len(warm_start)),
        print 'Lower bound provided by the warm start is %.3f.' % min([node.objective for node in warm_start])

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
        print ('%d' % self.explored_nodes).ljust(self.column_width+1),
        print ('%.3f' % self.lower_bound).ljust(self.column_width+1),
        print ('%.3f' % self.upper_bound).ljust(self.column_width+1)


    def print_and_update(self, lower_bound, upper_bound):
        '''
        Prints the status of the algorithm ONLY in case:
        - the root node is solved,
        - a new incumbent has been found after the last call of the function,
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
        root_node_solved = self.lower_bound == -inf
        new_incumbent = upper_bound < self.upper_bound
        print_time = (time() - self.last_print_time) > self.printing_period

        # update bounds (to be done before printing)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # continue only if print is required
        if any([root_node_solved, new_incumbent, print_time]):

            # write updates if new bounds has been found in this loop
            if root_node_solved:
                updates = 'Root node'
            elif new_incumbent:
                updates = 'New incumbent'
            elif print_time:
            	updates = ''

            # print
            self.print_new_row(updates)
            self.last_print_time = time()

    def print_solution(self):
        '''
        Prints the final massage.
        '''

        # print final row in the table
        if self.upper_bound == inf:
            self.print_new_row('Infeasible')
        else:
        	self.print_new_row('Solution found')

        # print nodes and time
        print '\nExplored %d nodes in %.3f seconds:' % (self.explored_nodes, time() - self.tic),

        # print bounds
        if self.upper_bound == inf:
            print 'problem is infeasible.'
        else:
            print 'optimal solution found with objective %.3f.' % self.upper_bound
        
class Drawer(object):
    '''
    Drawer of the branch and bound tree.
    '''

    def __init__(self):

        # initialize tree
        self.graph = AGraph(directed=True, strict=True, filled=True,)
        # self.graph.graph_attr['label'] = 'Branch and bound tree'
        self.graph.node_attr['style'] = 'filled'
        self.graph.node_attr['fillcolor'] = 'white'

    def draw_node(self, node, pruning_criteria):
        '''
        Adds a node to the tree.

        Arguments
        ----------
        node : instance of Node
            Leaf to be added to the tree.
        pruning_criteria : string or None
            Reason why the leaf has been pruned ('infeasibility', 'suboptimality', or 'new_incumbent').
            None in case the leaf has not been pruned.
        '''

        # node color based on the pruning criteria
        if pruning_criteria == 'infeasibility':
            color = 'red'
        # elif pruning_criteria == 'suboptimality':
        #     color = 'blue'
        elif pruning_criteria == 'new_incumbent':
            color = 'green'
        else:
            color = 'black'

        # node label
        label = 'Branch: ' + self.break_identifier(node.branch) + '\n'
        if node.objective is not None:
            label += 'Objective: %.3f' % node.objective + '\n'

        # add node to the tree
        self.graph.add_node(node.identifier, color=color, label=label)

        # connect node to the parent
        if node.parent is not None:
            self.graph.add_edge(node.parent.identifier, node.identifier)

    def draw_warm_start(self, warm_start):
        for node in warm_start:
            label = 'Branch: ' + self.break_identifier(node.branch)
            if not isinf(node.objective):
                color = 'green'
                label +=  '\nLower bound: %.3f' % node.objective + '\n'
            else:
                color = 'blue'
            self.graph.add_node(node.identifier, color=color, label=label)

    def break_identifier(self, identifier):
        broken_identifier = [str(k) + ', ' + str(v) for k, v in identifier.items()]
        return '\n'.join(broken_identifier)

    def draw_solution(self, node):
        '''
        Marks the leaf with the optimal solution.

        Arguments
        ----------
        node : instance of Node
            Leaf associated with the optimal solution.
        '''

        # fill node with green and make the border black again
        self.graph.get_node(node.identifier).attr['color'] = 'black'
        self.graph.get_node(node.identifier).attr['fillcolor'] = 'green'

    def save_and_open(self, file_name='branch_and_bound_tree'):
        '''
        Saves the tree in a pdf file and opens it.

        Arguments
        ----------
        file_name : string
            Name of the pdf in which to save the drawing of the tree.
        '''

        # write pdf file
        directory = getcwd() + '/' + file_name
        self.graph.write(directory + '.dot')
        self.graph = AGraph(directory + '.dot')
        self.graph.layout(prog='dot')
        self.graph.draw(directory + '.pdf')

        # open pdf file
        call(('open', directory + '.pdf'))

def branch_and_bound(
        solver,
        candidate_selection,
        branching_rule,
        printing_period=3.,
        tree_file_name=None,
        warm_start=None,
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
        Function that given the identifier of the (solved) candidate node, 
        returns a branch (dict) for each children.
    printing_period : float or None
        Period in seconds for printing the status of the solver.
        Set it ot None to shut the log.

    tree_file_name :

    warm_start :



    Returns
    ----------
    extra_data : unpecified
        Generic container of the data to keep from the solution of the incumbent node.
        It is the solution output provided by solver function when applied to
        the incumbent node.
    solution_time : float
        Overall time spent to solve the combinatorial program.
    explored_nodes : int
        Number of nodes at convergence in the tree.
    '''

    # initialization
    incumbent = None
    upper_bound = inf
    optimal_leaves = []

    # parse warm start (list of candidate nodes)
    if warm_start is not None:
        candidate_nodes = deepcopy(warm_start)
        lower_bound = min([node.objective for node in candidate_nodes])
    else:
        root_node = Node(None, {})
        candidate_nodes = [root_node]
        lower_bound = - inf

    # initialize printing
    if printing_period is not None:
        printer = Printer(printing_period, **kwargs)
        if warm_start is not None:
            printer.print_warm_start(warm_start)
        printer.print_first_row()

    # initialize drawing
    if tree_file_name is not None:
        drawer = Drawer()
        if warm_start is not None:
            drawer.draw_warm_start(warm_start)

    # termination check
    # (infeasibility also breaks the loop: upper_bound = lower_bound = inf)
    while candidate_nodes:

        # selection of candidate node
        candidate_node = candidate_selection(candidate_nodes)
        candidate_nodes.remove(candidate_node)

        # solution of candidate node 
        candidate_node.solve(solver)

        # pruning, infeasibility
        if not candidate_node.feasible:
            optimal_leaves.append(candidate_node)
            pruning_criteria = 'infeasibility'

        # pruning, suboptimality
        elif candidate_node.objective >= upper_bound:
            optimal_leaves.append(candidate_node)
            pruning_criteria = 'suboptimality'

        # pruning, new incumbent
        elif candidate_node.integer_feasible:
            if candidate_node.objective < upper_bound:
                pruning_criteria = 'new_incumbent'

                # set new incumbent
                upper_bound = candidate_node.objective
                incumbent = candidate_node

                # collect leaf
                optimal_leaves.append(candidate_node)

                # prune the branches that are now suboptimal
                new_candidate_nodes = []
                for node in candidate_nodes:

                    # discard node because its warm start objective is greater than the new upper bound
                    # (note that None is always "smaller" than upper_bound)
                    if node.objective >= upper_bound:
                        optimal_leaves.append(node)

                    # discard PARENT of the node if it objective is greater than the new upper bound
                    elif node.parent is not None and node.parent.objective > upper_bound:

                        # do not collect a parent more than once
                        if not node.parent in optimal_leaves:
                            optimal_leaves.append(node.parent)

                    # if none of the previous, keep the node in the list
                    else:
                        new_candidate_nodes.append(node)
                candidate_nodes = new_candidate_nodes

        # branching
        else:
            pruning_criteria = None
            for branch in branching_rule(candidate_node.identifier):
                child = Node(candidate_node, branch)
                candidate_nodes.append(child)

        # compute new lower bound
        lower_bound = get_lower_bound(candidate_nodes)
        lower_bound = min(lower_bound, upper_bound)

        # print status
        if printing_period is not None:
            printer.add_one_node()
            printer.print_and_update(lower_bound, upper_bound)

        # draw node
        if tree_file_name is not None:
            drawer.draw_node(candidate_node, pruning_criteria)

    # print solution
    if printing_period is not None:
        printer.print_solution()

    # draw solution
    if tree_file_name is not None:
        if incumbent is not None:
            drawer.draw_solution(incumbent)
        drawer.save_and_open(tree_file_name)

    # return solution
    if incumbent is None:
        return None, None, []
    else:
        return upper_bound, incumbent.extra_data, optimal_leaves

def get_lower_bound(candidate_nodes):
    '''
    Returns inf if candidate_nodes is empty.
    '''
    lower_bound = inf
    for node in candidate_nodes:
        if node.objective is not None:
            lower_bound = min(lower_bound, node.objective)
        else:
            lower_bound = min(lower_bound, node.parent.objective)
    return lower_bound

def breadth_first(candidate_nodes):
    '''
    candidate_selection function for the branch and bound algorithm.
    FIFO selection of the nodes.
    Good for proving optimality,bad for finding feasible solutions.

    Arguments
    ----------
    candidate_nodes : list of Node
        List of the nodes among which we need to select the next subproblem to solve.

    Returns
    ----------
    candidate_node : Node
        Node whose subproblem is the next to be solved.
    '''
    
    return candidate_nodes[0]


def depth_first(candidate_nodes):
    '''
    candidate_selection function for the branch and bound algorithm.
    LIFO selection of the nodes.
    Good for finding feasible solutions, bad for proving optimality.

    Arguments
    ----------
    candidate_nodes : list of Node
        List of the nodes among which we need to select the next subproblem to solve.

    Returns
    ----------
    candidate_node : Node
        Node whose subproblem is the next to be solved.
    '''

    return candidate_nodes[-1]


def best_first(candidate_nodes):
    '''
    candidate_selection function for the branch and bound algorithm.
    Gets the node whose parent has the lowest cost (in case there are siblings
    picks the first in the list, because argmin returns the first).

    Good for proving optimality, bad for finding feasible solutions.

    Arguments
    ----------
    candidate_nodes : list of Node
        List of the nodes among which we need to select the next subproblem to solve.

    Returns
    ----------
    candidate_node : Node
        Node whose subproblem is the next to be solved.
    '''

    # best node
    lower_bounds = []
    for node in candidate_nodes:
        if node.objective is not None:
            lower_bounds.append(node.objective)
        elif node.parent is not None:
            lower_bounds.append(node.parent.objective)
        else:
            lower_bounds.append(inf)

    return candidate_nodes[argmin(lower_bounds)]