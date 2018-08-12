import numpy as np
from time import time


class Node(object):

    def __init__(self, parent, branch):
        self.parent = parent
        self.identifier = branch
        if parent is not None:
            self.identifier.update(parent.identifier)
        self.solution = None
        self.feasible = None
        self.objective = None
        self.integer_feasible = None
        self.num_children = None
        self.num_solved_children = 0

    def solve(self, solver):
        self.solution, self.feasible, self.objective, self.integer_feasible = solver(self.identifier)
        if self.parent is not None:
            self.parent.num_solved_children += 1

    def branch(self, branching_rule):
        assert self.feasible
        children_branches = branching_rule(self.identifier, self.solution)
        self.num_children = len(children_branches)
        return children_branches

    def is_leaf(self):
        return self.num_solved_children < self.num_children


def branch_and_bound(solver, candidate_selection, branching_rule, tol=0., verbose=False):
    '''
    `solver`:
        function that given an `identifier` for a node returns `solution`, `feasible`, `objective`, `integer_feasible`.
    `candidate_selection`:
        function that given a list of `Node`s and the current `incumbent` picks the `Node` to solve next.
    `branching_rule`:
        function that given the `identifier` and the `solution` of the father returns a list of branches for the children.
    '''

    # initialization
    tic = time()
    candidate_nodes = [Node(None, {})]
    incumbent = None
    upper_bound = np.inf
    lower_bound = -np.inf
    leaves = []
    explored_nodes = 0

    # termination check
    while upper_bound - lower_bound > tol:

        # selection of candidate node
        candidate_node = candidate_selection(candidate_nodes, incumbent)
        candidate_nodes.remove(candidate_node)

        # solution of candidate node
        candidate_node.solve(solver)
        explored_nodes += 1

        # fathoming
        if not candidate_node.feasible:  # infeasible node
            pass
        elif candidate_node.objective >= upper_bound:  # feasible but worse than incumbent
            pass
        elif candidate_node.integer_feasible:  # integer feasible and better than incumbent
            if candidate_node.objective <= upper_bound:
                if incumbent is not None:
                    leaves.remove(incumbent)
                leaves.append(candidate_node)
                upper_bound = candidate_node.objective
                incumbent = candidate_node
                if verbose:
                    print 'New upper bound:', upper_bound

        # branching
        else:
            leaves.append(candidate_node)
            children_branches = candidate_node.branch(branching_rule)
            candidate_nodes += [Node(candidate_node, branch) for branch in children_branches]

        # update leaves and lower bound
        if candidate_node.parent is not None and not candidate_node.parent.is_leaf():
            leaves.remove(candidate_node.parent)
        new_lower_bound = min(leaf.objective for leaf in leaves)
        if new_lower_bound > lower_bound:
            lower_bound = new_lower_bound
            if verbose:
                print 'New lower bound:', lower_bound

    # return solution
    if verbose:
        print 'Problem solved in', time() - tic, 'seconds'
        print 'Number of nodes in the B&B tree:', explored_nodes
    if incumbent is None:  # infeasible problem
        return None
    else:  # feasible problem
        return incumbent.solution


def breadth_first(candidate_nodes):
    return candidate_nodes[0]


def depth_first(candidate_nodes):
    return candidate_nodes[-1]


def best_first(candidate_nodes):
    best = 0
    best_objective = np.inf
    for i, n in enumerate(candidate_nodes):
        if n.parent is not None and n.parent.objective < best_objective:
            best = i
    return candidate_nodes[best]