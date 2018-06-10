import os
import time
from pygraphviz import AGraph
from subprocess import call

class Tree(object):

    def __init__(self, get_cost):

        # stor data
        self.get_cost = get_cost
        self.root_node = None
        self.incumbent = None

    def explore(self):

        # check root node
        self.root_node = Node(self.get_cost)
        if not self.root_node.result['feasible']:
            return
        if self.root_node.result['integer_feasible']:
            self.incumbent = self.root_node
            return

        # explore the tree until convergence
        parent = self.root_node
        while True:

            # get the first not cutoff and feasible child
            child = parent.get_child(self.get_cost, self.get_cutoff())

            # if there is not such a child, go one level up
            while child is None:

                # if back to the root node, proved global optimality
                if parent.index is None:
                    return

                # get next child at the upper level
                parent = parent.parent
                child = parent.get_child(self.get_cost, self.get_cutoff())

            # here a child is never cutoff and always feasible, if integer feasible then new incumbent
            if child.result['integer_feasible']:
                self.incumbent = child

            # go one level down
            else:
                parent = child

    def get_solve_time(self):
        return self.root_node.subtree_solve_time()

    def get_cutoff(self):
        if self.incumbent is None:
            return None
        else:
            return self.incumbent.result['cost']

    def draw(self, file_name='branch_and_bound'):

        # initilize tree
        graph = AGraph(directed=True, strict=True, filled=True)

        # parameters of rthe nodes
        graph.node_attr['style'] = 'filled'
        graph.node_attr['fillcolor'] = 'white'

        # add every node to the grapf
        self.root_node.draw(graph)
        self.root_node.draw_children(graph)
        explore_nodes = graph.number_of_nodes() # draw_optimal_path can add non-explored nodes to the graph
        self.draw_optimal_path(graph)

        # get total solve time and number of nodes
        graph.graph_attr['label'] = 'Cumulative solve time: %.2e' % self.get_solve_time() + ' s\n'
        graph.graph_attr['label'] += 'Explored nodes: %d' % explore_nodes

        # save .dot and .pdf file
        directory = os.getcwd() + '/' + file_name
        graph.write(directory + '.dot')
        graph = AGraph(directory + '.dot')
        graph.layout(prog='dot')
        graph.draw(directory + '.pdf')

        # open .pdf file
        call(('open', directory + '.pdf'))

    def draw_optimal_path(self, graph):

        # iterate over the subpaths of the incumbent path
        if self.incumbent is not None:
            ms = self.incumbent.result['mode_sequence']
            for path in [ms[:i] for i in range(len(ms)+1)]:

                # if already in the tree fill with green
                if len(path) <= len(self.incumbent.get_path()):
                    graph.get_node(str(path)).attr['fillcolor'] = 'green'

                # if not, draw the node and fill with green
                else:
                    graph.add_node(str(path), fillcolor='green', label='Index: '+str(path[-1]))
                    graph.add_edge(str(path[:-1]), str(path))

class Node(object):

    def __init__(self, get_cost, parent=None, index=None, cutoff=None):

        # store data
        self.parent = parent
        self.index = index
        self.children = []

        # get warm start
        if parent is not None:
            warm_start = parent.result
        else:
            warm_start = None

        # solve problem for this node
        self.result = get_cost(self.get_path(), cutoff, warm_start)

    def get_path(self):
        if self.parent is None:
            return []
        return self.parent.get_path() + [self.index]

    def get_child(self, get_cost, cutoff):
        '''
        Continues to populate the list self.children until a child that is not cutoff nor infesible is found.
        Returns that child or None if there is no child that satisfies the above conditions.
        The search is performed by the order of self.result['children_order'].
        '''
        for index in self.result['children_order'][len(self.children):]:
            child = Node(get_cost, self, index, cutoff)
            self.children.append(child)
            if not child.result['cutoff'] and child.result['feasible']:
                return child
        return None

    def draw(self, graph):

        # initilization
        label = ''
        color = 'black'

        # if not root node, draw index and score
        if self.parent is not None:
            label += 'Index: '+ str(self.index) + '\n'
            label += 'Score: %.3f' % max(self.parent.result['children_score'][self.index], 0.) + '\n'

        # if blue node for cutoff, red for infeasible
        if self.result['cutoff']:
            color = 'blue'
        elif not self.result['feasible']:
            color = 'red'

        # if not cutoff and feasible, draw cost
        else:
            label += 'Cost: %.3f' % self.result['cost'] + '\n'

        # draw solve time
        label += 'Solve time: %.2e' % self.result['solve_time'] + ' s'

        # add node to the pygraphviz tree
        graph.add_node(str(self.get_path()), color=color, label=label)

        # add edge to the pygraphviz tree
        if self.parent is not None:
            graph.add_edge(str(self.parent.get_path()), str(self.get_path()))

    def draw_children(self, graph):
        for child in self.children:
            child.draw(graph)
            child.draw_children(graph)

    def subtree_solve_time(self, solve_time=None):

        # initialize the count with the solve time of this node
        if solve_time is None:
            solve_time = self.result['solve_time']

        # add solve time of the children
        for child in self.children:
            solve_time += child.result['solve_time']

            # add solve time of the gran-children (do not add the solve time of the children twice!)
            solve_time += child.subtree_solve_time(0.)

        return solve_time