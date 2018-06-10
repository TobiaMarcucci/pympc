import os
import time
from numpy import argsort
from numpy import inf
from pygraphviz import AGraph
from subprocess import call

class Tree(object):

    def __init__(self, get_cost):

        self.get_cost = get_cost
        self.incumbent = None
        self.root_node = Node(None, None, get_cost, self.get_upper_bound())

    def explore(self):
        if self.root_node.feasible and self.root_node.result['integer_feasible']:
            self.incumbent = self.root_node
            return
        if not self.root_node.feasible:
            return
        parent = self.root_node
        while True:
            child = parent.get_child(self.get_cost, self.get_upper_bound())
            while child is None or not child.feasible or child.pruned: # is there something better ?
                parent = parent.parent
                if parent is None:
                    return
                child = parent.get_child(self.get_cost, self.get_upper_bound())
            if child.result['integer_feasible']: # here a child is always feasible and never pruned
                self.incumbent = child
            else:
                parent = child

    def get_solve_time(self):
        return self.root_node.subtree_solve_time()

    def get_upper_bound(self):
        if self.incumbent is None:
            return None
        else:
            return self.incumbent.result['cost']

class Node(object):

    def __init__(self, parent, index, get_cost, upper_bound):
        self.parent = parent
        self.index = index
        if parent is None:
            self.result = get_cost(self.path())
        else:
            self.result = get_cost(
                self.path(),
                parent.result['variable_basis'],
                parent.result['constraint_basis'],
                upper_bound
                )
        self.feasible = self.result['feasible']
        self.pruned = self.result['cutoff']
        self.childern_order = self.get_children_order()
        self.children = []

    # def is_pruned(self, upper_bound):
    #     pruned = False
    #     if self.feasible and upper_bound is not None:
    #             pruned = self.result['cost'] > upper_bound
    #     return pruned

    def get_children_order(self):
        if self.feasible and self.result['children_score'] is not None:
            childern_order = argsort(self.result['children_score'])[::-1].tolist()
            if self.index is not None:
                childern_order.insert(0, childern_order.pop(childern_order.index(self.index)))
        else:
            childern_order = None
        return childern_order

    def path(self):
        if self.parent is None:
            return []
        return self.parent.path() + [self.index]

    def get_child(self, get_cost, upper_bound):
        for index in self.childern_order[len(self.children):]:
            child = Node(self, index, get_cost, upper_bound)
            self.children.append(child)
            if child.feasible and not child.pruned:
                return child
        return None

    def subtree_solve_time(self, solve_time=None):
        if solve_time is None:
            solve_time = self.result['solve_time']
        for child in self.children:
            solve_time += child.result['solve_time']
            solve_time += child.subtree_solve_time(0.)
        return solve_time

def draw_tree(tree, file_name='branch_and_bound'):
    graph = AGraph(
        directed=True,
        strict=True,
        filled=True
        )
    graph.node_attr['style'] = 'filled'
    graph.node_attr['fillcolor'] = 'white'
    graph.graph_attr['label'] = 'Cumulative solve time: %.3f' % tree.get_solve_time()
    draw_node(tree.root_node, graph)
    draw_children(tree.root_node, graph)
    draw_optimal_path(tree.incumbent, graph)
    directory = os.getcwd() + '/' + file_name
    graph.write(directory + '.dot')
    graph = AGraph(directory + '.dot')
    graph.layout(prog='dot')
    graph.draw(directory + '.pdf')
    call(('open', directory + '.pdf'))
    print 'Number of nodes explore:', graph.number_of_nodes()

def draw_optimal_path(incumbent, graph):
    if incumbent is not None:
        parent = incumbent
        while parent is not None:
            node = graph.get_node(str(parent.path()))
            node.attr['fillcolor'] = 'green'
            parent = parent.parent
        for path in [incumbent.result['mode_sequence'][:i] for i in range(len(incumbent.path())+1, len(incumbent.result['mode_sequence']))]:
            graph.add_node(
                str(path),
                fillcolor='green',
                label=str(path[-1])
                )
            graph.add_edge(
                str(path[:-1]),
                str(path)
                )

def draw_children(node, graph):
    for child in node.children:
        draw_node(child, graph)
        draw_children(child, graph)

def draw_node(node, graph):
    if not node.feasible:
        color = 'red'
    elif node.pruned:
        color = 'blue'
    else:
        color = 'black'
    label = ''
    if node.index is not None:
        label += 'Index: '+ str(node.index) + '\n'
    if node.parent is not None:
        label += 'Score: %.3f' % max(node.parent.result['children_score'][node.index], 0.) + '\n' # abs because sometimes I get -0
    if node.feasible and not node.pruned:
        label += 'Cost: %.3f' % node.result['cost'] + '\n'
    label += 'Solve time: %.4f' % node.result['solve_time']
    graph.add_node(
        str(node.path()),
        color=color,
        label=label
        )
    if node.parent is not None:
        graph.add_edge(
            str(node.parent.path()),
            str(node.path())
            )