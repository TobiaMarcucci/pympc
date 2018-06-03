import os
import time
import logging as log
from numpy import inf
from pygraphviz import AGraph
from subprocess import call

class Tree(object):

    def __init__(self, get_cost):
        self.get_cost = get_cost
        self.incumbent = None

    def explore(
        self,
        log_name='branch_and_bound',
        graph_name='branch_and_bound',
        graph_title='Branch-and-bound tree'
        ):
        log_path = os.getcwd() + '/' + log_name + '.log'
        if os.path.isfile(log_path):
            os.remove(log_path)
        log.shutdown()
        log.basicConfig(
            format='%(message)s',
            filename=log_name + '.log',
            filemode='w',
            level=log.INFO
            )
        graph = AGraph(
            directed=True,
            strict=True,
            filled=True
            )
        graph.node_attr['style'] = 'filled'
        graph.node_attr['fillcolor'] = 'white'
        graph.graph_attr['label'] = graph_title
        child = Node(None, None, self.get_cost, self.get_upper_bound())
        call(('open', log_path))
        tic = time.time()
        while child is not None:
            parent = child
            child = parent.get_child(self.get_cost, self.get_upper_bound(), graph)
            while child is None:
                parent = parent.parent
                if parent is None:
                    solution_time = time.time() - tic
                    self.exploration_completed(graph, graph_name, solution_time)
                    return
                child = parent.get_child(self.get_cost, self.get_upper_bound(), graph)
            if child.integer:
                self.new_incumbent(child)
                child = child.parent

    def get_upper_bound(self):
        if self.incumbent is None:
            return inf
        else:
            return self.incumbent.cost

    def new_incumbent(self, node):
        self.incumbent = node
        log.info('New upper bound:')
        log.info('Cost -> ' + str(node.cost))
        log.info('Path -> ' + str(node.path()))

    def exploration_completed(self, graph, graph_name, solution_time):
        log.info('Exploration completed in ' + str(solution_time) + ' s:')
        if self.incumbent is None:
            log.info('No feasible path found.')
        else:
            log.info('Optimal cost -> ' + str(self.incumbent.cost))
            log.info('Optimal path -> ' + str(self.incumbent.path()))
            parent = self.incumbent
            while parent.index is not None:
                node = graph.get_node(str(parent.path()))
                node.attr['fillcolor'] = 'green'
                parent = parent.parent
        graph.write(os.getcwd() + '/' + graph_name + '.dot')
        graph = AGraph(os.getcwd() + '/' + graph_name + '.dot')
        graph.layout(prog='dot')
        graph.draw(os.getcwd() + '/' + graph_name + '.pdf')
        call(('open', os.getcwd() + '/' + graph_name + '.pdf'))

class Node(object):

    def __init__(self, parent, index, get_cost, upper_bound):
        self.parent = parent
        self.index = index
        self.cost, self.integer, self.childern_order, self.others = get_cost(self.path())
        self.feasible = self.cost is not None
        self.pruned = self.cost > upper_bound # if cost is None gives False (also with upper_bound = np.inf)
        self.children = []
        self.node_report()

    def draw_node(self, graph):
        if not self.feasible:
            color = 'red'
        elif self.pruned:
            color = 'blue'
        else:
            color = 'black'
        graph.add_node(str(self.path()), color=color, label=self.index)
        if self.parent.index is not None:
            graph.add_edge(str(self.parent.path()), str(self.path()))

    def path(self):
        if self.parent is None:
            return []
        return self.parent.path() + [self.index]

    def get_child(self, get_cost, upper_bound, graph):
        for index in self.childern_order[len(self.children):]:
            child = Node(self, index, get_cost, upper_bound)
            child.draw_node(graph)
            if child.feasible and not child.pruned:
                self.children.append(child)
                return child
            else:
                self.children.append(None)
        return None

    def node_report(self):
        log.info('Explored path -> ' + str(self.path()))
        if not self.feasible:
            log.info(' unfeasible')
        elif self.pruned:
            log.info(' pruned, lower bound = ' + str(self.cost))
        else:
            log.info(' deepened')
            # if self.others['binaries'] is not None: ########### schifo qui
            #     log.info(' deepened, binaries = ' + str([round(i, 4) for i in self.others['binaries']]))
            # else:
            #     log.info(' deepened')
