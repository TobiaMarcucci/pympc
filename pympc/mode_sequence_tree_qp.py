import pygraphviz as pgv
import os
import webbrowser
from pympc.optimization.gurobi import linear_program
from pympc.optimization.parametric_programs import ParametricQP
import numpy as np
from collections import namedtuple
from pympc.recursive_feasibility_check import recursive_feasibility_check
from copy import copy

Solution = namedtuple('Solution',  ['min', 'argmin','mode_sequence'])

class ModeSequenceTree:

    def __init__(self, mode_sequence_fragment=None, parametric_program=None):
        if mode_sequence_fragment is None:
            self.mode_sequence_fragment = ()
        else:
            self.mode_sequence_fragment = mode_sequence_fragment
        self.parametric_program = parametric_program
        self.child_trees = []

    def expand(self, controller, qp_library):
        mode_sequences = qp_library.keys()
        branching_children = get_branching_children(self.mode_sequence_fragment, mode_sequences)
        for child in branching_children:
            parametric_program = controller.condense_program(child)
            # parametric_program = generate_subprogram(child, qp_library)
            child_tree = ModeSequenceTree(child, parametric_program)
            child_tree.expand(controller, qp_library)
            self.child_trees.append(child_tree)
        self.child_trees = self.order_children_by_length()

    def order_children_by_length(self):
        parent_length = len(self.mode_sequence_fragment)
        child_lengths = [len(child.mode_sequence_fragment) for child in self.child_trees]
        child_order = reversed(np.argsort(child_lengths))
        ordered_child_tress = [self.child_trees[i] for i in child_order]
        return ordered_child_tress


    def solve(self, x, candidate_solution):

        # check if warm start is provided
        if candidate_solution is None:
            print 'No warm start provided.'
        else:
            print 'Loaded solution with cost ' + str(candidate_solution.min) + ' and mode sequence ' + str(candidate_solution.mode_sequence) + '.'

        # solve the children
        for child in self.child_trees:
            candidate_solution = child._solve_sub_tree(x, candidate_solution)

        # check final result
        if candidate_solution is None:
            print 'Infeasible initial condition ' + str(x.flatten().tolist()) + '.'
        else:
            print 'Found solution with cost ' + str(candidate_solution.min) + ' and mode sequence ' + str(candidate_solution.mode_sequence) + '.'

        return candidate_solution

    def _solve_sub_tree(self, x, candidate_solution):
        # if the node has already been explored by the warm start
        if candidate_solution is not None and candidate_solution.mode_sequence[:len(self.mode_sequence_fragment)] == self.mode_sequence_fragment:
            print 'Mode sequence fragment ' + str(self.mode_sequence_fragment) + ' not checked:'
            feasible = True
            argmin = None
            cost = None
        else:
            print 'Checking mode sequence fragment ' + str(self.mode_sequence_fragment) + ':'
            argmin, cost = self.parametric_program.solve(x)
            feasible = not np.isnan(cost)
        if not feasible:
            print '    infeasible, branch cut.'
        else:
            if len(self.child_trees) == 0:
                if cost is not None:
                    if candidate_solution is None or cost < candidate_solution.min:
                        candidate_solution = Solution(min=cost, argmin=argmin, mode_sequence=self.mode_sequence_fragment)
                        print '    new upper bound with cost ' + str(cost) + '.'
                    else:
                        print '    suboptimal (cost is ' + str(cost) + '), branch cut.'
                else:
                    print '    warm start leaf.'
            else:
                if candidate_solution is None or cost is None or cost < candidate_solution.min:
                    print '    further branch.'
                    for child in self.child_trees:
                        candidate_solution = child._solve_sub_tree(x, candidate_solution)
                else:
                    print '    suboptimal (cost is ' + str(cost) + '), branch cut.'
        return candidate_solution

    def plot(self, file_name='tree', title='Mode Sequence Tree'):
        graph = pgv.AGraph(directed=True, strict=True)
        graph.graph_attr['label'] = title
        graph = self._plot(graph)
        graph.remove_node('()')
        graph.write(os.getcwd() + '/' + file_name + '.dot')
        graph = pgv.AGraph(os.getcwd() + '/' + file_name + '.dot')
        graph.layout(prog='dot')
        graph.draw(os.getcwd() + '/' + file_name + '.pdf')
        webbrowser.open_new(r'file://C:' + os.getcwd() + '/' + file_name + '.pdf')

    def _plot(self, graph):
        for child in self.child_trees:
            l_parent = len(self.mode_sequence_fragment)
            l_child = len(child.mode_sequence_fragment)
            for i in range(l_parent, l_child):
                subparent = self.mode_sequence_fragment + child.mode_sequence_fragment[l_parent:i]
                subchild = self.mode_sequence_fragment + child.mode_sequence_fragment[l_parent:i+1]
                graph.add_edge(subparent, subchild, color='blue')
                n = graph.get_node(subparent)
                if len(subparent) > 0:
                    n.attr['label'] = subparent[-1]
                n = graph.get_node(subchild)
                n.attr['label'] = subchild[-1]
                if i == l_child - 1:
                    n.attr['color'] = 'red'
                    # if len(child.branching_children) == 0:
                    n.attr['shape']='box'
            graph = child._plot(graph)
        return graph


def get_branching_children(mode_sequence_fragment, mode_sequences):
    branching_children = []
    children = get_children_list(mode_sequence_fragment, mode_sequences)
    for child in children:
        grandchildren = get_children_list(child, mode_sequences)
        while len(grandchildren) == 1:
            child = grandchildren[0]
            grandchildren = get_children_list(child, mode_sequences)
        branching_children.append(child)
    return branching_children

def get_children_list(mode_sequence_fragment, mode_sequences):
    children = []
    N = len(mode_sequences[0])
    if len(mode_sequence_fragment) < N:
        for ms in mode_sequences:
            if ms[:len(mode_sequence_fragment)] == mode_sequence_fragment:
                children.append(ms[:len(mode_sequence_fragment)+1])
    return list(set(children))

def generate_subprogram(mode_sequence_fragment, qp_library):
    n = len(mode_sequence_fragment)
    for mode_sequence, qp in qp_library.items():
        if mode_sequence[:n] == mode_sequence_fragment:
            break
    n_c = qp.row_sparsity[n]
    n_u = qp.column_sparsity[n]
    F_uu = qp.F_uu[:n_u, :n_u]
    F_xu = qp.F_xu[:, :n_u]
    F_xx = qp.F_xx
    F_u = qp.F_u[:n_u, :]
    F_x = qp.F_x
    F = qp.F
    C_u = qp.C_u[:n_c, :n_u]
    C_x = qp.C_x[:n_c, :]
    C = qp.C[:n_c, :]
    parametric_program = ParametricQP(F_uu, F_xu, F_xx, F_u, F_x, F, C_u, C_x, C)
    return parametric_program