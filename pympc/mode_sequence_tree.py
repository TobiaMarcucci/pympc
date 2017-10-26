import pygraphviz as pgv
import os
import webbrowser
from pympc.optimization.gurobi import linear_program
import numpy as np
from collections import namedtuple
from pympc.recursive_feasibility_check import recursive_feasibility_check
from copy import copy

QPBlocks = namedtuple('QPBlocks',  ['C_u', 'C_x', 'C', 'C_u_parent', 'C_x_parent', 'C_parent'])



class ModeSequenceTree:

    def __init__(self, mode_sequence_fragment=None, qp_blocks=None):
        if mode_sequence_fragment is None:
            self.mode_sequence_fragment = ()
        else:
            self.mode_sequence_fragment = mode_sequence_fragment
        self.qp_blocks = qp_blocks
        self.children = dict()

    def expand(self, qp_library):
        mode_sequences = qp_library.keys()
        branching_children = get_branching_children(self.mode_sequence_fragment, mode_sequences)
        for child in branching_children:
            for mode_sequence, qp in qp_library.items():
                if mode_sequence[:len(child)] == child:
                    row_start = qp.row_sparsity[len(self.mode_sequence_fragment)]
                    row_end = qp.row_sparsity[len(child)]
                    column_start = qp.column_sparsity[len(self.mode_sequence_fragment)]
                    column_end = qp.column_sparsity[len(child)]
                    qp_blocks = QPBlocks(
                        C_u = qp.C_u[row_start:row_end, :column_end],
                        C_x = qp.C_x[row_start:row_end, :],
                        C = qp.C[row_start:row_end, :],
                        C_u_parent = qp.C_u[:row_start, :column_start],
                        C_x_parent = qp.C_x[:row_start, :],
                        C_parent = qp.C[:row_start, :],
                        )
                    break
            child_tree = ModeSequenceTree(child, qp_blocks)
            child_tree.expand(qp_library)
            self.children[child[-1]] = child_tree

    def check_feasibility(self, x, basis_parent):
        A_1 = self.qp_blocks.C_u_parent
        b_1 = self.qp_blocks.C_x_parent.dot(x) + self.qp_blocks.C_parent
        A_2 = self.qp_blocks.C_u
        b_2 = self.qp_blocks.C_x.dot(x) + self.qp_blocks.C
        return recursive_feasibility_check(A_2, b_2, A_1, b_1, basis_parent)

    def get_feasible_mode_sequences(self, x):
        feasible_mode_sequences = []
        check_time = 0.
        return self._get_feasible_mode_sequences(x, feasible_mode_sequences, check_time)

    def _get_feasible_mode_sequences(self, x, feasible_mode_sequences, check_time, basis_parent=None):
        if len(self.children) == 0:
            feasible_mode_sequences.append(self.mode_sequence_fragment)
        else:
            for child in self.children.values():
                is_feasible, basis_child, single_check_time = child.check_feasibility(x, copy(basis_parent))
                check_time += single_check_time
                if is_feasible:
                    print('Mode sequence ' + str(child.mode_sequence_fragment) + ' feasible.')
                    feasible_mode_sequences, check_time = child._get_feasible_mode_sequences(x, feasible_mode_sequences, check_time, basis_child)
                else:
                    print('Mode sequence ' + str(child.mode_sequence_fragment) + ' infeasible.')
        return feasible_mode_sequences, check_time

    # def expand(self, mode_sequences):
    #     branching_children = get_branching_children(self.mode_sequence_fragment, mode_sequences)
    #     for child in branching_children:
    #         child_tree = ModeSequenceTree(child)
    #         child_tree.expand(mode_sequences)
    #         self.children[child[-1]] = child_tree

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
        for child in self.children.values():
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
                    # if len(child.children) == 0:
                    n.attr['shape']='box'
            graph = child._plot(graph)
        return graph


def get_branching_children(mode_sequence_fragment, mode_sequences):
    branching_children = []
    children = get_children(mode_sequence_fragment, mode_sequences)
    for child in children:
        grandchildren = get_children(child, mode_sequences)
        while len(grandchildren) == 1:
            child = grandchildren[0]
            grandchildren = get_children(child, mode_sequences)
        branching_children.append(child)
    return branching_children

def get_children(mode_sequence_fragment, mode_sequences):
    children = []
    N = len(mode_sequences[0])
    if len(mode_sequence_fragment) < N:
        for ms in mode_sequences:
            if ms[:len(mode_sequence_fragment)] == mode_sequence_fragment:
                children.append(ms[:len(mode_sequence_fragment)+1])
    return list(set(children))


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# def plot_mode_sequence_tree(mode_sequences):

#     # initilize tree
#     A=pgv.AGraph(directed=True, strict=True)
#     A.graph_attr['label']='Mode Sequence Tree'

#     # plot mode sequences
#     for ms in mode_sequences:
#         for i in range(1, len(ms)):
#             parent = ms[:i]
#             child = ms[:i+1]
#             A.add_edge(parent, child, color='blue')
#             n = A.get_node(parent)
#             n.attr['label'] = ms[i-1]
#             n = A.get_node(child)
#             n.attr['label'] = ms[i]

#     # write .dot file
#     path = os.getcwd()
#     A.write(path + '/tree.dot')
    
#     # generate .png file (also .pdf can be generated)
#     B = pgv.AGraph(path + '/tree.dot')
#     B.layout(prog='dot')
#     B.draw(path + '/tree.pdf')

#     # # read .png and plot
#     # img = mpimg.imread('tree.png')
#     # imgplot = plt.imshow(img)

def get_feasible_mode_sequences_inefficient(qp_library, x):
    feasible_mode_sequences = []
    check_time = 0.
    for ms, qp in qp_library.items():
        A = qp.C_u
        b = qp.C_x.dot(x) + qp.C
        is_feasible, _, single_check_time = recursive_feasibility_check(A, b)
        check_time += single_check_time
        if is_feasible:
            feasible_mode_sequences.append(ms)
    return feasible_mode_sequences, check_time