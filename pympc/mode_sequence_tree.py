import pygraphviz as pgv
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import os

def plot_mode_sequence_tree(mode_sequences):

    # initilize tree
    A=pgv.AGraph(directed=True, strict=True)
    A.graph_attr['label']='Mode Sequence Tree'

    # plot mode sequences
    for ms in mode_sequences:
        for i in range(1, len(ms)):
            parent = ms[:i]
            child = ms[:i+1]
            A.add_edge(parent, child, color='blue')
            n = A.get_node(parent)
            n.attr['label'] = ms[i-1]
            n = A.get_node(child)
            n.attr['label'] = ms[i]

    # write .dot file
    path = os.getcwd()
    A.write(path + '/tree.dot')
    
    # generate .png file (also .pdf can be generated)
    B = pgv.AGraph(path + '/tree.dot')
    B.layout(prog='dot')
    B.draw(path + '/tree.pdf')

    # # read .png and plot
    # img = mpimg.imread('tree.png')
    # imgplot = plt.imshow(img)