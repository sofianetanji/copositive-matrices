#! /usr/bin/env python3
# -*- coding: utf-8; mode: python -*-
"""
Visualization module to plot pretty histograms.
Plot the SOS decomposition built

Part of the following project:
http://www-ljk.imag.fr/membres/Roland.Hildebrand/emo/project_description.pdf from a MSIAM course.

- *Date:* Friday, November the 27th, 2020
- *Author:* Sofiane Tanji, for the MSIAM Master
- *Licence:* GNU GPL3 Licence
"""

# Libraries
import matplotlib.pyplot as plt
import matplotlib.style as style
from scipy.sparse import csr_matrix
import numpy as np
import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering
style.use("ggplot")

def plot_hist(array, n_family):
    """Plots a clean histogram of array
    giving it a title depending on n_family
    """
    array1 = np.copy(array)
    array1 = array1[array1 > -1e5]
    plt.figure(figsize = (12, 12))
    plt.subplot(111)
    plt.hist(array1)
    plt.xlabel("Distance to the COP6 cone", fontsize=16)
    plt.savefig('../fig/histogram-family-{}.png'.format(n_family), dpi = 140)

def blockdiag(mat):
    """
    Bandwidth reduction problem
    http://ciprian-zavoianu.blogspot.com/2009/01/project-bandwidth-reduction.html
    This increases barely block-diagonality of the matrix.
    """
    mat = csr_matrix(mat)
    graph = nx.from_scipy_sparse_matrix(mat)
    rcm = reverse_cuthill_mckee_ordering(graph)
    return nx.to_scipy_sparse_matrix(graph, nodelist=list(rcm), format='csr').toarray()

def plot_matrix(mat):
    """
    Plots the built matrix.
    ax2 = fig.add_subplot(221)
    ax2.set_aspect("equal")
    blockd = blockdiag(mat)
    plt.imshow(blockd, interpolation = 'nearest', cmap = plt.get_cmap("gist_rainbow"))
    plt.colorbar()
    """
    print(type(mat))
    fig = plt.figure(figsize = (12, 12))
    ax1 = fig.add_subplot(111)
    ax1.set_aspect('equal')
    plt.imshow(mat, interpolation = 'nearest', cmap = plt.get_cmap("coolwarm"))
    plt.colorbar()
    plt.show()

def random2dprojection(solved, nsolved, n_family):
    """
    For each instance of instances:
    plot (p1, p2) in blue if the instance was solved
    plot (p1, p2) in red if solving the instance failed.
    """
    fig = plt.figure()
    axis = fig.add_subplot(111)
    x_ns, y_ns = [i[0] for i in nsolved], [i[1] for i in nsolved]
    x_s, y_s = [i[0] for i in solved], [i[1] for i in solved]
    axis.scatter(x_ns, y_ns, alpha = 0.8, c = 'red', edgecolors = 'none', s = 30, label = "nonsolved")
    axis.scatter(x_s, y_s, alpha = 0.8, c = 'green', edgecolors = 'none', s = 30, label = "solved")
    plt.title("Random 2D projection")
    axis.legend(loc=2)
    plt.savefig('../fig/random-2dproj-family{}.png'.format(n_family), dpi = 140)
    plt.show()
