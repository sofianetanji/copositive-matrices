#! /usr/bin/env python3
# -*- coding: utf-8; mode: python -*-
"""
Treat the solutions of the SDP to infer structure of the C matrix.

Part of the following project:
http://www-ljk.imag.fr/membres/Roland.Hildebrand/emo/project_description.pdf from a MSIAM course.

- *Date:* Friday, November the 27th, 2020
- *Author:* Sofiane Tanji, for the MSIAM Master
- *Licence:* GNU GPL3 Licence
"""
import numpy as np
import matplotlib.pyplot as plt
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def mean_matrix(matrices):
    """input is a matrix
    output : element by element mean
    """
    mean = np.empty((56, 56))
    for mat in matrices:
        mean += mat
    return mean / len(matrices)

with np.load('../outputs/.npz') as data:
    lst = data.files
    variables = data['arr_1']
    mean_vars = []
    for matrix in variables[0]:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        try:
            plt.imshow(matrix, interpolation = 'nearest', cmap = plt.cm.jet)
            plt.colorbar()
            plt.show()
            print("matrice est : \n", matrix)
        except TypeError:
            pass

np.load = np_load_old
