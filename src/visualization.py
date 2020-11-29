#! /usr/bin/env python3
# -*- coding: utf-8; mode: python -*-
"""
Implementation of the semidefinite program to check the exactness of Parillo relaxations.

Part of the following project http://www-ljk.imag.fr/membres/Roland.Hildebrand/emo/project_description.pdf from a MSIAM course.

- *Date:* Friday, November the 27th, 2020
- *Author:* Sofiane Tanji, for the MSIAM Master
- *Licence:* GNU GPL3 Licence
"""

# Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Visualization settings
sns.set(font="Franklin Gothic Book",
        rc={
 "axes.axisbelow": False,
 "axes.edgecolor": "lightgrey",
 "axes.facecolor": "None",
 "axes.grid": False,
 "axes.labelcolor": "dimgrey",
 "axes.spines.right": False,
 "axes.spines.top": False,
 "figure.facecolor": "white",
 "lines.solid_capstyle": "round",
 "patch.edgecolor": "w",
 "patch.force_edgecolor": True,
 "text.color": "dimgrey",
 "xtick.bottom": False,
 "xtick.color": "dimgrey",
 "xtick.direction": "out",
 "xtick.top": False,
 "ytick.color": "dimgrey",
 "ytick.direction": "out",
 "ytick.left": False,
 "ytick.right": False})
sns.set_context("notebook", rc={"font.size":16,
                                "axes.titlesize":20,
                                "axes.labelsize":18})

def plotting(array):
    """
    """
    array1 = np.copy(array)
    array1 = array1[array1 > -1e5]
    plt.figure(figsize = (30, 5))
    plt.subplot(111)
    print(array)
    plt.hist(array1)
    plt.savefig('../fig/figure.png', dpi = 140)