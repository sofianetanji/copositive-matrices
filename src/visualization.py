#! /usr/bin/env python3
# -*- coding: utf-8; mode: python -*-
"""
Visualization module to plot pretty histograms

Part of the following project:
http://www-ljk.imag.fr/membres/Roland.Hildebrand/emo/project_description.pdf from a MSIAM course.

- *Date:* Friday, November the 27th, 2020
- *Author:* Sofiane Tanji, for the MSIAM Master
- *Licence:* GNU GPL3 Licence
"""

# Libraries
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import numpy as np

# Visualization settings
sns.set(font="DejaVu Sans",
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

style.use("ggplot")

def plot(array, n_family):
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
