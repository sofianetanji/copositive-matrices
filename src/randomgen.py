#! /usr/bin/env python3
# -*- coding: utf-8; mode: python -*-
"""
Implementation of the semidefinite program to check the exactness of Parillo relaxations.

Part of the following project http://www-ljk.imag.fr/membres/Roland.Hildebrand/emo/project_description.pdf from a MSIAM course.

- *Date:* Friday, November the 27th, 2020
- *Author:* Sofiane Tanji, for the MSIAM Master
- *Licence:* GNU GPL3 Licence
"""

# Constants
DIMENSION = 6

# Libraries

import numpy as np
from numpy import cos

# Randomly generate instances of 6x6 copositive matrices in the extreme rays

def random_initialization(n_family):
    """
    n_family is an integer between 1 and 5.
    It represents one of the families of special copositive matrices.
    """

    assert (n_family in [1, 2, 3, 4, 5]), "n_family should be between 1 and 5"

    if n_family == 1:
        X = [[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0], [0, 0, 1/2, 1/2, 0, 0], [0, 1/2, 1/2, 0, 0, 0]]
        X = np.array(X)
        p = np.msort(np.random.random(9,))
        p = (np.append(p, 1) - np.insert(p, 0, 0)) @ X
        p1, p2, p3, p4, p5, p6 = np.pi * p[0], np.pi * p[1], np.pi * p[2], np.pi * p[3], np.pi * p[4], np.pi * p[5]
        A = np.array([[1, -cos(p1), cos(p1+p2), -cos(p1+p2+p3), cos(p5+p6), -cos(p6)],
        [-cos(p1), 1, -cos(p2), cos(p2+p3), -cos(p2+p3+p4), cos(p1+p6)],
        [cos(p1+p2), -cos(p2), 1, -cos(p3), cos(p3+p4), -cos(p3+p4+p5)],
        [-cos(p1+p2+p3), cos(p2+p3), -cos(p3), 1, -cos(p4), cos(p4+p5)],
        [cos(p5+p6), -cos(p2+p3+p4), cos(p3+p4), -cos(p4), 1, -cos(p5)],
        [-cos(p6), cos(p1+p6), -cos(p3+p4+p5), cos(p4+p5), -cos(p5), 1]])
        return(A)
    
    elif n_family == 2:
        X = [[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0]]
        X = np.array(X)
        p = np.msort(np.random.random(6,))
        p = (np.append(p, 1) - np.insert(p, 0, 0)) @ X
        p1, p2, p3, p4, p5, p6 = np.pi * p[0], np.pi * p[1], np.pi * p[2], np.pi * p[3], np.pi * p[4], np.pi * p[5]
        A = np.array([[1, -cos(p1), cos(p1+p2), -cos(p1+p2+p3), cos(p5+p6), -cos(p6)],
        [-cos(p1), 1, -cos(p2), cos(p2+p3), -cos(p5+p6+p1), cos(p1+p6)],
        [cos(p1+p2), -cos(p2), 1, -cos(p3), cos(p3+p4), -cos(p3+p4+p5)],
        [-cos(p1+p2+p3), cos(p2+p3), -cos(p3), 1, -cos(p4), cos(p4+p5)],
        [cos(p5+p6), -cos(p5+p6+p1), cos(p3+p4), -cos(p4), 1, -cos(p5)],
        [-cos(p6), cos(p1+p6), -cos(p3+p4+p5), cos(p4+p5), -cos(p5), 1]])
        return(A)
    
    elif n_family == 3: # nondegenerate stratum 16
        X = [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 1/2, 0, 0, 1/2], [0, 0, 0, 1, 0, 0, 0]]
        X = np.array(X)
        p = np.msort(np.random.random(8,))
        p = (np.append(p, 1) - np.insert(p, 0, 0)) @ X
        p1, p2, p3, p4, p5, p6, p7 = np.pi * p[0], np.pi * p[1], np.pi * p[2], np.pi * p[3], np.pi * p[4], np.pi * p[5], np.pi * p[6]
        A = np.array([[1, -cos(p2), -cos(p1), cos(p2+p3), cos(p2+p4), cos(p1+p5)],
        [-cos(p2), 1, cos(p1+p2), -cos(p3), -cos(p4), cos(p3+p6)],
        [-cos(p1), cos(p1+p2), 1, cos(p5+p6), cos(p5+p7), -cos(p5)],
        [cos(p2+p3), -cos(p3), cos(p5+p6), 1, cos(p6-p7), -cos(p6)],
        [cos(p2+p4), -cos(p4), cos(p5+p7), cos(p6-p7), 1, -cos(p7)],
        [cos(p1+p5), cos(p3+p6), -cos(p5), -cos(p6), -cos(p7), 1]])
        return(A)

    
    elif n_family == 4: # nondegenerate stratum 19
        # TODO
        return()
    
    # nondegenerate stratum 17
    # TODO
