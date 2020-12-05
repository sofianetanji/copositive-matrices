#! /usr/bin/env python3
# -*- coding: utf-8; mode: python -*-
"""
Generation of random instances of the 6x6 copositive matrices belonging to the extreme rays.

Part of the following project:
http://www-ljk.imag.fr/membres/Roland.Hildebrand/emo/project_description.pdf from a MSIAM course.

- *Date:* Friday, November the 27th, 2020
- *Author:* Sofiane Tanji, for the MSIAM Master
- *Licence:* GNU GPL3 Licence
"""

# Libraries

import numpy as np
from numpy import cos, sin

# Randomly generate instances of 6x6 copositive matrices in the extreme rays

def random_initialization(n_family):
    """
    n_family is an integer between 1 and 5.
    It represents one of the families of special copositive matrices.
    """

    assert (n_family in [1, 2, 3, 4, 5]), "n_family should be between 1 and 5"

    if n_family == 1: #nondegenerate stratum 13.1
        array_x = [[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0],
         [0, 0, 1/2, 1/2, 0, 0], [0, 1/2, 1/2, 0, 0, 0]]
        array_x = np.array(array_x)
        angp = np.msort(np.random.random(9,))
        angp = (np.append(angp, 1) - np.insert(angp, 0, 0)) @ array_x
        p_1, p_2, p_3 = np.pi * angp[0], np.pi * angp[1], np.pi * angp[2]
        p_4, p_5, p_6 = np.pi * angp[3], np.pi * angp[4], np.pi * angp[5]
        instance = np.array(
            [[1, -cos(p_1), cos(p_1 + p_2), -cos(p_1 + p_2 + p_3), cos(p_5 + p_6), -cos(p_6)],
        [-cos(p_1), 1, -cos(p_2), cos(p_2 + p_3), -cos(p_2 + p_3 + p_4), cos(p_1 + p_6)],
        [cos(p_1 + p_2), -cos(p_2), 1, -cos(p_3), cos(p_3 + p_4), -cos(p_3 + p_4 + p_5)],
        [-cos(p_1 + p_2 + p_3), cos(p_2 + p_3), -cos(p_3), 1, -cos(p_4), cos(p_4 + p_5)],
        [cos(p_5 + p_6), -cos(p_2 + p_3 + p_4), cos(p_3 + p_4), -cos(p_4), 1, -cos(p_5)],
        [-cos(p_6), cos(p_1 + p_6), -cos(p_3 + p_4 + p_5), cos(p_4 + p_5), -cos(p_5), 1]])

    if n_family == 2: #nondegenerate stratum 13.2
        array_x = [[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0]]
        array_x = np.array(array_x)
        angp = np.msort(np.random.random(6,))
        angp = (np.append(angp, 1) - np.insert(angp, 0, 0)) @ array_x
        p_1, p_2, p_3 = np.pi * angp[0], np.pi * angp[1], np.pi * angp[2]
        p_4, p_5, p_6 = np.pi * angp[3], np.pi * angp[4], np.pi * angp[5]
        instance = np.array(
            [[1, -cos(p_1), cos(p_1 + p_2), -cos(p_1 + p_2 + p_3), cos(p_5 + p_6), -cos(p_6)],
        [-cos(p_1), 1, -cos(p_2), cos(p_2 + p_3), -cos(p_5 + p_6 + p_1), cos(p_1 + p_6)],
        [cos(p_1 + p_2), -cos(p_2), 1, -cos(p_3), cos(p_3 + p_4), -cos(p_3 + p_4 + p_5)],
        [-cos(p_1 + p_2 + p_3), cos(p_2 + p_3), -cos(p_3), 1, -cos(p_4), cos(p_4 + p_5)],
        [cos(p_5 + p_6), -cos(p_5 + p_6 + p_1), cos(p_3 + p_4), -cos(p_4), 1, -cos(p_5)],
        [-cos(p_6), cos(p_1 + p_6), -cos(p_3 + p_4 + p_5), cos(p_4 + p_5), -cos(p_5), 1]])

    if n_family == 3: # nondegenerate stratum 16
        array_x = [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1/2, 0, 0, 1/2], [0, 0, 0, 1, 0, 0, 0]]
        array_x = np.array(array_x)
        angp = np.msort(np.random.random(8,))
        angp = (np.append(angp, 1) - np.insert(angp, 0, 0)) @ array_x
        p_1, p_2, p_3 = np.pi * angp[0], np.pi * angp[1], np.pi * angp[2]
        p_4, p_5, p_6, p_7 = np.pi * angp[3], np.pi * angp[4], np.pi * angp[5], np.pi * angp[6]
        instance = np.array(
            [[1, -cos(p_2), -cos(p_1), cos(p_2 + p_3), cos(p_2 + p_4), cos(p_1 + p_5)],
        [-cos(p_2), 1, cos(p_1 + p_2), -cos(p_3), -cos(p_4), cos(p_3 + p_6)],
        [-cos(p_1), cos(p_1 + p_2), 1, cos(p_5 + p_6), cos(p_5 + p_7), -cos(p_5)],
        [cos(p_2 + p_3), -cos(p_3), cos(p_5 + p_6), 1, cos(p_6 - p_7), -cos(p_6)],
        [cos(p_2 + p_4), -cos(p_4), cos(p_5 + p_7), cos(p_6-p_7), 1, -cos(p_7)],
        [cos(p_1 + p_5), cos(p_3 + p_6), -cos(p_5), -cos(p_6), -cos(p_7), 1]])


    if n_family == 4: # nondegenerate stratum 19
        angp = np.msort(np.random.random(5,))
        angp = angp - np.insert(angp[1:], 0, 0)
        p_1, p_2, p_3 = np.pi * angp[0], np.pi * angp[1], np.pi * angp[2]
        p_4, p_5 = np.pi * angp[3], np.pi * angp[4]
        p_6 = p_2 + np.random.random() * (np.pi + p_1 - p_2 - p_3 - p_4 + p_5) / 2
        p_9 = np.pi + p_2 - p_6
        p_7min = np.max([np.pi - p_6 + p_2 - p_1 - p_5, p_3 + p_4 + p_6])
        p_7max = np.min([np.pi , np.pi - p_1 - p_5 + p_6 - p_2, np.pi + p_1 + p_2 + p_5 - p_6])
        p_7 = p_7min + np.random.random() * (p_7max - p_7min)
        c_8min = - (cos(p_7) * sin(p_1) + cos(p_9) * sin(p_5) ) / sin(p_1 + p_5)
        c_8max = - np.max([cos(p_5 + p_7), cos(p_1 + p_9)])
        p_8 = np.arccos(c_8min + np.random.random() * (c_8max - c_8min))
        w_1 = np.sqrt((sin(p_1) ** 2) * (sin(p_8) ** 2) - (cos(p_9) + cos(p_1) * cos(p_8)) ** 2)
        w_5 = (np.sqrt(sin(p_5) ** 2) * (sin(p_8) ** 2) - (cos(p_7) + cos(p_5) * cos(p_8)) ** 2)
        a24 = (cos(p_1) * cos(p_5) + cos(p_1) * cos(p_7) * cos(p_8) + cos(p_5) * cos(p_8)
         * cos(p_9) + cos(p_7) * cos(p_9) - w_1 * w_5 ) / (sin(p_8) ** 2)
        instance = np.array(
        [[1, - cos(p_4), cos(p_4 + p_5), cos(p_2 + p_3), -cos(p_3), cos(p_3 + p_6)],
        [-cos(p_4), 1, -cos(p_5), a24, cos(p_3 + p_4), -cos(p_7)],
        [cos(p_4 + p_5), -cos(p_5), 1, -cos(p_1), cos(p_1 + p_2), -cos(p_8)],
        [cos(p_2 + p_3), a24, -cos(p_1), 1, -cos(p_2), cos(p_6 - p_2)],
        [-cos(p_3), cos(p_3 + p_4), cos(p_1 + p_2), -cos(p_2), 1, -cos(p_6)],
        [cos(p_3 + p_6), -cos(p_7), -cos(p_8), cos(p_6-p_2), -cos(p_6), 1]])

    if n_family == 5: # nondegenerate stratum 17
        angp = np.msort(np.random.random(8,))
        angp = angp - np.insert(angp[:-1], 0, 0)
        p_1 = np.pi * angp[0]
        p_2 = np.pi * angp[1]
        p_3 = np.pi * angp[2]
        p_5 = np.pi * angp[4]
        p_6 = np.pi * angp[5]
        p_7 = np.pi * (angp[5] + angp[6]) / 2
        p_4 = p_3 + p_6 + p_7 + np.pi * angp[7]
        instance = np.array(
            [[1, -cos(p_2), -cos(p_1), cos(p_2 + p_3), cos(p_2 + p_4), cos(p_1 + p_5)],
            [-cos(p_2), 1, cos(p_1 + p_2), -cos(p_3), -cos(p_4), cos(p_3 + p_6)],
            [-cos(p_1), cos(p_1 + p_2), 1, cos(p_5 - p_6), cos(p_5 + p_7), -cos(p_5)],
            [cos(p_2 + p_3), -cos(p_3), cos(p_5 - p_6), 1, cos(p_6 + p_7), -cos(p_6)],
            [cos(p_2 + p_4), -cos(p_4), cos(p_5 + p_7), cos(p_6 + p_7), 1, -cos(p_7)],
            [cos(p_1 + p_5), cos(p_3 + p_6), -cos(p_5), -cos(p_6), -cos(p_7), 1]])

    return instance
