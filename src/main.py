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
from visualization import plot
from solver import solver
from multiprocessing import Pool, cpu_count

num_workers = cpu_count()
pool = Pool(num_workers)

distances, status = solver(100, 3)
print(distances)
print(status)
plot(distances, 3)