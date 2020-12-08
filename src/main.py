#! /usr/bin/env python3
# -*- coding: utf-8; mode: python -*-
"""
Implementation of the semidefinite program to check the exactness of Parillo relaxations.

Part of the following project:
http://www-ljk.imag.fr/membres/Roland.Hildebrand/emo/project_description.pdf from a MSIAM course.

- *Date:* Friday, November the 27th, 2020
- *Author:* Sofiane Tanji, for the MSIAM Master
- *Licence:* GNU GPL3 Licence
"""

# Libraries
import os
import csv
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
import numpy as np
from visualization import plot_hist, plot_matrix, random2dprojection
from solver import solver
num_workers = cpu_count()
pool = Pool(num_workers)

def main(parameters):
    """Parameters are:
    - nb_instances
    - n_family
    - verb
    """
    if parameters.n_family != 0:
        distances, status, variables, solved, nonsolved = solver(parameters.nb_instances,
        parameters.n_family, parameters.verb)
        np.savez(os.path.join(parameters.save_dir), distances, status, variables)
        # plot_hist(distances, parameters.n_family)
        sos = next(item for item in variables if item is not None)
        plot_matrix(sos, parameters.n_family)
        random2dprojection(solved, nonsolved, parameters.n_family)
    else:
        distances, variables = [], []
        for i in [1, 2, 3, 4, 5]:
            distance, _, variable, solved, nonsolved = solver(parameters.nb_instances,
             i, parameters.verb)
            # PLOTTING
            sos = next(item for item in variable if item is not None)
            plot_matrix(sos, i)
            random2dprojection(solved, nonsolved, i)
            plot_hist(distance, i)
            # SAVING AS NPZ
            distances.append(distance)
            variables.append(variable)
        np.savez(os.path.join(parameters.save_dir), distances, variables)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--nb_instances', type = int, default = 100)
    parser.add_argument('--n_family', type = int, default = 1)
    parser.add_argument('--verb', type = bool, default = False)
    parser.add_argument('--save_dir', type = str, default = '../outputs/')

    params = parser.parse_args()

    if not os.path.exists(params.save_dir):
        os.makedirs(params.save_dir)
    csv_file = os.path.join(params.save_dir, 'params{}.csv'.format(params.n_family))
    with open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in params.__dict__.items():
            writer.writerow([key, value])

    main(params)
