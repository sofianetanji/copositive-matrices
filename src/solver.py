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
import numpy as np
import cvxpy as cp
from scipy.special import comb
from parrilo import creation_matrix_moments, creation_parrilo_polynomial
from randomgen import random_initialization

def define_problem(a_value, r_value):
    """Define the SDP problem using CVXPY
    """
    # Define constants
    dim_c = int(comb(7 + r_value, 2 + r_value))
    moments = np.array(creation_matrix_moments(2+r_value))

    # Define input parameters
    a_parameter = cp.Parameter((6, 6), value = a_value)
    r_parameter = cp.Parameter(value = r_value)

    # Define variables
    c_var = cp.Variable((dim_c, dim_c), PSD = True)
    t_var = cp.Variable()

    # Define constraints
    lefthandside = np.array(creation_parrilo_polynomial(a_parameter.value, r_parameter.value))
    righthandside = c_var.flatten() @ moments.T
    constraint1 = [c_var - t_var * np.eye(dim_c) >> 0]
    constraint2 = [i == j for i,j in zip(righthandside,lefthandside)]
    constraints = constraint1 + constraint2

    # Define objective
    obj = cp.Maximize(t_var)

    # Define problem
    prob = cp.Problem(obj, constraints)

    return prob, c_var

def solver(n_experiments, n_family, verb, r_value = 1):
    """
    Solves n_experiments instances of the SDP for
    - a matrix belonging to the n_family
    - a Parrilo value r_value
    """
    distances, status, variables = [], [], []
    for _ in range(n_experiments):
        problem, var = define_problem(random_initialization(n_family), r_value)
        # problem.solve(solver = cp.CVXOPT, kktsolver = "robust", verbose = verb)
        try:
            problem.solve(solver = cp.MOSEK, verbose = verb)
            distances.append(problem.value)
            variables.append(var.value)
            status.append(problem.status)
        except ValueError:
            pass
    return(distances, status, variables)
