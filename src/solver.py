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
import cvxpy as cp
from scipy.special import comb
from parrilo import CreationMatrixMoments, CreationParriloPolynomial
from randomgen import random_initialization

def define_problem(A_value, r_value):
    """
    """
    # Define constants
    d = int(comb(7 + r_value, 2 + r_value))
    I = np.eye(d)
    M = np.array(CreationMatrixMoments(2+r_value))
    
    # Define input parameters
    A = cp.Parameter((6, 6), value = A_value)
    r = cp.Parameter(value = r_value)
    
    # Define variables
    C = cp.Variable((d, d), PSD = True)
    t = cp.Variable()

    # Define constraints
    LHS = np.array(CreationParriloPolynomial(A.value, r.value))
    RHS = C.flatten() @ M.T
    constraint1 = [C - t * I >> 0]
    constraint2 = [r == l for r,l in zip(RHS,LHS)]
    constraints = constraint1 + constraint2

    # Define objective
    obj = cp.Maximize(t)

    # Define problem
    prob = cp.Problem(obj, constraints)

    return(prob)

def solver(n_family, r, N_experiments):
    """
    """
    distances, status = [], []
    for _ in range(N_experiments):
        problem = define_problem(random_initialization(n_family), r)
        problem.solve(solver = cp.CVXOPT, kktsolver = "robust", verbose = True)
        distances.append(problem.value)
        status.append(problem.status)
    return(distances, status)