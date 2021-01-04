#! /usr/bin/env python3
# -*- coding: utf-8; mode: python -*-
"""
Construction of the C matrix using zeros of the Parrilo polynomial

Part of the following project:
http://www-ljk.imag.fr/membres/Roland.Hildebrand/emo/project_description.pdf from a MSIAM course.

- *Date:* Friday, November the 27th, 2020
- *Author:* Sofiane Tanji, for the MSIAM Master
- *Licence:* GNU GPL3 Licence
"""

# Libraries
import time
import numpy as np
import sympy as sp
from sympy import symbols, Matrix, sin
from sympy.solvers.solveset import nonlinsolve
from monomials import FAMILY
from parrilo import creation_monomial_vectors

phi1, phi2, phi3, phi4, phi5, phi6, phi7 = symbols('phi1 phi2 phi3 phi4 phi5 phi6 phi7')
slices = [0, 10, 16, 22, 28, 31, 37, 38, 44, 50, 56]

# family 13.1 and 13.2
roots_13 = [ Matrix([[sin(phi2), sin(phi1 + phi2), sin(phi1), 0, 0, 0]]),
Matrix([[0, sin(phi3), sin(phi2 + phi3), sin(phi2), 0, 0]]),
Matrix([[0, 0, sin(phi4), sin(phi3 + phi4), sin(phi3), 0]]),
Matrix([[0, 0, 0, sin(phi5), sin(phi4 + phi5), sin(phi4)]]),
Matrix([[sin(phi5), 0, 0, 0, sin(phi6), sin(phi5 + phi6)]]),
Matrix([[sin(phi1 + phi6), sin(phi6), 0, 0, 0, sin(phi1)]])
]

def build_empty_c():
    """
    build empty matrix C with symbols
    """
    matrix = np.empty((56, 56), dtype=sp.symbol.Symbol)
    for i, j in FAMILY:
        sym = symbols('c_{}_{}'.format(i, j))
        matrix[i][j]= sym
    return matrix

def build_csymbs(n_block):
    '''
    build array of C symbols
    only those of block number n_block
    '''
    c_matrix = build_empty_c()
    c_symbs = []

    for i in range(slices[n_block], slices[n_block + 1]):
        for j in range(slices[n_block], slices[n_block + 1]):
            if c_matrix[i][j] is not None:
                c_symbs.append(c_matrix[i][j])
    return list(set(c_symbs))

def build_bigroot(root):
    """
    This function returns the vector of all monomials of degree 3
    evaluated in root
    """
    monomials = creation_monomial_vectors(3, 1)
    bigroot = []
    for i in range(56):
        monomial = 1
        for l, r in zip(monomials[i], root):
            monomial = monomial*r**l
        bigroot.append(monomial)
    return bigroot

def set_constraints_byblock(bigroot, n_block):
    """
    This function returns the set of constraints on
    the C matrix depending on Phi.
    We use the specific structure of the matrix C to have
    10 different sets of constraints
    """
    c_symbols = build_empty_c()
    constraints = []
    for j in range(slices[n_block], slices[n_block + 1]):
        kernel = 0
        for k in range(slices[n_block], slices[n_block + 1]):
            if (j, k) in FAMILY:
                kernel += c_symbols[j][k] * bigroot[k]
                # constraints.append(c_symbols[j][k] - c_symbols[k][j])
        constraints.append(kernel)
    return constraints

def solving(n_block):
    """
    Final solver
    """
    cstrnts = []
    for single_root in roots_13:
        constrnt = set_constraints_byblock(build_bigroot(single_root), n_block)
        constrnt = [i for i in constrnt if i != 0]
        if constrnt != []:
            cstrnts += constrnt
    print("Number of constraints obtained for block {}: ".format(n_block), len(cstrnts))
    print("Number of variables for block {}: ".format(n_block), len(build_csymbs(n_block)))
    temps = time.time()
    csymbs = build_csymbs(n_block)
    solution = nonlinsolve(cstrnts, phi1, phi2, phi3, phi4, phi5, phi6)
    elapsed = time.time() - temps
    print("Time for solving the system: %.5f sec" % elapsed)
    print("solution: ", solution)
    # print("Vars: ", csymbs)
    print("---------------------------------------------------------------")
    return solution
solving(3)
print("Number of variables to find: ", sum([len(build_csymbs(i)) for i in range(10)]))
# Solve system : using the current constraints take too much time ( > 6 hours, did not complete)
# > 6 hours when we dont use symmetry.
# With symmetry (146 vars instead of 236), maybe we can check.
# We need to make pre-processing
