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
# from sympy.solvers.solveset import linsolve
from monomials import POSITIONS
from parrilo import creation_monomial_vectors

NUM = len(POSITIONS)
phi1, phi2, phi3, phi4, phi5, phi6, phi7 = symbols('phi1 phi2 phi3 phi4 phi5 phi6 phi7')

# family 13.1 and 13.2
roots_13 = [ Matrix([[sin(phi1), sin(phi1 + phi2), sin(phi1), 0, 0, 0]]),
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
    matrix = np.empty((56, 56), dtype=sp.core.symbol.Symbol)
    for i, j in POSITIONS:
        matrix[i][j] = symbols('c_{}_{}'.format(i, j))
    return matrix

def build_csymbs():
    '''
    build array of C symbols
    '''
    c_matrix = build_empty_c()
    c_symbs = []
    for i in range(56):
        for j in range(56):
            if c_matrix[i][j] is not None:
                c_symbs.append(c_matrix[i][j])
    return c_symbs

csymbs = build_csymbs()

def build_bigroot(root):
    """
    This function returns the vector of all monomials of degree 3
    evaluated in root
    """
    monomials = creation_monomial_vectors(3, 1)
    bigroot = []
    for i in range(56):
        bigroot.append(tuple(l * r for l, r in zip(monomials[i], root)))
    return bigroot

def set_constraints(bigroot):
    """
    This function returns the set of constraints on
    the C matrix depending on Phi.
    """

    # Build empty C
    c_symbols = build_empty_c()

    # Build constraints
    constraints = []
    for i in range(56):
        kernel = 0
        for j in range(56):
            if (i, j) in POSITIONS:
                kernel += c_symbols[i][j] * sum(bigroot[j])
        constraints.append(kernel)

    return constraints

cstrnts = []
bigroots = []
for single_root in roots_13:
    cstrnts += set_constraints(build_bigroot(single_root))
cstrnts = [c for c in cstrnts if c != 0]
print("All constraints on C for one root, families 13.1 and 13.2")
for cons in cstrnts:
    print(cons)
print("---------------------------------------------------------------")
# Start timer
temps = time.time()
# Solve system : using the current constraints take too much time ( > 6 hours, did not complete)
# We need to make pre-processing
# linsolve(cstrnts, tuple(csymbs))
elapsed = time.time() - temps
print("Time for solving the full system: %.5f sec" % elapsed)
