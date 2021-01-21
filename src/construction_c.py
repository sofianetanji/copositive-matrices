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
from sympy import symbols, Matrix, sin, cos
from sympy.solvers.solveset import linsolve
from monomials import FAMILY
from parrilo import creation_monomial_vectors, creation_matrix_m, creation_parrilo_polynomial

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

# family 13.1 A matrix
matrix_131 = Matrix(
[[1, - cos(phi1), cos(phi1 + phi2), - cos(phi1 + phi2 + phi3), cos(phi5 + phi6), - cos(phi6)],
[- cos(phi1), 1, - cos(phi2), cos(phi2 + phi3), - cos(phi2 + phi3 + phi4), cos(phi1 + phi6)],
[cos(phi1 + phi2), - cos(phi2), 1, - cos(phi3), cos(phi3 + phi4), - cos(phi3 + phi4 + phi5)],
[- cos(phi1 + phi2 + phi3), cos(phi2 + phi3), - cos(phi3), 1, - cos(phi4), cos(phi4 + phi5)],
[cos(phi5 + phi6), - cos(phi2 + phi3 + phi4), cos(phi3 + phi4), - cos(phi4), 1, - cos(phi5)],
[- cos(phi6), cos(phi1 + phi6), - cos(phi3 + phi4 + phi5), cos(phi4 + phi5), - cos(phi5), 1]])

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
        bigroot.append(tuple(l * r for l, r in zip(monomials[i], root)))
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
                kernel += c_symbols[j][k] * sum(bigroot[k])
                # constraints.append(c_symbols[j][k] - c_symbols[k][j])
        constraints.append(kernel)
    return constraints


# MODIFIED
def additional_constraints(matrix_a, n_block, r_value = 1):
    """
    To enforce C =/= 0, we add all the constraints we already used in the solver part
    """
    matrix_c = build_empty_c()
    # We convert the None entries to zeros, right?
    for i in range(matrix_c.shape[0]):
        for j in range(matrix_c.shape[1]):
            if matrix_c[i][j] is None:
                matrix_c[i][j] = 0
            
    matrix_m = np.array(creation_matrix_m(2+r_value))
    shape = (matrix_m.shape[0], (slices[n_block + 1] - slices[n_block])**2)
    # We restrict the elements of M to the entries relating to the elements of the nth block of C
    matrix_m_n_block = np.empty(shape)
    k = 0
    for i in range(slices[n_block], slices[n_block + 1]):
        for j in range(slices[n_block], slices[n_block + 1]): 
            matrix_m_n_block[:, k] =  matrix_m[:, i*matrix_c.shape[1] + j]
            k = k + 1
        
    # We restrict the elements of C to those of the nth block
    matrix_c = matrix_c[slices[n_block]:slices[n_block + 1], slices[n_block]:slices[n_block + 1]]
    
    lefthandside = np.array(creation_parrilo_polynomial(matrix_a, r_value))
    # np.matmul complaind about safety and refused to work, changed it to np.dot
    #righthandside = np.matmul(matrix_c.flatten(), matrix_m_n_block.T)
    righthandside = np.dot(matrix_c.flatten(), matrix_m_n_block.T)
    # expressions equal 0
    return [i - j for i,j in zip(righthandside,lefthandside)]

# MODIFIED
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
    # We add the constraints relating to the parrilo criterion
    additional_cstrnts = additional_constraints(matrix_131, n_block)
    for c in additional_cstrnts:
        if c != 0:
            cstrnts.append(c)
            
    print("Number of constraints obtained for block {}: ".format(n_block), len(cstrnts))
    print("Number of variables for block {}: ".format(n_block), len(build_csymbs(n_block)))
    csymbs = build_csymbs(n_block)
    print("Vars: ", csymbs)
    print("Constraints: ")
    for c in cstrnts:
        print(c)
    temps = time.time()
    solution = linsolve(cstrnts, tuple(csymbs))
    elapsed = time.time() - temps
    print("Time for solving the system: %.5f sec" % elapsed)
    print("solution: ", solution)
    print("---------------------------------------------------------------")
    return solution
solving(0)
print("Number of variables to find: ", sum([len(build_csymbs(i)) for i in range(10)]))
# Solve system : using the current constraints take too much time ( > 6 hours, did not complete)
# > 6 hours when we dont use symmetry.
# With symmetry (146 vars instead of 236), maybe we can check.
# We need to make pre-processing
