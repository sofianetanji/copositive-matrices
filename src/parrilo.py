#! /usr/bin/env python3
# -*- coding: utf-8; mode: python -*-
"""
Implementation of the functions to define the Parrilo cone constraints

Part of the following project:
http://www-ljk.imag.fr/membres/Roland.Hildebrand/emo/project_description.pdf from a MSIAM course.

- *Date:* Friday, November the 27th, 2020
- *Author:* Sofiane Tanji, for the MSIAM Master
- *Licence:* GNU GPL3 Licence
"""

# Libraries
import numpy as np
from math import factorial
from numpy import prod

def multinomial_coefficient(r_value, list_of_k):
    """
    r_value : int
    list_of_k : list of integers whose sum is sum

    This function considers the multinomial coefficient equal to the number of ordered partitions
    of an n-element set into len(list_of_k) subsets of cardinalities list_of_k[i]
    """
    return factorial(r_value) / prod([factorial(k) for k in list_of_k])

def creation_monomial_vectors(power, step = 2):
    '''
    power: int
    step: 1 or 2

    This function creates all monomial degrees of power.
    The argument step equal to 1 means that we take all monomials,
    and step equal to 2 means that we take only even monomials
    '''
    monomials = []
    for k_1 in range(0, power + 1, step):
        for k_2 in range(0, power + 1 - k_1, step):
            for k_3 in range(0, power + 1 - k_1 - k_2, step):
                for k_4 in range(0, power + 1 - k_1 - k_2 - k_3, step):
                    for k_5 in range(0, power + 1 - k_1 - k_2 - k_3 - k_4, step):
                        monomials.append((k_1, k_2, k_3, k_4, k_5,
                        power - k_1 - k_2 - k_3 - k_4 - k_5))
    return monomials

def additional_polynomial_creation(r_value):
    '''
    r : int

    This function finds the coefficients of the given expression:
    (sum from k=1 to k=6  x_k^2 )^r
    '''
    return {monom: multinomial_coefficient(r_value, [int(val / 2) for val in monom])
     for monom in creation_monomial_vectors(2 * r_value, 2)}


def matrix_polynomial_creation(matrix):
    '''
    matrix : matrix 6×6

    This function finds the coefficients of the given expression:
    sum from k,l=1 to k,l=6  A_kl * x_k^2 * x_l^2
    '''
    polynomial = {monom : 0 for monom in creation_monomial_vectors(4, 2)}
    for i in range(6):
        for j in range(6):
            vector_of_powers = [0 for _ in range(6)]
            vector_of_powers[i] +=2
            vector_of_powers[j] +=2
            polynomial[tuple(vector_of_powers)] += matrix[i][j]
    return polynomial


def multiplication_of_polynomials(polynomial_1, polynomial_2, max_power):
    '''
    polynomial_1, polynomial_2: dictionary. The key is a tuple of size 6
     indicating the degree of occurrence of each x. The value is the
     coefficient for such a term}

     For example, polynomial x1^4+2*x1^2*x2^2+3*x6^4 will be written as
     {
         (4,0,0,0,0,0,) : 1,
         (2,2,0,0,0,0,) : 2,
         (0,0,0,0,0,4,) : 3
     }

     max_power : int
         degree of polynomial after multiplication

    The function multiplies two polynomials
    '''
    polynomial = {monom : 0 for monom in creation_monomial_vectors(max_power, 2)}
    for vector_of_powers_1, coefficient_1 in polynomial_1.items():
        for vector_of_powers_2, coefficient_2 in polynomial_2.items():
            polynomial[tuple(vector_of_powers_1[i] + vector_of_powers_2[i]
             for i in range(6))] += coefficient_1 * coefficient_2
    return polynomial

def creation_parrilo_polynomial(matrix, r_value):
    '''
    This function finds the coefficients of the given expression:
    (sum from k,l=1 to k,l=6  matrix_kl * x_k^2 * x_l^2) * (sum from k=1 to k=6  x_k^2 )^r

    '''
    monomials = creation_monomial_vectors(4 + 2 * r_value, 1)
    dict_coeffs = multiplication_of_polynomials(matrix_polynomial_creation(matrix),
     additional_polynomial_creation(r_value), 4 + 2 * r_value)
    return [dict_coeffs.get(monomial,0.0) for monomial in monomials]

def creation_matrix_m(max_power, dim=6):
    """
    This function enforces the linear constraints
    on entries of the SOS representation
    """
    size_small = int(factorial(dim + max_power -1) / (factorial(max_power) * factorial(dim - 1)))
    size_big = int(factorial(dim + 2 * max_power -1) /
    (factorial(2 * max_power) * factorial(dim - 1)))
    monomials_small = creation_monomial_vectors(max_power, 1)
    monomials_big = creation_monomial_vectors(max_power * 2, 1)
    matrix = [[0 for _ in range(size_small*size_small)] for _ in range(size_big)]
    for k in range(size_big):
        for i in range(size_small):
            for j in range(size_small):
                if monomials_big[k] == tuple(monomials_small[i][k] + monomials_small[j][k]
                 for k in range(6)):
                    matrix[k][i * size_small + j] = 1
    return matrix
