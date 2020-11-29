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
from numpy import prod
from math import factorial

def MultinomialCoefficient(n, list_of_k):
    """
    n : int
    list_of_k : list of integers whose sum is n

    This function considers the multinomial coefficient equal to the number of ordered partitions
    of an n-element set into len(list_of_k) subsets of cardinalities list_of_k[i]
    """
    return factorial(n) / prod([factorial(k) for k in list_of_k])

def CreationMonomialVectors(power, step = 2):
    '''
    power: int
    step: 1 or 2

    This function creates all monomial degrees of power.
    The argument step equal to 1 means that we take all monomials, 
    and step equal to 2 means that we take only even monomials
    '''
    monomials = []
    for k1 in range(0,power+1,step):
      for k2 in range(0,power+1-k1,step):
        for k3 in range(0,power+1-k1-k2,step):
          for k4 in range(0,power+1-k1-k2-k3,step):
            for k5 in range(0,power+1-k1-k2-k3-k4,step):
              monomials.append((k1,k2,k3,k4,k5,power-k1-k2-k3-k4-k5,))
    return monomials

def AdditionalPolynomialCreation(r):
    '''
    r : int

    This function finds the coefficients of the given expression:
    (sum from k=1 to k=6  x_k^2 )^r
    '''
    return {monom: MultinomialCoefficient(r, [int(val / 2) for val in monom]) for monom in CreationMonomialVectors(2*r, 2)}


def MatrixPolynomialCreation(A):
    '''
    A : matrix 6×6

    This function finds the coefficients of the given expression:
    sum from k,l=1 to k,l=6  A_kl * x_k^2 * x_l^2
    '''
    polynomial = {monom : 0 for monom in CreationMonomialVectors(4, 2)}
    for i in range(6):
      for j in range(6):
        vector_of_powers = [0 for _ in range(6)]
        vector_of_powers[i] +=2
        vector_of_powers[j] +=2
        polynomial[tuple(vector_of_powers)] += A[i][j]
    return polynomial
    
def MultiplicationOfPolynomials (polynomial_1, polynomial_2, max_power):
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
    polynomial = {monom : 0 for monom in CreationMonomialVectors(max_power, 2)}
    for vector_of_powers_1, coefficient_1 in polynomial_1.items():
        for vector_of_powers_2, coefficient_2 in polynomial_2.items():
            polynomial[tuple(vector_of_powers_1[i]+vector_of_powers_2[i] for i in range(6))] += coefficient_1 * coefficient_2 
    return polynomial

def CreationParriloPolynomial(A,r):
    '''
    This function finds the coefficients of the given expression:
    (sum from k,l=1 to k,l=6  A_kl * x_k^2 * x_l^2) * (sum from k=1 to k=6  x_k^2 )^r

    '''
    Monomials = CreationMonomialVectors(4+2*r, 1)
    DictionariesOfCoefficients = MultiplicationOfPolynomials(MatrixPolynomialCreation(A), AdditionalPolynomialCreation(r), 4+2*r)
    return [DictionariesOfCoefficients.get(monomial,0.0) for monomial in Monomials] 

def CreationMatrixMoments(max_power, n=6):

    size_small = int(factorial(n+max_power -1) / (factorial(max_power) * factorial(n-1)))
    size_big = int(factorial(n+2* max_power -1) / (factorial(2* max_power) * factorial(n-1)))
    Monomials_small = CreationMonomialVectors(max_power, 1)
    Monomials_big = CreationMonomialVectors(max_power * 2, 1)
    M = [[0 for _ in range(size_small*size_small)] for _ in range(size_big)]
    for k in range(size_big):
        for i in range(size_small):
            for j in range(size_small):
                if Monomials_big[k] == tuple(Monomials_small[i][k]+Monomials_small[j][k] for k in range(6)):
                    M[k][i*size_small+j] = 1 
    return M