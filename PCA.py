#!/usr/local/bin/python3
# encoding: utf-8

import os
import sys
import numpy as NP
from scipy import linalg as LA
from matplotlib import pyplot as MPL
from mpl_toolkits.mplot3d import Axes3D


def PCA(D, num_eigenvalues=None, EV=0, LDA=0):
    '''
    pass in: 
        (i) a raw data array--features encoded in the cols;
            one data instance per row; 
        (ii) EV, explanatory variable, is included in D as last column;
        (iii) the LDA flag is set to False so PCA is the default techique;
            if both LDA & EV are set to True then LDA is performed
            instead of PCA
    returns:
        (i) eigenvalues (1D array);
        (ii) eigenvectors (2D array)
        (iii) covariance matrix
        
    some numerical assertions:
    
    >>> # sum of the eigenvalues is equal to trace of R    
    >>> x = R.trace()
    >>> x1 = eva.sum()
    >>> NP.allclose(x, x1)
    True
    
    >>> # determinant of R is product of eigenvalues
    >>> q = LA.det(R)
    >>> q1 = NP.prod(eva)
    >>> NP.allclose(q, q1)
    True
    '''
    if not (LDA & EV):
        D, EV = NP.hsplit(D, [-1])
    # D -= D.mean(axis=0)
    R = NP.corrcoef(D, rowvar=False)
    m, n = R.shape
    if num_eigenvalues:
        num_eigenvalues = (m - num_eigenvalues, m-1)
    eva, evc = LA.eigh(R, eigvals=num_eigenvalues)
    NP.ascontiguousarray(evc)  
    NP.ascontiguousarray(eva)
    idx = NP.argsort(eva)[::-1]
    evc = evc[:,idx]
    eva = eva[idx]
    return eva, evc, R
	
	
def eigenvalue_variance_display(eva, num_eigenvalues):
    eva = eva[:num_eigenvalues]
    eva = eva/eva.sum()
    eva_cs = eva.cumsum()
    s = 'eigenvalue        value%'
    s1 = '_' * len(s)
    print(s)
    print(s1)
    for i in range(eva.shape[0]):
        print( '{:^10.3f} {:^20.2f}'.format(eva[i], 100*eva_cs[i]) )