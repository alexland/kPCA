#!/usr/local/bin/python2.7
# encoding: utf-8

import os
import sys
import numpy as NP
from scipy import linalg as LA
from matplotlib import pyplot as MPL
from mpl_toolkits.mplot3d import Axes3D


def pca(data, rescaled_dim=2, normalize=1):
	"""
		returns: rescaled 1D vectors (one for each dimension, according to
		'rescaled_dim'); eivenvectors, and eigenvalues
		pass in: 'data' (with response variable removed);
		rescaled_dim, number of dimensions in transformed data
	"""
	data -= data.mean(axis=0)
	data /= data.std(axis=0)
	C = NP.cov(data.T)		# n x n matrix w/ "1" down main diagonal
	evals, evecs = LA.eig(C)
	ndx = NP.argsort(evals)
	ndx = ndx[::-1]
	evecs = evecs[:,ndx]		# re-order eigenvectors column-wise
	evals = evals[ndx]
	if rescaled_dim > 0 :
		evecs = evecs[:,:rescaled_dim]
	x_resc = NP.dot(evecs.T, data.T)			
	y_resc = NP.dot(evecs, x_resc).T + data.mean(axis=0)
	return x_resc, y_resc, evals, evecs
