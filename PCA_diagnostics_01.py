#!/usr/local/bin/python2.7
# encoding: utf-8

import sys
import os
import numpy as NP
import numpy.linalg as LA
from matplotlib import pyplot as PLT
from matplotlib import cm as CM
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

NP.set_printoptions(precision=3, linewidth=85, suppress=True)

# 10 sets of measurements, representing a data row
	# a: thickness; b: horizontal displacement; c: vertical displacement
a = '7 4 6 8 8 7 5 9 7 8'
b = '4 1 3 6 5 2 3 5 4 2'
c = '3 8 5 1 7 9 3 8 5 2'

fnx = lambda v : NP.array(map(int, v.split())).reshape(10, 1)

a, b, c, = fnx(a), fnx(b), fnx(c)
assert len(a) == len(b) == len(c)
X = NP.hstack((a, b, c))

#------------------ Begin PCA --------------------#

# calculate the correlation matrix:
C = NP.corrcoef(X, rowvar=0)

# gives a 3 x 3 matrix (for three dimensions of data points)

# calculate the eigenvalues of C:
eva, evc = LA.eig(C)

# sort the eigenvalue array in decending order:
eva1 = NP.sort(eva)[::-1]

# get value proportion of each eigenvalue:
eva2 = NP.cumsum(eva1/NP.sum(eva1))

# print/display this intermediate result in table:
evas = NP.arange(1, 4)
evas = evas.reshape(3,1)
eva1 = eva1.reshape(3,1)
eva2 = eva2.reshape(3,1)
q = NP.hstack((evas, eva1, eva2))

title1 = "ev value proportion"
print(title1)
print( "{0}".format("-"*len(title1)) )
for row in q :
	print("{0:1d} {1:3f} {2:3f}".format(int(row[0]), row[1], row[2]))

# ev value proportion
# -------------------
# 1 1.768774 0.589591
# 2 0.927076 0.898617
# 3 0.304150 1.000000


########################## Plotting ###############################

fig = PLT.figure()
ax1 = fig.add_subplot(111)

num_eigenvals = 3
x = range(1, num_eigenvals+1)[::-1]

ax1.bar(x, eva1[::-1], width=.3, color='#6699CC', edgecolor='#6699CC', 
            align='center')

width=.3
xtick_labels = ['ev1', 'ev2', 'ev3']
pos = NP.arange(num_eigenvals) + 1.
PLT.xticks(pos, xtick_labels, color='#4B5320', weight='regular', size='small')


PLT.show()