# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:28:59 2020

@author: alper
"""

import mglearn

from sklearn.datasets.samples_generator import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

from dbscan import MyDBSCAN
import matplotlib.pyplot as plt
'''

# Create three gaussian blobs to use as our clustering data.
X, labels_true = make_moons(n_samples=200, noise=0.06, random_state=42)



X = StandardScaler().fit_transform(X)

###############################################################################
# My implementation of DBSCAN
#

# Run my DBSCAN implementation.
print ('Running my implementation...')
my_labels = MyDBSCAN(X, eps=0.5, MinPts=5)
print(my_labels)
plt.scatter(X[:,0], X[:,1], c=my_labels, cmap=mglearn.cm2, s=60)
'''
###############################################################################
# Create three gaussian blobs to use as our clustering data.
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, cluster_std=0.6,
                            random_state=0)

X = StandardScaler().fit_transform(X)

###############################################################################
# My implementation of DBSCAN
#

# Run my DBSCAN implementation.
print ("Running my implementation...")
my_labels = MyDBSCAN(X, eps=0.3, MinPts=10)
print(my_labels)
plt.scatter(X[:,0], X[:,1], c=my_labels, cmap=mglearn.cm2, s=60)
