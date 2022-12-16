# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 09:34:17 2022
@author: S. M. Hossein Mousavi
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import sklearn.cluster as cl
import sklearn.mixture as mx

# Data Loading
IRIS = dt.load_iris()
# Separating data and target 
X = IRIS.data
Y = IRIS.target

# KMeans Clustering
KMN = cl.KMeans(n_clusters = 3)
KMN.fit(X)
Labels = KMN.labels_
Centroids = KMN.cluster_centers_
# Mean Shift Clustering
MS = cl.MeanShift(bandwidth = 0.85)
MS.fit(X)
MSLabels = MS.labels_
MSCentroids = MS.cluster_centers_
# DBSCAN Clustering
dbscan = cl.DBSCAN(eps = 0.41, min_samples = 2)
dbscan.fit(X)
DBLabels = dbscan.labels_
Noises = Labels==-1
# GMM Clustering
GMM = mx.GaussianMixture(n_components = 3)
GMM.fit(X)
Means = GMM.means_
GMMLabels = GMM.predict(X)

# Plot KMeans
plt.figure(figsize=(18,13))
plt.subplot(2, 3, 1)
plt.scatter(X[:, 2], X[:, 3], s = 60, c = Labels)
plt.scatter(Centroids[:, 2], Centroids[:, 3], s = 100, c = 'r', marker = 'x')
plt.xlabel('X',fontsize=18)
plt.ylabel('Y',fontsize=18)
plt.title('K-Means Clustering',fontsize=18)
# Plot Mean Shift
plt.subplot(2, 3, 2)
plt.scatter(X[:, 2], X[:, 3], s = 60, c = MSLabels)
plt.scatter(MSCentroids[:, 2], MSCentroids[:, 3], s = 100, c = 'r', marker = 'x')
plt.xlabel('X',fontsize=18)
plt.ylabel('Y',fontsize=18)
plt.title('Mean Shift Clustering',fontsize=18)
# Plot DBSCAN
plt.subplot(2, 3, 3)
plt.scatter(X[:, 2], X[:, 3], s = 60, c = DBLabels)
plt.scatter(X[Noises, 2], X[Noises, 3], s = 100, c = 'r', marker = 'x')
plt.xlabel('X',fontsize=18)
plt.ylabel('Y',fontsize=18)
plt.title('DBSCAN Clustering',fontsize=18)
# Plot DBSCAN
plt.subplot(2, 3, 4)
plt.scatter(X[:, 2], X[:, 3], s = 60, c = GMMLabels)
plt.scatter(Means[:, 2], Means[:, 3], s = 100, c = 'r', marker = 'x')
plt.xlabel('X',fontsize=18)
plt.ylabel('Y',fontsize=18)
plt.title('GMM Clustering',fontsize=18)
# Plot Original
plt.subplot(2, 3, 5)
plt.scatter(X[:, 2], X[:, 3], s = 60, c = Y)
plt.xlabel('X',fontsize=18)
plt.ylabel('Y',fontsize=18)
plt.title('Original Iris',fontsize=18)
# Show All Plots
plt.show()