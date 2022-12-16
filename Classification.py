# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:57:24 2022
@author: S. M. Hossein Mousavi
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import sklearn.model_selection as ms
import sklearn.neighbors as ne
import sklearn.linear_model as lm
import sklearn.naive_bayes as nb
import sklearn.tree as tr
import sklearn.svm as sv
import sklearn.neural_network as nn
import warnings

# Suppressing All Warnings
warnings.filterwarnings("ignore")
# Data Loading
IRIS = dt.load_iris()
# Data Split
X = IRIS.data
Y = IRIS.target
Xtr, Xte, Ytr, Yte = ms.train_test_split(X, Y, train_size = 0.8)
# KNN Classifier
trAcc=[]
teAcc=[]
Ks=[]
for i in range(1,5):
    KNN = ne.KNeighborsClassifier(n_neighbors = i)
    KNN.fit(Xtr, Ytr)
    trAcc.append(KNN.score(Xtr, Ytr))
    teAcc.append(KNN.score(Xte, Yte))
    Ks.append(i)
# Logistic Regression Classifier
LR = lm.LogisticRegression(max_iter = 100)
LR.fit(Xtr, Ytr)
LRtrAcc = LR.score(Xtr, Ytr)
LRteAcc = LR.score(Xte, Yte)
# Naive Bayes Classifier
NB = nb.GaussianNB()
NB.fit(Xtr, Ytr)
NBtrAcc = NB.score(Xtr, Ytr)
NBteAcc = NB.score(Xte, Yte)
# Decision Tree Classifier
DTtrAcc = []
DTteAcc = []
MD = []
for i in range(2, 12):
    DT = tr.DecisionTreeClassifier(max_depth = i)
    DT.fit(Xtr, Ytr)
    DTtrAcc.append(DT.score(Xtr, Ytr))
    DTteAcc.append(DT.score(Xte, Yte))
    MD.append(i)
# SVM Classifier
SVMtrAcc = []
SVMteAcc = []
Kernel = []
for i in ['linear', 'poly', 'rbf', 'sigmoid']:
    Clsfr = sv.SVC(kernel = i, degree = 2)
    Clsfr.fit(Xtr, Ytr)
    SVMtrAcc.append(Clsfr.score(Xtr, Ytr))
    SVMteAcc.append(Clsfr.score(Xte, Yte))
    Kernel.append(i)
# MLP Classifier
MLPtrAcc = []
MLPteAcc = []
HLS = []
for i in [15, 25]:
    for j in [24, 30]:
        NN = nn.MLPClassifier(hidden_layer_sizes = (i, j), activation = 'relu', solver = 'adam', max_iter = 300, alpha = 0.001)
        NN.fit(Xtr, Ytr)
        MLPtrAcc.append(NN.score(Xtr, Ytr))
        MLPteAcc.append(NN.score(Xte, Yte))
        HLS.append([i, j])
# Train and Test Results
print ('KNN Train Accuracy is :')
print (trAcc[-1])
print ('KNN Test Accuracy is :')
print (teAcc[-1])

print('Logestic Regression Train Accuracy is : ')
print (LRtrAcc)
print('Logestic Regression Test Accuracy is :')
print (LRteAcc)

print('Naive Bayes Train Accuracy is :')
print (NBtrAcc)
print('Naive Bayes Test Accuracy is :')
print (NBteAcc)

print('Decision Tree Train Accuracy is :')
print (DTtrAcc[-1])
print('Decision Tree Test Accuracy is :')
print (DTteAcc[-1])

print('SVM Train Accuracy is :')
print (SVMtrAcc[1])
print('SVM Test Accuracy is :')
print (SVMteAcc[1])

print('MLP Train Accuracy is :')
print (MLPtrAcc[-1])
print('MLP Test Accuracy is :')
print (MLPteAcc[-1])