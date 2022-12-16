# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 12:49:01 2022
@author: S. M. Hossein Mousavi
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import random
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as met
import scipy.stats as stt
import sklearn.neighbors as ne
import sklearn.svm as sv
import sklearn.tree as tr
import sklearn.neural_network as nn

# Data Creation
N = 1000 # Number of Samples
X = np.zeros((N, 5)) # Number of Features
Y = np.zeros((N, 1))
for i in range(0, N):
    X[i, 0] = np.random.rand()
    X[i, 1] = np.random.rand()
    X[i, 2] = np.random.rand()
    X[i, 3] = np.random.rand()
    X[i, 4] = np.random.rand()
    Y[i, 0] = 2*X[i, 0] - 1.2*X[i, 1] + 0.5 + np.random.randn()/25 # -1 >> 2.8
# Data Split
Xtr, Xte, Ytr, Yte = ms.train_test_split(X, Y, train_size = 0.7, random_state = 10)

# Linear Regression
LM = lm.LinearRegression()
LM.fit(Xtr, Ytr)
trPred = LM.predict(Xtr)
tePred = LM.predict(Xte)
trMSE = met.mean_squared_error(Ytr, trPred)
teMSE = met.mean_squared_error(Yte, tePred)
trPCC, _ = stt.pearsonr(Ytr[:, 0], trPred[:, 0])
tePCC, _ = stt.pearsonr(Yte[:, 0], tePred[:, 0])
# KNN Regression
KNN = ne.KNeighborsRegressor(n_neighbors = 3, weights = 'distance')
KNN.fit(Xtr, Ytr)
KNNtrPred = KNN.predict(Xtr)
KNNtePred = KNN.predict(Xte)
KNNtrMSE = met.mean_squared_error(Ytr, KNNtrPred)
KNNteMSE = met.mean_squared_error(Yte, KNNtePred)
KNNtrPCC, _ = stt.pearsonr(Ytr[:, 0], KNNtrPred[:, 0])
KNNtePCC, _ = stt.pearsonr(Yte[:, 0], KNNtePred[:, 0])
# SVM Regression
SVM = sv.SVR(kernel = 'rbf')
SVM.fit(Xtr, Ytr)
SVMtrPred = SVM.predict(Xtr)
SVMtePred = SVM.predict(Xte)
SVMtrMSE = met.mean_squared_error(Ytr, SVMtrPred)
SVMteMSE = met.mean_squared_error(Yte, SVMtePred)
SVMtrPCC, _ = stt.pearsonr(Ytr[:, 0], SVMtrPred)
SVMtePCC, _ = stt.pearsonr(Yte[:, 0], SVMtePred)
# Decision Tree Regression
DT = tr.DecisionTreeRegressor(max_depth = 5, random_state = 10)
DT.fit(Xtr, Ytr)
DTtrPred = DT.predict(Xtr)
DTtePred = DT.predict(Xte)
DTtrMSE = met.mean_squared_error(Ytr, DTtrPred)
DTteMSE = met.mean_squared_error(Yte, DTtePred)
DTtrPCC, _ = stt.pearsonr(Ytr[:, 0], DTtrPred)
DTtePCC, _ = stt.pearsonr(Yte[:, 0], DTtePred)
# MLP Regression
NN = nn.MLPRegressor(hidden_layer_sizes = (40, 50),
                        activation = 'relu',
                        solver = 'adam',
                        learning_rate_init = 1e-3,
                        max_iter = 500,
                        random_state = 48)
NN.fit(Xtr, Ytr)
NNtrPred = NN.predict(Xtr)
NNtePred = NN.predict(Xte)
NNtrMSE = met.mean_squared_error(Ytr, NNtrPred)
NNteMSE = met.mean_squared_error(Yte, NNtePred)
NNtrPCC, _ = stt.pearsonr(Ytr[:, 0], NNtrPred)
NNtePCC, _ = stt.pearsonr(Yte[:, 0], NNtePred)

# Plots
plt.figure(figsize=(18,13))
plt.subplot(2, 3, 1)
plt.scatter(Ytr, trPred, label = 'Train', s = 30)
plt.scatter(Yte, tePred, label = 'Test', s = 30)
plt.plot([-1, +2.8], [-1, +2.8], label = 'x = y', c = 'r')
plt.xlabel('Traget Values',fontsize=18)
plt.ylabel('Predicted Values',fontsize=18)
plt.title('Linear Regression',fontsize=18)
plt.legend(fontsize=12)
plt.subplot(2, 3, 2)
plt.scatter(Ytr, KNNtrPred, label = 'Train', s = 30)
plt.scatter(Yte, KNNtePred, label = 'Test', s = 30)
plt.plot([-1, +2.8], [-1, +2.8], label = 'x = y', c = 'r')
plt.xlabel('Traget Values',fontsize=18)
plt.ylabel('Predicted Values',fontsize=18)
plt.title('KNN Regression',fontsize=18)
plt.legend(fontsize=12)
plt.subplot(2, 3, 3)
plt.scatter(Ytr, SVMtrPred, label = 'Train', s = 30)
plt.scatter(Yte, SVMtePred, label = 'Test', s = 30)
plt.plot([-1, +2.8], [-1, +2.8], label = 'x = y', c = 'r')
plt.xlabel('Traget Values',fontsize=18)
plt.ylabel('Predicted Values',fontsize=18)
plt.title('SVM Regression',fontsize=18)
plt.legend(fontsize=12)
plt.subplot(2, 3, 4)
plt.scatter(Ytr, DTtrPred, label = 'Train', s = 30)
plt.scatter(Yte, DTtePred, label = 'Test', s = 30)
plt.plot([-1, +2.8], [-1, +2.8], label = 'x = y', c = 'r')
plt.xlabel('Traget Values',fontsize=18)
plt.ylabel('Predicted Values',fontsize=18)
plt.title('Decision Tree Regression',fontsize=18)
plt.legend(fontsize=12)
plt.subplot(2, 3, 5)
plt.scatter(Ytr, NNtrPred, label = 'Train', s = 30)
plt.scatter(Yte, NNtePred, label = 'Test', s = 30)
plt.plot([-1, +2.8], [-1, +2.8], label = 'x = y', c = 'r')
plt.xlabel('Traget Values',fontsize=18)
plt.ylabel('Predicted Values',fontsize=18)
plt.title('MLP Regression',fontsize=18)
plt.legend(fontsize=12)
plt.show()

# Statistical Results (MSE and Correlation Coefficient)
print('Linear Regression')
print('-----------------------------------')
print('Linear Regression Train MSE: ', trMSE)
print('Linear Regression Test MSE: ', teMSE)
print('Linear Regression Train PCC: ', trPCC)
print('Linear Regression Test PCC: ', tePCC)
print('-----------------------------------')
print('KNN')
print('-----------------------------------')
print('KNN Train MSE: ', KNNtrMSE)
print('KNN Test MSE: ', KNNteMSE)
print('KNN Train PCC: ', KNNtrPCC)
print('KNN Test PCC: ', KNNtePCC)
print('-----------------------------------')
print('SVM')
print('-----------------------------------')
print('SVM Train MSE: ', SVMtrMSE)
print('SVM Test MSE: ', SVMteMSE)
print('SVM Train PCC: ', SVMtrPCC)
print('SVM Test PCC: ', SVMtePCC)
print('-----------------------------------')
print('Decision Tree')
print('-----------------------------------')
print('Decision Tree Train MSE: ', DTtrMSE)
print('Decision Tree Test MSE: ', DTteMSE)
print('Decision Tree Train PCC: ', DTtrPCC)
print('Decision Tree Test PCC: ', DTtePCC)
print('-----------------------------------')
print('Multi Layer Perceptron (MLP)')
print('-----------------------------------')
print('MLP Train MSE: ', NNtrMSE)
print('MLP Test MSE: ', NNteMSE)
print('MLP Train PCC: ', NNtrPCC)
print('MLP Test PCC: ', NNtePCC)
print('-----------------------------------')