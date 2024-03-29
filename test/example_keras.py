# -*- coding: utf-8 -*-
# Example of Neyman-Pearson (NP) Classification Algorithms
# Richard Zhao, Yang Feng, Jingyi Jessica Li and Xin Tong

import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from nproc import npc


test = npc()

np.random.seed()

# Create a dataset (x,y) with 2 features, binary label and sample size 10000.
n = 10000
x = np.random.normal(0, 1, (n,2))    
c = 1+3*x[:,0]
y = np.random.binomial(1, 1/(1+np.exp(-c)), n)  #binomial(number of trials, probability of success, num of observations)

# Plot the dataset
plt.scatter(x[:,0], x[:,1], c=y)
plt.show()

# Create custom deep neural network model using keras
nn_model = Sequential()
nn_model.add(Dense(8, input_dim=2, activation='relu'))
nn_model.add(Dense(8, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Call the npc function to construct Neyman-Pearson classifiers.
fit = test.npc(x, y, method="keras", model=nn_model, n_cores=os.cpu_count(), alpha=0.1)

# Evaluate the prediction of the NP classifier fit on a test set (xtest, ytest).
x_test = np.random.normal(0, 1, (n,2))
c_test = 1+3*x_test[:,0]
y_test = np.random.binomial(1, 1/(1+np.exp(-c_test)), n)

# Calculate the overall accuracy of the classifier as well as the realized 
# type I error rate on test data.
# Strictly speaking, to demonstrate the effectiveness of the fit classifier 
# under the NP paradigm, we should repeat this experiment many times, and 
# show that in 1 - delta of these repetitions, type I error rate is smaller than alpha.

fitted_score = test.predict(fit,x)
print("Accuracy on training set:", accuracy_score(fitted_score[0], y))
pred_score = test.predict(fit,x_test)
print("Accuracy on test set:", accuracy_score(pred_score[0], y_test))

cm = confusion_matrix(y_test, pred_score[0])
print("Confusion matrix:")
print(cm)
tn, fp, fn, tp = cm.ravel()
print("Type I error rate: {:.5f}".format(fp/(fp+tn)))
print("Type II error rate: {:.5f}".format(fn/(fn+tp)))
