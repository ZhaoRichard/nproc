# -*- coding: utf-8 -*-

import numpy as np
import os
from npc import npc


test = npc()

np.random.seed(1)

n = 10000
x = np.random.normal(0, 1, (n,2))
c = 1+3*x[:,0]
y = np.random.binomial(1, 1/(1+np.exp(-c)), n)  #binomial(number of trials, probability of success, num of observations)

fit = test.npc(x, y, 'logistic', n_cores=os.cpu_count())


x_test = np.random.normal(0, 1, (n,2))
c_test = 1+3*x_test[:,0]
y_test = np.random.binomial(1, 1/(1+np.exp(-c_test)), n)

pred_score = test.predict(fit,x_test)
fitted_score = test.predict(fit,x)

print(np.mean(pred_score[0]==y_test))
print(np.mean(fitted_score[0]==y))