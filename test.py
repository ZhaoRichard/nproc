# -*- coding: utf-8 -*-


from numpy import *
from npc import npc

test = npc()
#result = test.find_order(list(range(1, 101)), list(range(200, 301)))
#result = test.npc([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]], [0,0,0,1,1,1], 'svm')


random.seed(1)

n = 1000
x = random.normal(0, 1, (n,2))
c = 1+3*x[:,0]
y = random.binomial(1, 1/(1+exp(-c)), n)  #binomial(number of trials, probability of success, num of observations)


fit = test.npc(x, y, 'svm')


x_test = random.normal(0, 1, (n,2))
c_test = 1+3*x_test[:,0]
y_test = random.binomial(1, 1/(1+exp(-c_test)), n)

pred_score = test.predict(fit,x_test)
fitted_score = test.predict(fit,x)

print(mean(pred_score[0]==y_test))
print(mean(fitted_score[0]==y))