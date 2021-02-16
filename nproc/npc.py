#!/usr/bin/env python

__author__ = "Richard Zhao, Yang Feng, Jingyi Jessica Li and Xin Tong"

import scipy.stats as ss
import numpy as np
import math
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
#import multiprocessing


class npc:
    

    # Given a type I error upper bound alpha and a violation upper bound delta, 
    # npc calculates the Neyman-Pearson Classifier which controls the type I error 
    # under alpha with probability at least 1-delta.
    # returns a result containing the fits with extra information
    def npc(self, x, y, method, alpha = None, delta = None, split = None, split_ratio = None, n_cores = None, band = None, rand_seed = None):
        
        if alpha is None:
            alpha = 0.05
        if delta is None:
            delta = 0.05
        if split is None:
            split = 1
        if split_ratio is None:
            split_ratio = 0.5
        if n_cores is None:
            n_cores = 1
        if band is None:
            band = False
        if rand_seed is None:
            rand_seed = 0

        if split_ratio == 'adaptive':
            ret = self.find_optim_split(x, y, method, alpha, delta, split, 10, band, rand_seed)
            split_ratio = ret[0]
            errors = ret[2]
        else:
            errors = 0
        
        np.random.seed(rand_seed)
        
        p = len(x[0])
        if p == 1 and method == 'penlog':
            print ("The one predictor case is not supported.")
            return 
        
        
        #ind0 = which(y == 0)  ##indices for class 0
        indices0 =  [index for index, item in enumerate(y) if item == 0]
        #ind1 = which(y == 1)  ##indices for class 1
        indices1 =  [index for index, item in enumerate(y) if item == 1]
        
        len0 = len(indices0)
        len1 = len(indices1)
        
        if split == 0:
            # no split, use all class 0 obs for training and for calculating the cutoff
            fits = self.npc_split(x, y, p, alpha, delta, indices0, indices0, indices1, indices1, method, n_cores)
        else:
            # with split
            num0 = round(len0 * split_ratio)  #default size for calculating the classifier
            num1 = round(len1 * split_ratio)
            
            fits = []
            
            for i in range(split):
    
                indices0train = np.random.choice(indices0, num0, replace = False).tolist()
                indices1train = np.random.choice(indices1, num1, replace = False).tolist()
    
                indices0set = set(indices0train)
                indices0test = [item for item in indices0 if item not in indices0set]
                indices1set = set(indices1train)
                indices1test = [item for item in indices1 if item not in indices1set]
        
        
                if rand_seed is not None:
                    np.random.seed(rand_seed+i)
                
                if (band == True):
                    
                    fits.append(
                            self.npc_split(x, y, p, alpha, delta, indices0train, indices0test,
                              indices1train, indices1test, method, n_cores))
                else:
                    fits.append(
                            self.npc_split(x, y, p, alpha, delta, indices0train, indices0test,
                              indices1, indices1, method, n_cores))

    
        res = [fits, method, split, split_ratio, errors]
        return res
        
    
    
    # Find the optimal split
    def find_optim_split(self, x, y, method, alpha, delta, split, n_folds, band, rand_seed):
        # TODO
        split_ratio_min = 0
        split_ratio_1se = 0
        error_m = 0
        error_se = 0
        return [split_ratio_min, split_ratio_1se, error_m, error_se]
    
    

    
    # NPC split
    # returns a fit
    def npc_split(self, x, y, p, alpha, delta, indices0train, indices0test, indices1train, indices1test, method, n_cores):
        
        indices_train = indices0train + indices1train
        indices_test = indices0test + indices1test
                
        x_train = [x[index] for index in indices_train]
        y_train = [y[index] for index in indices_train]
        
        x_test = [x[index] for index in indices_test]
        y_test = [y[index] for index in indices_test]
        
        class_data = self.classification(method, x_train, y_train, x_test)           
        if class_data == []:
            return []
        
        fit_model=class_data[0]
        y_decision_values= class_data[1]
        
        obj = self.npc_core(y_test, y_decision_values, alpha, delta, n_cores)
        
        cutoff = obj[0]
        sign = obj[1]
        beta_l_list = obj[2]
        beta_u_list = obj[3]
        alpha_l_list = obj[4]
        alpha_u_list = obj[5]
        n_small = obj[6]
        return [fit_model, y_test, y_decision_values, cutoff, sign, method, beta_l_list, beta_u_list, alpha_l_list, alpha_u_list, n_small]
 
    
    def classification(self, method, x_train, y_train, x_test):
        
        #print (x_train)
        #print (y_train)
        #print (x_test)

        if method == 'logistic':
            clf_logistic = LogisticRegression()
            clf_logistic.fit(x_train, y_train)
            fit_model = clf_logistic
            test_score = clf_logistic.predict_proba(x_test)[:,1]
        elif method == 'svm':
            clf_SVM = SVC(probability=True)
            clf_SVM.fit(x_train, y_train)
            fit_model = clf_SVM
            test_score = clf_SVM.predict_proba(x_test)[:,1]
        elif method == 'nb':
            clf_nb = GaussianNB()
            clf_nb.fit(x_train, y_train)
            fit_model = clf_nb
            test_score = clf_nb.predict_proba(x_test)[:,1]
        elif method == 'nb_m':
            clf_nb = MultinomialNB()
            clf_nb.fit(x_train, y_train)
            fit_model = clf_nb
            test_score = clf_nb.predict_proba(x_test)[:,1]
        elif method == 'rf':
            clf_rf = RandomForestClassifier()
            clf_rf.fit(x_train, y_train)
            fit_model = clf_rf
            test_score = clf_rf.predict_proba(x_test)[:,1]
        elif method == 'dt':
            clf_dt = DecisionTreeClassifier()
            clf_dt.fit(x_train, y_train)
            fit_model = clf_dt
            test_score = clf_dt.predict_proba(x_test)[:,1]
        #TODO: more methods
        else:
            print("Method not supported.")
            return []
        
        
        return [fit_model, test_score]

        
    def npc_core(self, y_test, y_decision_values, alpha, delta, n_cores):
        
        #ind0 = which(y == 0)  ##indices for class 0
        indices0 =  [index for index, item in enumerate(y_test) if item == 0]
        #ind1 = which(y == 1)  ##indices for class 1
        indices1 =  [index for index, item in enumerate(y_test) if item == 1]
        
        if len(indices0) == 0 or len(indices1) == 0:
            print("Both class 0 and class 1 responses are needed to decide the cutoff.")
            return []
        
        #whether the class 0 has a larger average score than class 1
        test_list0 = [y_decision_values[index] for index in indices0]
        test_list1 = [y_decision_values[index] for index in indices1]
        
        
        sign = np.mean(test_list0) > np.mean(test_list1) 
    
        if sign == False:
            y_decision_values = [ -y for y in y_decision_values]
            

        
        obj = self.find_order(test_list0, test_list1, delta, n_cores)
        
        cutoff_list = obj[0]

        beta_l_list = obj[1]
        beta_u_list = obj[2]
        alpha_l_list = obj[3]
        alpha_u_list = obj[4]
        #alpha_u_len = len(alpha_u_list)
        alpha_u_min = min(alpha_u_list)
        
        n_small = False;
        
        if alpha != None:
            if alpha_u_min > alpha + 1e-10:
                cutoff = math.inf
                loc = len(indices0)
                n_small = True
                print ('Sample size is too small for the given alpha. Try a larger alpha.')
                
            else:
                #loc = min(which(obj$alpha.u <= alpha + 1e-10))
                temp_list = [index for index, item in enumerate(alpha_u_list) if item <= alpha + 1e-10]
                loc = min(temp_list)
                cutoff = cutoff_list[loc]
  
        
        return [cutoff, sign, beta_l_list, beta_u_list, alpha_l_list, alpha_u_list, n_small]
    
    
    # find the order such that the type-I error bound is satisfied with probability 
    # at least 1-delta
    def find_order(self, score0, score1 = None, delta = None, n_cores = None):
        
        if delta is None:
            delta = 0.05
        if n_cores is None:
            n_cores = 1
        
        score0 = sorted(score0)
        score1 = sorted(score1)
        
        
        len0 = len(score0)
        len1 = len(score1)
      
        scores = score0
        #alpha_l_list: type I error lower bound
        #alpha_u_list: type I error upper bound
        #beta_l_list: type II error lower bound
        #beta_u_list: type II error upper bound
        alpha_l_list = [0] * len0
        alpha_u_list = [0] * len0
        beta_l_list = [0] * len0
        beta_u_list = [0] * len0
        
        v_list = np.arange(0, 1.001, 0.001)
        
        r_lower0 = ss.rankdata(score0, method='min')
        r_upper0 = ss.rankdata(score0, method='max')

        # Count the number of items in score1 that are less than each item in scores
        
        # original approach
        #def r_lower_helper(s):
        #    return sum(i <= s for i in score1)
        #def r_upper_helper(s):
        #    return sum(i < s for i in score1) + max(1,sum(i == s for i in score1))
        #r_lower1 = [r_lower_helper(s) for s in scores]        
        #r_upper1 = [r_upper_helper(s) for s in scores]
        
        # parallel approach
        #n_cores = multiprocessing.cpu_count()
        #r_lower1 = Parallel(n_jobs=n_cores)(delayed(r_lower_helper)(s) for s in scores)
        #r_upper1 = Parallel(n_jobs=n_cores)(delayed(r_upper_helper)(s) for s in scores)

        r_lower1 =[0] * len(scores)
        r_upper1 =[0] * len(scores)
        
        r_lower_index = 0
        r_upper_index = 0

        score1_index_l = 0
        score1_index_u = 0
        
        for s in scores:
            while score1[score1_index_l] <= s:
                if score1_index_l == len1-1:
                    break
                score1_index_l+=1
            
            r_lower1[r_lower_index] = score1_index_l
            r_lower_index+=1
        
            while score1[score1_index_u] <= s:
                if score1_index_u == len1-1:
                    break                
                score1_index_u+=1
            
            equal = False
            if score1_index_u > 0 and score1[score1_index_u-1] == s:
                equal = True
                
            r_upper1[r_upper_index] = score1_index_u
            if equal == False:
                r_upper1[r_upper_index]+=1
                
            r_upper_index+=1
        

        
        
        def alpha_helper(s):
            #alpha_u = v_list[min(which(pbinom(n0 - ru0[s], len0, v_list) <= delta))]
            prob = binom.cdf(len0-r_upper0[s], len0, v_list)
            for index, item in enumerate(prob):
                if item <= delta:
                    alpha_u = v_list[index]
                    break
            

            if r_upper0[s] == len0:
                alpha_l = 0
            else:
                #alpha_l = v_list[max(which(pbinom(n0 - rl0[i], n0, v_list) >= 1 - delta))]      
                prob = binom.cdf(len0-r_lower0[s], len0, v_list)
                for index, item in reversed(list(enumerate(prob))):
                    if item >= 1-delta:
                        alpha_l = v_list[index]
                        break
                    
            if r_upper1[s] >= len1:
                #beta_l = v_list[max(which(pbinom(rl1[i]-1, n1, v_list) >= 1 - delta))]    
                prob = binom.cdf(r_lower1[s]-1, len1, v_list)
                for index, item in reversed(list(enumerate(prob))):
                    if item >= 1-delta:
                        beta_l = v_list[index]
                        break            
                beta_u = 1
            elif r_lower1[s] == 0:
                beta_l = 0
                
                #beta_u = v_list[min(which(pbinom(ru1[i]-1, n1, v_list) <= delta))]
                prob = binom.cdf(r_upper1[s]-1, len1, v_list)
                for index, item in enumerate(prob):
                    if item <= delta:
                        beta_u = v_list[index]
                        break            
            else:
                #beta_l = v_list[max(which(pbinom(rl1[i]-1, n1, v_list) >= 1 - delta))]
                prob = binom.cdf(r_lower1[s]-1, len1, v_list)
                for index, item in reversed(list(enumerate(prob))):
                    if item >= 1-delta:
                        beta_l = v_list[index]
                        break        
            
                #beta_u = v_list[min(which(pbinom(ru1[i]-1, n1, v_list) <= delta))]            
                prob = binom.cdf(r_upper1[s]-1, len1, v_list)
                for index, item in enumerate(prob):
                    if item <= delta:
                        beta_u = v_list[index]
                        break     
            
            #print ( [alpha_l, alpha_u, beta_l, beta_u])
            return [alpha_l, alpha_u, beta_l, beta_u]
        
        
        alpha = Parallel(n_jobs=n_cores)(delayed(alpha_helper)(s) for s in list(range(0, len0)))

        alpha = np.array(alpha)
        alpha_l_list = alpha[:,0]
        alpha_u_list = alpha[:,1]
        beta_l_list = alpha[:,2]
        beta_u_list = alpha[:,3]

        
        return [scores, beta_l_list, beta_u_list, alpha_l_list, alpha_u_list, r_upper1, r_lower1, r_upper0, r_lower0]


    # Predicting the outcome of a set of new observations using the fitted npc object.
    def predict(self, result, newx):
        
        label = []
        score = []
        split = result[2]
        
        if split < 1:
            pred = self.pred_npc_core(result, newx)
            label = pred[0]
            score = pred[1]
        else:
            pred = self.pred_npc_core(result[0][0], newx)
            label = pred[0]
            score = pred[1]
            
            if split > 1:
                for i in range(2, split):
                    pred = self.pred_npc_core(result[0][i], newx)
                    label += pred[0]
                    score += pred[1]
                    
            for i in range(len(label)):
                if label[i]/split > 0.5:
                    label[i] = 1
                else:
                    label[1] = 0

            for i in range(len(score)):
                score[i] = score[i] / split
            
        return [label, score]
        
    # pred.npc.core    
    def pred_npc_core(self, fit, newx):
        
        #method = fit[5]
        fit_model = fit[0]
        cutoff = fit[3]
        label = []
        #score = []

        score = fit_model.predict_proba(newx)[:,1]


        for i in range(len(score)):
            if score[i] > cutoff:
                label.append(1)
            else:
                label.append(0)
                
        return [label, score]

