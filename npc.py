#!/usr/bin/env python

__author__ = "Richard Zhao, Yang Feng, Jingyi Jessica Li and Xin Tong"

import scipy.stats as ss
from scipy.stats import binom
from numpy import *

class npc:
    
    

    
    # Given a type I error upper bound alpha and a violation upper bound delta, 
    # npc calculates the Neyman-Pearson Classifier which controls the type I error 
    # under alpha with probability at least 1-delta.
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

            
        if split_ratio == 'adaptive':
            ret = self.find_optim_split(x, y, method, alpha, delta, split, 10, band, rand_seed)
            split_ratio = ret[0]
            errors = ret[2]
        else:
            errors = 0
            
        random.seed(rand_seed)
        
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
    
                indices0train = random.choice(indices0, num0, replace = False)
                indices1train = random.choice(indices1, num1, replace = False)
    
                indices0test = [item for item in indices0 if item not in indices0train]
                indices1test = [item for item in indices1 if item not in indices1train]
                
                if rand_seed is not None:
                    random.seed(rand_seed+i)
                
                if (band == True):
                    
                    fits.append(
                            self.npc_split(x, y, p, alpha, delta, indices0train, indices0test,
                              indices1train, indices1test, method, n_cores))
                else:
                    fits.append(
                            self.npc_split(x, y, p, alpha, delta, indices0train, indices0test,
                              indices1, indices1, method, n_cores))

    
        ret = [fits, method, split, split_ratio, errors]
        return ret
        
    
    
    # Find the optimal split
    def find_optim_split(self, x, y, method, alpha, delta, split, n_folds, band, rand_seed):
        return [split_ratio_min, split_ratio_1se, error_m, error_se]
    
    
    # NPC split
    def npc_split(self, x, y, p, alpha, delta, indices0train, indices0test, indices1train, indices1test, method, n_cores):
        return []
    
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
        
        v_list = arange(0, 1.001, 0.001)
        
        r_lower0 = ss.rankdata(score0, method='min')
        r_upper0 = ss.rankdata(score0, method='max')

        def r_lower_helper(s):
            return sum(i <= s for i in score1)

        def r_upper_helper(s):
            return sum(i < s for i in score1) + max(1,sum(i == s for i in score1))
        

        r_lower1 = [r_lower_helper(s) for s in scores]
        r_upper1 = [r_upper_helper(s) for s in scores]

        #print(r_lower1)
        #print(r_upper1)
        
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
            if (r_upper1[s] >= len1):
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
        
        
        
        alpha = [alpha_helper(s) for s in list(range(0, len0))]
        
        alpha = array(alpha)
        alpha_l_list = alpha[:,0]
        alpha_u_list = alpha[:,1]
        beta_l_list = alpha[:,2]
        beta_u_list = alpha[:,3]

        
        return [scores, beta_l_list, beta_u_list, alpha_l_list, alpha_u_list, r_upper1, r_lower1, r_upper0, r_lower0]




test = npc()
#result = test.find_order(list(range(1, 101)), list(range(200, 301)))
result = test.npc([[1,2,3,4,5],[2,3,4,5,6]], [0,0,0,1,1], 'np')
