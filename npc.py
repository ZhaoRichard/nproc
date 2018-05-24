
import scipy.stats as ss
from scipy.stats import binom
from numpy import *

class npc:
    
    # find the order such that the type-I error bound is satisfied with probability at
    # least 1-delta
    def find_order(score0, score1 = None, delta = None, n_cores = None):
        if delta is None:
            delta = 0.05
        if n_cores is None:
            n_cores = 1
        
        score0 = sorted(score0)
        score1 = sorted(score1)
        len0 = len(score0)
        len1 = len(score1)
      
        scores = score0
        #alpha_l: type I error lower bound
        #alpha_u: type I error upper bound
        #beta_l: type II error lower bound
        #beta_u: type II error upper bound
        alpha_l = [0] * len0
        alpha_u = [0] * len0
        beta_l = [0] * len0
        beta_u = [0] * len0
        
        v_list = arange(0, 1.001, 0.001)
        
        rlower0 = ss.rankdata(score0, method='min')
        rupper0 = ss.rankdata(score0, method='max')


        def func1(s):
            return sum(i < s for i in score1) + max(1,sum(i == s for i in score1))
        
        def func2(s):
            return sum(i <= s for i in score1)

        rlower1 = [func2(s) for s in scores]
        rupper1 = [func1(s) for s in scores]

        #print(rlower1)
        #print(rupper1)
        
        def func3(s):
            #alphau = v_list[min(which(pbinom(n0 - ru0[s], len0, v_list) <= delta))]
            prob = binom.cdf(len0-rupper0[s], len0, v_list)
            for index, item in enumerate(prob):
                if item <= delta:
                    alphau = v_list[index]
                    break
            


            if rupper0[s] == len0:
                alphal = 0
            else:
                #alphal = v_list[max(which(pbinom(n0 - rl0[i], n0, v.list) >= 1 - delta))]      
                prob = binom.cdf(len0-rlower0[s], len0, v_list)
                for index, item in reversed(list(enumerate(prob))):
                    if item >= 1-delta:
                        alphal = v_list[index]
                        break
            if (rupper1[s] >= len1):
                #betal = v_list[max(which(pbinom(rl1[i]-1, n1, v.list) >= 1 - delta))]    
                prob = binom.cdf(rlower1[s]-1, len1, v_list)
                for index, item in reversed(list(enumerate(prob))):
                    if item >= 1-delta:
                        betal = v_list[index]
                        break            
                
                betau = 1
                
            elif rlower1[s] == 0:
                betal = 0
                
                #betau = v_list[min(which(pbinom(ru1[i]-1, n1, v.list) <= delta))]
                prob = binom.cdf(rupper1[s]-1, len1, v_list)
                for index, item in enumerate(prob):
                    if item <= delta:
                        betau = v_list[index]
                        break            
                
            else:
                #betal = v_list[max(which(pbinom(rl1[i]-1, n1, v.list) >= 1 - delta))]
                prob = binom.cdf(rlower1[s]-1, len1, v_list)
                for index, item in reversed(list(enumerate(prob))):
                    if item >= 1-delta:
                        betal = v_list[index]
                        break        
            
                #betau = v_list[min(which(pbinom(ru1[i]-1, n1, v.list) <= delta))]            
                prob = binom.cdf(rupper1[s]-1, len1, v_list)
                for index, item in enumerate(prob):
                    if item <= delta:
                        betau = v_list[index]
                        break     
            
            #print ( [alphal, alphau, betal, betau])
            return [alphal, alphau, betal, betau]
        
        alpha = [func3(s) for s in list(range(0, len0))]
        
        #print(alpha)
        alpha = array(alpha)
        alpha_l = alpha[:,0]
        alpha_u = alpha[:,1]
        beta_l = alpha[:,2]
        beta_u = alpha[:,3]

        
        return [scores, beta_l, beta_u, alpha_l, alpha_u, rupper1, rlower1, rupper0, rlower0]


test = npc.find_order(list(range(1, 101)), list(range(200, 301)))
            