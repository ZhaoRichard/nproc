# nproc: Neyman-Pearson (NP) Classification Algorithms and NP Receiver Operating Characteristic (NP-ROC) Curves

In many binary classification applications, such as disease diagnosis and spam detection, practitioners commonly face the need to limit type I error rate (i.e., the conditional probability of misclassifying a class 0 observation as class 1) so that it remains below a desired threshold. To address this need, the Neyman-Pearson (NP) classification paradigm is a natural choice; it minimizes type II error rate (i.e., the conditional probability of misclassifying a class 1 observation as class 0) while enforcing an upper bound, alpha, on the type I error rate. Although the NP paradigm has a century-long history in hypothesis testing, it has not been well recognized and implemented in classification schemes. Common practices that directly limit the empirical type I error rate to no more than alpha do not satisfy the type I error rate control objective because the resulting classifiers are still likely to have type I error rates much larger than alpha. As a result, the NP paradigm has not been properly implemented for many classification scenarios in practice. In this work, we develop the first umbrella algorithm that implements the NP paradigm for all scoring-type classification methods, including popular methods such as logistic regression, support vector machines and random forests. Powered by this umbrella algorithm, we propose a novel graphical tool for NP classification methods: NP receiver operating characteristic (NP-ROC) bands, motivated by the popular receiver operating characteristic (ROC) curves. NP-ROC bands will help choose in a data adaptive way and compare different NP classifiers. 

Details

	See details in: http://advances.sciencemag.org/content/4/2/eaao1659.full

Usage

	npc(x, y, method = ("logistic", "svm", "nb", "rf"...), alpha = 0.05, delta = 0.05, split = 1, split_ratio = 0.5, n_cores = 1, band = False, randSeed = 0)

Arguments

	x   		n * p observation matrix. n observations, p covariates.
	y   		n 0/1 observatons.
	method  	logistic: Logistic Regression.
	    		svm: Support Vector Machine.
	    		nb: Gaussian Naive Bayes.
	    		nb_m: Multinomial Naive Bayes.
	    		rf: Random Forest.
	    		dt: Decision Tree.
	alpha		the desirable upper bound on type I error rate. Default = 0.05.
	delta		the violation rate of the type I error rate. Default = 0.05.
	split		the number of splits for the class 0 sample. Default = 1. For ensemble version, choose split > 1.
	split_ratio	the ratio of splits used for the class 0 sample to train the classifier. Default = 0.5.
	n_cores		number of cores used for parallel computing. Default = 1.
	band		whether to generate both lower and upper bounds of type II error rate. Default = False.
	randSeed	the random seed used in the algorithm.
  
Example

    import numpy as np
    import os
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from nproc import npc
    
    
    test = npc()
    
    np.random.seed()
    
    # Create a dataset (x,y) with 2 features, binary label and sample size 10000.
    n = 10000
    x = np.random.normal(0, 1, (n,2))
    c = 1+3*x[:,0]
    y = np.random.binomial(1, 1/(1+np.exp(-c)), n)
    
    # Call the npc function to construct Neyman-Pearson classifiers.
    # The default type I error rate upper bound is alpha=0.05.
    fit = test.npc(x, y, 'logistic', n_cores=os.cpu_count())
    
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
