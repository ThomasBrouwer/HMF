'''
Script for using HMF to predict the promoter region methylation values, using 
gene expression as a second dataset.

We append the columns of the two matrices, and mark the unknown rows as 0 in M.
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.methylation.load_methylation import load_ge_pm_top_n_genes
from HMF.code.models.hmf_Gibbs import HMF_Gibbs

from sklearn.cross_validation import KFold

import numpy

''' Model settings '''
no_genes = 100 #13966
iterations, burn_in, thinning = 100, 80, 2

settings = {
    'priorF'  : 'exponential',
    'priorSn' : 'normal',
    'orderF'  : 'columns',
    'orderSn' : 'rows',
    'ARD'     : True
}
hyperparameters = {
    'alphatau' : 1.,
    'betatau'  : 1.,
    'alpha0'   : 0.001,
    'beta0'    : 0.001,
    'lambdaF'  : 0.1,
    'lambdaSn' : 0.1,
}
init = {
    'F'       : 'kmeans',
    'Sn'      : 'random',
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

E = ['genes','samples']
I = {'genes':no_genes, 'samples':254}
K = {'genes':10, 'samples':10}
alpha_n = [.1, 10.] # GE, PM


''' Load in data '''
(R_ge_n, R_pm_n, genes, samples) = load_ge_pm_top_n_genes(no_genes)
X, Y = R_ge_n.T, R_pm_n.T
C, D = [], []

''' Compute the folds '''
n = len(R_ge_n)
n_folds = 10
shuffle = True
folds = KFold(n=n,n_folds=n_folds,shuffle=shuffle)

''' Run HMF to predict Y from X '''
all_MSE, all_R2, all_Rp = numpy.zeros(n_folds), numpy.zeros(n_folds), numpy.zeros(n_folds)
for i, (train_index, test_index) in enumerate(folds):
    print "Training fold %s for NMF." % (i+1)
    
    ''' Split into train and test '''
    M_X, M_Y_train = numpy.ones(X.shape), numpy.ones(Y.shape)
    M_Y_train[test_index] = 0.
    M_Y_test = 1. - M_Y_train
    
    R = [
        (X, M_X,       'samples', 'genes', alpha_n[0]),
        (Y, M_Y_train, 'samples', 'genes', alpha_n[1])
    ]
    
    ''' Train and predict '''
    HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
    HMF.initialise(init)
    HMF.run(iterations)
    
    ''' Compute the performances '''
    performances = HMF.predict_Rn(n=1,M_pred=M_Y_test,burn_in=burn_in,thinning=thinning)
    
    all_MSE[i], all_R2[i], all_Rp[i] = performances['MSE'], performances['R^2'], performances['Rp']
    print "MSE: %s. R^2: %s. Rp: %s." % (performances['MSE'], performances['R^2'], performances['Rp'])

print "Average MSE: %s +- %s. \nAverage R^2: %s +- %s. \nAverage Rp:  %s +- %s." % \
    (all_MSE.mean(),all_MSE.std(),all_R2.mean(),all_R2.std(),all_Rp.mean(),all_Rp.std())


"""
F ~ Exp, S ~ N (kmeans, random)

    10 folds, K = {'genes':1, 'samples':1}, alpha_n = [1., 1.]
    Average MSE: 0.117651366722 +- 0.0298548177772. 
    Average R^2: -1.26606510328 +- 0.420622968113. 
    Average Rp:  0.241168685116 +- 0.090845204215.
    
    10 folds, K = {'genes':1, 'samples':1}, alpha_n = [2., .5]
    Average MSE: 0.161406056211 +- 0.0479807544257. 
    Average R^2: -2.14793405047 +- 0.957895830721. 
    Average Rp:  0.126945212981 +- 0.0599642190343.
    
    10 folds, K = {'genes':1, 'samples':1}, alpha_n = [10., .1]
    Average MSE: 0.208091626515 +- 0.0335425004304. 
    Average R^2: -3.07100888448 +- 0.670907869579. 
    Average Rp:  0.0376879160985 +- 0.0291385188905.
    
    10 folds, K = {'genes':1, 'samples':1}, alpha_n = [.5, 2.]
    Average MSE: 0.105595791535 +- 0.0325040924271. 
    Average R^2: -1.03872154112 +- 0.543172163437. 
    Average Rp:  0.279137459256 +- 0.0744145354584.
    
    10 folds, K = {'genes':1, 'samples':1}, alpha_n = [.1, 10.]
    Average MSE: 0.067457682849 +- 0.0116381144111. 
    Average R^2: -0.315306702401 +- 0.252593697919. 
    Average Rp:  0.363955822806 +- 0.0895892010406.
    

    10 folds, K = {'genes':5, 'samples':5}, alpha_n = [1., 1.]
    Average MSE: 0.111865974174 +- 0.102632189013. 
    Average R^2: -1.20736389562 +- 2.06809623465. 
    Average Rp:  0.390839452864 +- 0.149491707925.

    10 folds, K = {'genes':5, 'samples':5}, alpha_n = [2., 0.5]
    Average MSE: 0.243555752455 +- 0.318902035875. 
    Average R^2: -3.94065499045 +- 6.89957698886. 
    Average Rp:  0.35425588746 +- 0.121414980162.

    10 folds, K = {'genes':5, 'samples':5}, alpha_n = [.5, 2.]
    Average MSE: 0.576838850508 +- 1.08480878219. 
    Average R^2: -9.85213183449 +- 20.6901951536. 
    Average Rp:  0.336696472942 +- 0.210922239443.
    
    10 folds, K = {'genes':5, 'samples':5}, alpha_n = [.1, 10.]
    Average MSE: 0.469655478443 +- 0.556869365353. 
    Average R^2: -7.4685431054 +- 9.16749419379. 
    Average Rp:  0.194421227735 +- 0.139253463727.


    10 folds, K = {'genes':10, 'samples':10}, alpha_n = [1., 1.]
    Average MSE: 0.59383818538 +- 0.510565168668. 
    Average R^2: -11.2622246756 +- 12.0345050519. 
    Average Rp:  0.180734688965 +- 0.11893042806.
    
    10 folds, K = {'genes':10, 'samples':10}, alpha_n = [.1, 10.]
    Average MSE: 0.856961971739 +- 1.13255427032. 
    Average R^2: -15.1533853109 +- 20.4762795124. 
    Average Rp:  0.187463255485 +- 0.179042061954.

---

F ~ Exp, S ~ N (kmeans, least)

---

F, S ~ N (kmeans, least)



"""