'''
Script for using multiple matrix factorisation to predict the promoter region 
methylation values, using gene expression as a second dataset.

We append the columns of the two matrices, and mark the unknown rows as 0 in M.
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.methylation.load_methylation import load_ge_pm_top_n_genes, filter_driver_genes, filter_driver_genes_std
from HMF.code.models.hmf_Gibbs import HMF_Gibbs

from sklearn.cross_validation import KFold

import numpy

''' Model settings '''
no_genes = 100 #13966
iterations, burn_in, thinning = 1000, 900, 2

settings = {
    'priorF'  : 'exponential',
    'priorG'  : ['exponential','normal'], #PM,GE
    'orderF'  : 'columns',
    'orderG'  : ['columns','columns'],
    'ARD'     : True
}
hyperparameters = {
    'alphatau' : 1.,
    'betatau'  : 1.,
    'alpha0'   : 0.001,
    'beta0'    : 0.001,
    'lambdaF'  : 0.1,
    'lambdaG'  : 0.1,
}
init = {
    'F'       : 'kmeans',
    'G'       : ['random','least'],
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

E = ['genes','samples']
#I = {'genes':no_genes, 'samples':254}

all_K_alpha = [ # alpha order: PM, GE
    #({'genes':1,  'samples':1},  [1.0, 1.0]),
    #({'genes':5,  'samples':5},  [1.0, 1.0]),
    #({'genes':5,  'samples':5},  [1.5, 0.5]),
    #({'genes':5,  'samples':5},  [1.8, 0.2]),
    #({'genes':5,  'samples':5},  [0.5, 1.5]),
    #({'genes':5,  'samples':5},  [0.2, 1.8]),
    #({'genes':10, 'samples':10}, [1.0, 1.0]),
    ({'genes':10, 'samples':10}, [1.5, 0.5]),
    #({'genes':10, 'samples':10}, [1.8, 0.2]),
    #({'genes':10, 'samples':10}, [0.5, 1.5]),
    #({'genes':10, 'samples':10}, [0.2, 1.8]),
    #({'genes':20, 'samples':20}, [1.0, 1.0]),
    #({'genes':20, 'samples':20}, [1.5, 0.5]),
    #({'genes':20, 'samples':20}, [1.8, 0.2]),
    #({'genes':20, 'samples':20}, [0.5, 1.5]),
    #({'genes':20, 'samples':20}, [0.2, 1.8]),
]

''' Load in data '''
#(R_ge, R_pm, genes, samples) = load_ge_pm_top_n_genes(no_genes)
#R_ge, R_pm, R_gm, genes, samples = filter_driver_genes()
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes_std()

X, Y = R_pm.T, R_ge.T
R, C = [], []

''' Use a method to run the cross-validation under different settings - varying K and alpham '''
def run_all_settings(all_K_alpha):
    fout = open('results_mf_pm_to_ge_std.txt','w')
    for K, alpha in all_K_alpha:
        ''' Compute the folds '''
        n = len(X)
        n_folds = 10
        shuffle, random_state = True, 1
        folds = KFold(n=n,n_folds=n_folds,shuffle=shuffle,random_state=random_state)
        
        ''' Run HMF to predict Y from X '''
        all_MSE, all_R2, all_Rp = numpy.zeros(n_folds), numpy.zeros(n_folds), numpy.zeros(n_folds)
        for i, (train_index, test_index) in enumerate(folds):
            print "Training fold %s for HMF-MF." % (i+1)
            
            ''' Split into train and test '''
            M_X, M_Y_train = numpy.ones(X.shape), numpy.ones(Y.shape)
            M_Y_train[test_index] = 0.
            M_Y_test = 1. - M_Y_train
            
            D = [
                (X, M_X,       'samples', alpha[0]),
                (Y, M_Y_train, 'samples', alpha[1])
            ]
            
            ''' Train and predict '''
            HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
            HMF.initialise(init)
            HMF.run(iterations)
            
            ''' Compute the performances '''
            performances = HMF.predict_Dl(l=1,M_pred=M_Y_test,burn_in=burn_in,thinning=thinning)
            
            all_MSE[i], all_R2[i], all_Rp[i] = performances['MSE'], performances['R^2'], performances['Rp']
            print "MSE: %s. R^2: %s. Rp: %s." % (performances['MSE'], performances['R^2'], performances['Rp'])
        
        print "Average MSE: %s +- %s. \nAverage R^2: %s +- %s. \nAverage Rp:  %s +- %s." % \
            (all_MSE.mean(),all_MSE.std(),all_R2.mean(),all_R2.std(),all_Rp.mean(),all_Rp.std())
            
        fout.write('Tried MF on PM -> GE, with K = %s, alpham = %s.\n' % (K,alpha))
        fout.write('Average MSE: %s +- %s. \nAverage R^2: %s +- %s. \nAverage Rp:  %s +- %s.\n' % \
            (all_MSE.mean(),all_MSE.std(),all_R2.mean(),all_R2.std(),all_Rp.mean(),all_Rp.std()))
        fout.write('All MSE: %s. \nAll R^2: %s. \nAll Rp: %s.\n\n' % (list(all_MSE),list(all_R2),list(all_Rp)))
        fout.flush()

''' Run all the settings '''
run_all_settings(all_K_alpha)