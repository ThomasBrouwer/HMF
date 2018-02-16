'''
Script for using matrix factorisation with similarity kernels to predict the 
gene expression values.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from HMF.methylation.load_methylation import filter_driver_genes_std, load_kernels
from HMF.code.models.hmf_Gibbs import HMF_Gibbs

from sklearn.cross_validation import KFold

import numpy

''' Model settings '''
no_genes = 100 #13966
iterations, burn_in, thinning = 200, 180, 2

settings = {
    'priorF'  : 'exponential',
    'priorSn' : ['normal'], 
    'priorSm' : ['normal','normal'],
    'orderF'  : 'columns',
    'orderSn' : ['rows'],
    'orderSm' : ['rows','rows'],
    'ARD'     : True
}
hyperparameters = {
    'alphatau' : 1.,
    'betatau'  : 1.,
    'alpha0'   : 0.001,
    'beta0'    : 0.001,
    'lambdaF'  : 0.1,
    'lambdaSn' : 0.1,
    'lambdaSm' : 0.1,
}
init = {
    'F'       : 'kmeans',
    'Sn'      : ['least'],
    'Sm'      : ['least','least'],
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

E = ['genes','samples']
#I = {'genes':no_genes, 'samples':254}

all_K_alpha = [ # alpha order: PM, GM, GE
    ({'genes':10, 'samples':10}, [0.25, 0.25, 0.25]),
    ({'genes':10, 'samples':10}, [0.5,  0.5,  0.25]),
    ({'genes':10, 'samples':10}, [1.0,  1.0,  0.25]),
    ({'genes':10, 'samples':10}, [1.5,  1.5,  0.25]),
    ({'genes':10, 'samples':10}, [2.0,  2.0,  0.25]),
    ({'genes':10, 'samples':10}, [0.25, 0.25, 0.5]),
    ({'genes':10, 'samples':10}, [0.5,  0.5,  0.5]),
    ({'genes':10, 'samples':10}, [1.0,  1.0,  0.5]),
    ({'genes':10, 'samples':10}, [1.5,  1.5,  0.5]),
    ({'genes':10, 'samples':10}, [2.0,  2.0,  0.5]),
    ({'genes':10, 'samples':10}, [0.25, 0.25, 1.0]),
    ({'genes':10, 'samples':10}, [0.5,  0.5,  1.0]),
    ({'genes':10, 'samples':10}, [1.0,  1.0,  1.0]),
    ({'genes':10, 'samples':10}, [1.5,  1.5,  1.0]),
    ({'genes':10, 'samples':10}, [2.0,  2.0,  1.0]),
    ({'genes':10, 'samples':10}, [0.25, 0.25, 1.5]),
    ({'genes':10, 'samples':10}, [0.5,  0.5,  1.5]),
    ({'genes':10, 'samples':10}, [1.0,  1.0,  1.5]),
    ({'genes':10, 'samples':10}, [1.5,  1.5,  1.5]),
    ({'genes':10, 'samples':10}, [2.0,  2.0,  1.5]),
    ({'genes':10, 'samples':10}, [0.25, 0.25, 2.0]),
    ({'genes':10, 'samples':10}, [0.5,  0.5,  2.0]),
    ({'genes':10, 'samples':10}, [1.0,  1.0,  2.0]),
    ({'genes':10, 'samples':10}, [1.5,  1.5,  2.0]),
    ({'genes':10, 'samples':10}, [2.0,  2.0,  2.0]),
]


''' Load in data '''
#(R_ge, R_pm, genes, samples) = load_ge_pm_top_n_genes(no_genes)
#R_ge, R_pm, R_gm, genes, samples = filter_driver_genes()
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes_std()
K_ge, K_pm, K_gm = load_kernels()

X1, X2, Y = K_pm.T, K_gm.T, R_ge.T
D = []

''' Use a method to run the cross-validation under different settings - varying K and alpham '''
def run_all_settings(all_K_alpha):
    fout = open('results_smtf_pm_gm_to_ge_std_varying_importance.txt','w')
    
    for K, alpha in all_K_alpha:
        ''' Compute the folds '''
        n = len(X1)
        n_folds = 10
        shuffle, random_state = True, None
        folds = KFold(n=n,n_folds=n_folds,shuffle=shuffle,random_state=random_state)
        
        ''' Run HMF to predict Y from X '''
        all_MSE, all_R2, all_Rp = numpy.zeros(n_folds), numpy.zeros(n_folds), numpy.zeros(n_folds)
        for i, (train_index, test_index) in enumerate(folds):
            print "Training fold %s for HMF S-MTF." % (i+1)
            
            ''' Split into train and test '''
            M_X1, M_X2, M_Y_train = numpy.ones(X1.shape), numpy.ones(X2.shape), numpy.ones(Y.shape)
            M_Y_train[test_index] = 0.
            M_Y_test = 1. - M_Y_train
            
            C = [
                (X1, M_X1,      'samples', alpha[0]),
                (X2, M_X2,      'samples', alpha[1]),
            ]
            R = [
                (Y,  M_Y_train, 'samples', 'genes', alpha[2])
            ]
            
            ''' Train and predict '''
            HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
            HMF.initialise(init)
            HMF.run(iterations)
            
            ''' Compute the performances '''
            performances = HMF.predict_Rn(n=0,M_pred=M_Y_test,burn_in=burn_in,thinning=thinning)
            
            all_MSE[i], all_R2[i], all_Rp[i] = performances['MSE'], performances['R^2'], performances['Rp']
            print "MSE: %s. R^2: %s. Rp: %s." % (performances['MSE'], performances['R^2'], performances['Rp'])

        print "Average MSE: %s +- %s. \nAverage R^2: %s +- %s. \nAverage Rp:  %s +- %s." % \
            (all_MSE.mean(),all_MSE.std(),all_R2.mean(),all_R2.std(),all_Rp.mean(),all_Rp.std())
            
        fout.write('Tried S-MTF on PM -> GE, with K = %s, alpham = %s.\n' % (K,alpha))
        fout.write('Average MSE: %s +- %s. \nAverage R^2: %s +- %s. \nAverage Rp:  %s +- %s.\n' % \
            (all_MSE.mean(),all_MSE.std(),all_R2.mean(),all_R2.std(),all_Rp.mean(),all_Rp.std()))
        fout.write('All MSE: %s. \nAll R^2: %s. \nAll Rp: %s.\n\n' % (list(all_MSE),list(all_R2),list(all_Rp)))
        fout.flush()

''' Run all the settings '''
run_all_settings(all_K_alpha)