'''
Script for using multiple matrix tri-actorisation to predict the gene expression 
values, using promoter region methylation as a second dataset.
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.methylation.load_methylation import filter_driver_genes_std
from HMF.code.models.hmf_Gibbs import HMF_Gibbs

from sklearn.cross_validation import KFold

import numpy

''' Model settings '''
no_genes = 100 #13966
iterations, burn_in, thinning = 200, 180, 2

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
    'Sn'      : ['least','least','least'],
    'lambdat' : 'exp',
    'tau'     : 'exp'
}
alpha = [0.5, 0.5, 1.5]
K = {'genes':10, 'samples':10}

values_settings = [
    # Semi-nonnegative
    {
        'priorF'  : 'exponential',
        'priorG'  : 'normal',
        'priorSn' : 'normal',
        'priorSm' : 'normal',
        'orderF'  : 'columns',
        'orderG'  : 'rows',
        'orderSn' : 'rows',
        'orderSm' : 'rows',
        'ARD'     : True
    },
    # Semi-nonnegative, with column draws
    {
        'priorF'  : 'exponential',
        'priorG'  : 'normal',
        'priorSn' : 'normal',
        'priorSm' : 'normal',
        'orderF'  : 'columns',
        'orderG'  : 'columns',
        'orderSn' : 'individual',
        'orderSm' : 'individual',
        'ARD'     : True
    },
    # Real-valued
    {
        'priorF'  : 'normal',
        'priorG'  : 'normal',
        'priorSn' : 'normal',
        'priorSm' : 'normal',
        'orderF'  : 'rows',
        'orderG'  : 'rows',
        'orderSn' : 'rows',
        'orderSm' : 'rows',
        'ARD'     : True
    },
    # Real-valued, with column draws
    {
        'priorF'  : 'normal',
        'priorG'  : 'normal',
        'priorSn' : 'normal',
        'priorSm' : 'normal',
        'orderF'  : 'columns',
        'orderG'  : 'columns',
        'orderSn' : 'individual',
        'orderSm' : 'individual',
        'ARD'     : True
    },
]


''' Load in data '''
#(R_ge, R_pm, genes, samples) = load_ge_pm_top_n_genes(no_genes)
#R_ge, R_pm, R_gm, genes, samples = filter_driver_genes()
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes_std()

X1, X2, Y = R_ge.T, R_pm.T, R_gm.T
C, D = [], []


''' Use method to run the cross-validation under different settings - varying negativity constraints '''
fout = open('results_hmf_ge_pm_to_gm_varying_negativity.txt','w')
for settings in values_settings:
    ''' Compute the folds '''
    n = len(X1)
    n_folds = 10
    shuffle, random_state = True, None
    folds = KFold(n=n,n_folds=n_folds,shuffle=shuffle,random_state=random_state)
    
    ''' Run HMF to predict Y from X '''
    all_MSE, all_R2, all_Rp = numpy.zeros(n_folds), numpy.zeros(n_folds), numpy.zeros(n_folds)
    for i, (train_index, test_index) in enumerate(folds):
        print "Training fold %s for HMF-MTF." % (i+1)
        
        ''' Split into train and test '''
        M_X1, M_X2, M_Y_train = numpy.ones(X1.shape), numpy.ones(X2.shape), numpy.ones(Y.shape)
        M_Y_train[test_index] = 0.
        M_Y_test = 1. - M_Y_train
        
        R = [
            (X1, M_X1,      'samples', 'genes', alpha[0]),
            (X2, M_X2,      'samples', 'genes', alpha[1]),
            (Y,  M_Y_train, 'samples', 'genes', alpha[2])
        ]
        
        ''' Train and predict '''
        HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
        HMF.initialise(init)
        HMF.run(iterations)
        
        ''' Compute the performances '''
        performances = HMF.predict_Rn(n=2,M_pred=M_Y_test,burn_in=burn_in,thinning=thinning)
        
        all_MSE[i], all_R2[i], all_Rp[i] = performances['MSE'], performances['R^2'], performances['Rp']
        print "MSE: %s. R^2: %s. Rp: %s." % (performances['MSE'], performances['R^2'], performances['Rp'])
    
    print "Average MSE: %s +- %s. \nAverage R^2: %s +- %s. \nAverage Rp:  %s +- %s." % \
        (all_MSE.mean(),all_MSE.std(),all_R2.mean(),all_R2.std(),all_Rp.mean(),all_Rp.std())

    fout.write('Tried MF on GE, GM -> PM, with settings = %s.\n' % (settings))
    fout.write('Average MSE: %s +- %s. \nAverage R^2: %s +- %s. \nAverage Rp:  %s +- %s.\n' % \
        (all_MSE.mean(),all_MSE.std(),all_R2.mean(),all_R2.std(),all_Rp.mean(),all_Rp.std()))
    fout.write('All MSE: %s. \nAll R^2: %s. \nAll Rp: %s.\n\n' % (list(all_MSE),list(all_R2),list(all_Rp)))
    fout.flush()