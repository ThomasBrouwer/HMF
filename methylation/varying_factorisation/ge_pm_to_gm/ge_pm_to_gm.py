'''
Script for trying different combinations of factorisations to predict the GM
values, given GE and PM values.

We try: (GE,PM,GM)
- MTF, MTF, MTF
- MTF, MTF, MF
- MTF, MF,  MTF
- MF,  MTF, MTF
- MF,  MF,  MF
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/" # "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.methylation.load_methylation import filter_driver_genes_std
from HMF.code.models.hmf_Gibbs import HMF_Gibbs

from sklearn.cross_validation import KFold

import numpy

''' Model settings '''
iterations, burn_in, thinning = 200, 180, 2

hyperparameters = {
    'alphatau' : 1.,
    'betatau'  : 1.,
    'alpha0'   : 0.001,
    'beta0'    : 0.001,
    'lambdaF'  : 0.1,
    'lambdaS'  : 0.1,
    'lambdaG'  : 0.1,
}
init = {
    'F'       : 'kmeans',
    'G'       : ['least','least','least'],
    'Sn'      : ['least','least','least'],
    'lambdat' : 'exp',
    'tau'     : 'exp'
}
alpha = [0.5, 0.5, 0.5]
K = {'genes':10, 'samples':10}

settings = {
    'priorF'  : 'exponential',
    'priorG'  : 'normal',
    'priorSn' : 'normal',
    'priorSm' : 'normal',
    'orderF'  : 'columns',
    'orderG'  : 'rows',
    'orderSn' : 'rows',
    'orderSm' : 'rows',
    'ARD'     : True
}


''' Load in data '''
#(R_ge, R_pm, genes, samples) = load_ge_pm_top_n_genes(no_genes)
#R_ge, R_pm, R_gm, genes, samples = filter_driver_genes()
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes_std()

X1, X2, Y = R_ge.T, R_pm.T, R_gm.T

# The different R, C, D values
values_factorisation = [
    { 'X1': 'R', 'X2': 'R', 'Y': 'R' },
    { 'X1': 'R', 'X2': 'R', 'Y': 'D' },
    { 'X1': 'R', 'X2': 'D', 'Y': 'R' },
    { 'X1': 'D', 'X2': 'R', 'Y': 'R' },
    { 'X1': 'D', 'X2': 'D', 'Y': 'D' },
]
def construct_RCD(factorisation, M_Y_train):
    ''' Given a dictionary {'X1':..,'X2':..,'Y':..}, all giving either 'R' or 
        'D', construct the (R,C,D) lists. For the Y entry, use M_Y as the mask. 
        Note that the Y entry always has index 0. '''
    R, C, D = [], [], []
    M_X1, M_X2 = numpy.ones(X1.shape), numpy.ones(X2.shape)
    if factorisation['Y'] == 'R':
        R.append((Y, M_Y_train, 'samples', 'genes', alpha[2]))
    else: # factorisation['Y'] == 'D'
        D.append((Y, M_Y_train, 'samples', alpha[2])) 
    if factorisation['X1'] == 'R':
        R.append((X1, M_X1, 'samples', 'genes', alpha[0]))
    else: # factorisation['X2'] == 'D'
        D.append((X1, M_X1, 'samples', alpha[0]))
    if factorisation['X2'] == 'R':
        R.append((X2, M_X2, 'samples', 'genes', alpha[1]))
    else: # factorisation['X2'] == 'D'
        D.append((X2, M_X2, 'samples', alpha[1]))   
    return R, C, D


''' Use method to run the cross-validation under different settings - varying factorisation types. '''
fout = open('results_ge_pm_to_gm_varying_factorisation.txt','w')
for factorisation in values_factorisation:
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
        M_Y_train = numpy.ones(Y.shape)
        M_Y_train[test_index] = 0.
        M_Y_test = 1. - M_Y_train
        
        R, C, D = construct_RCD(factorisation, M_Y_train)
        
        ''' Train and predict '''
        HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
        HMF.initialise(init)
        HMF.run(iterations)
        
        ''' Compute the performances - be careful to use the right method (R or D) '''
        if factorisation['Y'] == 'R':
            performances = HMF.predict_Rn(n=0,M_pred=M_Y_test,burn_in=burn_in,thinning=thinning)
        else: # factorisation['Y'] == 'D'
            performances = HMF.predict_Dl(l=0,M_pred=M_Y_test,burn_in=burn_in,thinning=thinning)
        
        all_MSE[i], all_R2[i], all_Rp[i] = performances['MSE'], performances['R^2'], performances['Rp']
        print "MSE: %s. R^2: %s. Rp: %s." % (performances['MSE'], performances['R^2'], performances['Rp'])
    
    print "Average MSE: %s +- %s. \nAverage R^2: %s +- %s. \nAverage Rp:  %s +- %s." % \
        (all_MSE.mean(),all_MSE.std(),all_R2.mean(),all_R2.std(),all_Rp.mean(),all_Rp.std())

    fout.write('Tried HMF on GE, PM -> GM, with factorisation = %s.\n' % (factorisation))
    fout.write('Average MSE: %s +- %s. \nAverage R^2: %s +- %s. \nAverage Rp:  %s +- %s.\n' % \
        (all_MSE.mean(),all_MSE.std(),all_R2.mean(),all_R2.std(),all_Rp.mean(),all_Rp.std()))
    fout.write('All MSE: %s. \nAll R^2: %s. \nAll Rp: %s.\n\n' % (list(all_MSE),list(all_R2),list(all_Rp)))
    fout.flush()