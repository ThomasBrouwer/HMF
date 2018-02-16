'''
Test the effects of different K values, as well as ARD, for out-of-matrix
predictions. For alpha we use values 1.0.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from HMF.methylation.load_methylation import load_ge_pm_top_n_genes, filter_driver_genes, filter_driver_genes_std
from HMF.code.models.hmf_Gibbs import HMF_Gibbs

from sklearn.cross_validation import KFold

import numpy

''' Model settings '''
no_genes = 100 #13966
iterations, burn_in, thinning = 100, 80, 2

settings = {
    'priorF'  : 'normal',
    'priorSn' : ['normal','normal','normal'], #GE,PM
    'orderF'  : 'rows',
    'orderSn' : ['rows','rows','rows'],
    'ARD'     : False # True # 
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
    'Sn'      : ['least','least','least'],
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

E = ['genes','samples']
alpha = [1., 1., 1.]

all_K = [
    {'genes':1,  'samples':1},
    {'genes':2,  'samples':2},
    {'genes':4,  'samples':4},
    {'genes':6,  'samples':6},
    {'genes':8,  'samples':8},
    {'genes':10, 'samples':10},
    {'genes':15, 'samples':15},
    {'genes':20, 'samples':20},
    {'genes':25, 'samples':25},
    {'genes':30, 'samples':30},
    {'genes':35, 'samples':35},
    {'genes':40, 'samples':40},
    {'genes':45, 'samples':45},
    {'genes':50, 'samples':50},
]


''' Load in data '''
#(R_ge, R_pm, genes, samples) = load_ge_pm_top_n_genes(no_genes)
#R_ge, R_pm, R_gm, genes, samples = filter_driver_genes()
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes_std()

X1, X2, Y = R_ge.T, R_gm.T, R_pm.T
C, D = [], []

''' Use a method to run the cross-validation under different settings - varying K and alpham '''
def run_all_settings(all_K_alpha):
    fout = open('results_varying_K_hmf_no_ARD_ge_gm_to_pm_std.txt','w')
        
    all_average_performances = []
    for K in all_K:
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

        fout.write('Tried MF on PM -> GE, with K = %s, alphan = %s.\n' % (K,alpha))
        fout.write('Average MSE: %s +- %s. \nAverage R^2: %s +- %s. \nAverage Rp:  %s +- %s.\n' % \
            (all_MSE.mean(),all_MSE.std(),all_R2.mean(),all_R2.std(),all_Rp.mean(),all_Rp.std()))
        fout.write('All MSE: %s. \nAll R^2: %s. \nAll Rp: %s.\n\n' % (list(all_MSE),list(all_R2),list(all_Rp)))
        fout.flush()
        
        all_average_performances.append(all_MSE.mean())
    
    ''' Print for plotting. '''
    print "all_K = %s \nall_average_performances = %s" % (all_K, all_average_performances)

''' Run all the settings '''
run_all_settings(all_K)


'''
ARD:
    all_K = [{'genes': 1, 'samples': 1}, {'genes': 2, 'samples': 2}, {'genes': 4, 'samples': 4}, {'genes': 6, 'samples': 6}, {'genes': 8, 'samples': 8}, {'genes': 10, 'samples': 10}, {'genes': 15, 'samples': 15}, {'genes': 20, 'samples': 20}, {'genes': 25, 'samples': 25}, {'genes': 30, 'samples': 30}, {'genes': 35, 'samples': 35}, {'genes': 40, 'samples': 40}, {'genes': 45, 'samples': 45}, {'genes': 50, 'samples': 50}] 
    all_average_performances = [0.96687919064173988, 0.95642859255925061, 0.94183674422775865, 0.91058768715878879, 0.90272335932951753, 0.84169156169043335, 0.80773291957782178, 0.78792060879368808, 0.77616162161044078, 0.76364626947916814, 0.74255960811908162, 0.73981982592197293, 0.74183804024271838, 0.73888837609339864]

No ARD:
    all_K = [{'genes': 1, 'samples': 1}, {'genes': 2, 'samples': 2}, {'genes': 4, 'samples': 4}, {'genes': 6, 'samples': 6}, {'genes': 8, 'samples': 8}, {'genes': 10, 'samples': 10}, {'genes': 15, 'samples': 15}, {'genes': 20, 'samples': 20}, {'genes': 25, 'samples': 25}, {'genes': 30, 'samples': 30}, {'genes': 35, 'samples': 35}, {'genes': 40, 'samples': 40}, {'genes': 45, 'samples': 45}, {'genes': 50, 'samples': 50}] 
    all_average_performances = [0.96899210395395552, 0.95584451613916599, 0.94668711345253498, 0.91234460070836376, 0.91957780803072209, 0.86065341792206862, 0.83832077958264151, 0.84694769579052465, 0.83790157315686342, 0.8797200549187455, 0.88414963700170401, 0.86713496498571341, 0.8517108828305473, 0.87947057376576621]
'''