'''
Test the effects of different K values, as well as ARD, for out-of-matrix
predictions. For alpha we use values 1.0.
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
iterations, burn_in, thinning = 100, 80, 2

settings = {
    'priorF'  : 'exponential',
    'priorSn' : ['normal','normal','normal'], #GE,PM
    'orderF'  : 'columns',
    'orderSn' : ['rows','rows','rows'],
    'ARD'     : False # True
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

X1, X2, Y = R_pm.T, R_gm.T, R_ge.T
C, D = [], []


''' Use a method to run the cross-validation under different settings - varying K and alpham '''
def run_all_settings(all_K_alpha):
    fout = open('results_varying_K_hmf_no_ARD_pm_gm_to_ge_std.txt','w')
        
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
    all_average_performances = [1.0008179293234307, 1.0027935398482632, 0.9725686537406224, 0.89406943742551093, 0.90911482664537524, 0.93221899674787956, 0.96545060035367736, 1.0105451043621445, 0.96346641937002442, 1.0115351385666709, 0.97922436610607677, 0.94731013699836597, 0.98923686335869621, 1.0222804345913752]

No ARD:
    all_K = [{'genes': 1, 'samples': 1}, {'genes': 2, 'samples': 2}, {'genes': 4, 'samples': 4}, {'genes': 6, 'samples': 6}, {'genes': 8, 'samples': 8}, {'genes': 10, 'samples': 10}, {'genes': 15, 'samples': 15}, {'genes': 20, 'samples': 20}, {'genes': 25, 'samples': 25}, {'genes': 30, 'samples': 30}, {'genes': 35, 'samples': 35}, {'genes': 40, 'samples': 40}, {'genes': 45, 'samples': 45}, {'genes': 50, 'samples': 50}] 
    all_average_performances = [1.0011176009123834, 1.001208239789396, 0.99740877082053725, 0.91132346942128706, 0.94069263452160179, 0.95233362349919037, 1.1101377682300149, 1.0986209628136276, 1.2216930864968425, 1.2224873115331814, 1.2334890299622294, 1.2569353083033512, 1.3493863257094616, 1.362851123555044]
'''