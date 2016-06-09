'''
Script for using HMF to predict the promoter region methylation values, using 
gene expression as a second dataset.

We append the columns of the two matrices, and mark the unknown rows as 0 in M.
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.methylation.load_methylation import load_ge_pm_top_n_genes, filter_driver_genes
from HMF.code.models.hmf_Gibbs import HMF_Gibbs

from sklearn.cross_validation import KFold

import numpy

''' Model settings '''
no_genes = 100 #13966
iterations, burn_in, thinning = 100, 80, 2

settings = {
    'priorF'  : 'exponential',
    'priorSn' : ['normal','exponential'], #GE,ME
    'orderF'  : 'columns',
    'orderSn' : ['rows','individual'],
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
    'Sn'      : ['least','random'],
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

E = ['genes','samples']
#I = {'genes':no_genes, 'samples':254}
K = {'genes':10, 'samples':10}
alpha_n = [1., 1.] # GE, PM


''' Load in data '''
#(R_ge, R_pm, genes, samples) = load_ge_pm_top_n_genes(no_genes)
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes()

X, Y = R_ge.T, R_pm.T
C, D = [], []

''' Compute the folds '''
n = len(R_ge)
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
160 driver genes, F ~ Exp, S_ge ~ N, S_me ~ Exp (kmeans, least/random)

    K = {'genes':1, 'samples':1}
        alpha_n = [1., 1.]
            Average MSE: 0.017509413843 +- 0.00474386578188. 
            Average R^2: 0.610447618989 +- 0.102762710141. 
            Average Rp:  0.821033745449 +- 0.047097829043.

    K = {'genes':5, 'samples':5}
        alpha_n = [1., 1.]
            Average MSE: 0.00437767141612 +- 0.000894667081691. 
            Average R^2: 0.902476189408 +- 0.019750922672. 
            Average Rp:  0.950983976206 +- 0.0104815329002.
        
    K = {'genes':10, 'samples':10}
        alpha_n = [1., 1.]
            Average MSE: 0.00427123925447 +- 0.00122210799119. 
            Average R^2: 0.904975232112 +- 0.0263593122673. 
            Average Rp:  0.952835315512 +- 0.0126607987076.

160 driver genes, F ~ Exp, S ~ N (kmeans, least)

    K = {'genes':1, 'samples':1}
        alpha_n = [1., 1.]
            Average MSE: 0.0149405897035 +- 0.00308398760147. 
            Average R^2: 0.6667677241 +- 0.0694125181607. 
            Average Rp:  0.845228199569 +- 0.0324158523075.

        alpha_n = [.9, .1]
            Average MSE: 0.0656341622939 +- 0.00171696001673. 
            Average R^2: -0.462783762411 +- 0.0120617222758. 
            Average Rp:  0.187213235549 +- 0.00771361505496.

        alpha_n = [.1, .9]
            Average MSE: 0.00915173355995 +- 0.00169133136236. 
            Average R^2: 0.795891845187 +- 0.0379651615426. 
            Average Rp:  0.901861507044 +- 0.0137047011738.
    
        alpha_n = [.01, .99]
            Average MSE: 0.0110235306303 +- 0.00546498020856. 
            Average R^2: 0.755366806415 +- 0.116517550767. 
            Average Rp:  0.894185206293 +- 0.0256377237106.
    
    K = {'genes':5, 'samples':5}
        alpha_n = [1., 1.]
            Average MSE: 0.00428166461493 +- 0.000768395577562. 
            Average R^2: 0.904677337726 +- 0.0159646931067. 
            Average Rp:  0.95247478509 +- 0.00834541107817.
        
        alpha_n = [.9, .1]
            Average MSE: 0.00423015773172 +- 0.00058639227605. 
            Average R^2: 0.90595123264 +- 0.0106390461575. 
            Average Rp:  0.953807184655 +- 0.00602184541991.
        
        alpha_n = [.1, .9]
            Average MSE: 0.00518338758917 +- 0.000863827879565. 
            Average R^2: 0.884450623678 +- 0.0190712561609. 
            Average Rp:  0.942220181337 +- 0.00915853781432.
        
    K = {'genes':10, 'samples':10}
        alpha_n = [1., 1.]
            Average MSE: 0.00439354343213 +- 0.00101391271495. 
            Average R^2: 0.902550292426 +- 0.0197568819492. 
            Average Rp:  0.952108337596 +- 0.0105324758504.
    
        alpha_n = [.9, .1]
            Average MSE: 0.00412125686078 +- 0.000590578816185. 
            Average R^2: 0.908220554609 +- 0.0123826019994. 
            Average Rp:  0.954082273181 +- 0.00652850217941.
            
        alpha_n = [.1, .9]
            Average MSE: 0.00414024222346 +- 0.00132728722418. 
            Average R^2: 0.908309856252 +- 0.0271272241042. 
            Average Rp:  0.95469975701 +- 0.0130031570097.

        alpha_n = [.99, .01]
            Average MSE: 0.0402457592316 +- 0.00310555730664. 
            Average R^2: 0.103304165696 +- 0.061906017555. 
            Average Rp:  0.350994980153 +- 0.0700154493853.

        alpha_n = [.01, .99]
            Average MSE: 0.00496648102011 +- 0.000722434029565. 
            Average R^2: 0.889422981353 +- 0.0153008528164. 
            Average Rp:  0.945494166098 +- 0.00750350117297.
        
    K = {'genes':20, 'samples':20}
        alpha_n = [1., 1.]
            Average MSE: 0.0044598741759 +- 0.000913800346634. 
            Average R^2: 0.900547898049 +- 0.0208816560466. 
            Average Rp:  0.949972805762 +- 0.0106506318386.
        
160 driver genes, F, S ~ N (kmeans, least)

    K = {'genes':1, 'samples':1}
        alpha_n = [1., 1.]
            Average MSE: 0.0173221463233 +- 0.00263170868037. 
            Average R^2: 0.614101426396 +- 0.0560110815874. 
            Average Rp:  0.821861319092 +- 0.0288131801361.
            
        alpha_n = [.9, .1]
            Average MSE: 0.0778849517187 +- 0.00221018080142. 
            Average R^2: -0.73603489696 +- 0.0163240539674. 
            Average Rp:  0.184475194156 +- 0.00569522428729.

        alpha_n = [.1, .9]
            Average MSE: 0.0452501968297 +- 0.00512244782668. 
            Average R^2: -0.00809190800206 +- 0.10741100805. 
            Average Rp:  0.617592049041 +- 0.0672342318636.
            
    K = {'genes':5, 'samples':5}
        alpha_n = [1., 1.]
            Average MSE: 0.00408600479349 +- 0.000640348489736. 
            Average R^2: 0.908986020211 +- 0.013976335171. 
            Average Rp:  0.954707428468 +- 0.00757168211272.
            
        alpha_n = [.9, .1]
            Average MSE: 0.0044063835313 +- 0.000733744428324. 
            Average R^2: 0.901912241184 +- 0.0151746239543. 
            Average Rp:  0.950893181887 +- 0.00770766583976.
"""