'''
Script for using a random forest regressor to predict the promoter region
methylation values, using gene body methylation as features.
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.methylation.load_methylation import load_ge_pm_top_n_genes, filter_driver_genes
from HMF.code.statistics.statistics import all_statistics_matrix

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold

import numpy

''' Model settings '''
no_genes = 100      #13966
n_estimators = 1000 # number of trees
max_depth = None    # until what depth of feature splits we go

''' Load in data '''
#(R_ge, R_pm, genes, samples) = load_ge_pm_top_n_genes(no_genes)
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes()

X = R_gm.T
Y = R_pm.T

''' Compute the folds '''
n = len(X)
n_folds = 10
shuffle, random_state = True, 0
folds = KFold(n=n,n_folds=n_folds,shuffle=shuffle,random_state=random_state)

''' Run the RF regression to predict Y from X '''
all_MSE, all_R2, all_Rp = numpy.zeros(n_folds), numpy.zeros(n_folds), numpy.zeros(n_folds)
for i, (train_index, test_index) in enumerate(folds):
    print "Training fold %s for the Random Forest regressor." % (i+1)
    
    ''' Split into train and test '''
    X_train, Y_train = X[train_index], Y[train_index]
    X_test,  Y_test  = X[test_index],  Y[test_index]
    
    ''' Train and predict '''
    rf = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth)
    rf.fit(X=X_train,y=Y_train)
    Y_pred = rf.predict(X=X_test)
    
    ''' Measure performance '''
    rows, cols = Y_test.shape
    no_datapoints = rows * cols
    MSE, R2, Rp = all_statistics_matrix(R=Y_test, R_pred=Y_pred, M=numpy.ones((rows,cols)))
    
    all_MSE[i], all_R2[i], all_Rp[i] = MSE, R2, Rp
    print "MSE: %s. R^2: %s. Rp: %s." % (MSE,R2,Rp)

print "Average MSE: %s +- %s. \nAverage R^2: %s +- %s. \nAverage Rp:  %s +- %s." % \
    (all_MSE.mean(),all_MSE.std(),all_R2.mean(),all_R2.std(),all_Rp.mean(),all_Rp.std())


"""
100 top genes

    
160 driver genes
    10 folds, 1 estimator:
    Average MSE: 0.00449925315503 +- 0.00058144348208. 
    Average R^2: 0.899838032285 +- 0.0127687957201. 
    Average Rp:  0.949988956134 +- 0.00636403719185.

    10 folds, 10 estimators:
    Average MSE: 0.00234521901746 +- 0.000425910193757. 
    Average R^2: 0.947805042336 +- 0.00933415150274. 
    Average Rp:  0.973632130605 +- 0.00474059433098.
    
    10 folds, 100 estimators:
    Average MSE: 0.0021360065684 +- 0.00044292484733. 
    Average R^2: 0.952439531057 +- 0.00982668610762. 
    Average Rp:  0.975990078977 +- 0.00501557743499.

    10 folds, 1000 estimators:
    Average MSE: 0.00210739093339 +- 0.000442502614183. 
    Average R^2: 0.953082173031 +- 0.00979139932692. 
    Average Rp:  0.976314652877 +- 0.00499973374292.
"""