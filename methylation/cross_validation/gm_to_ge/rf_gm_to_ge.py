'''
Script for using a random forest regressor to predict the promoter region
methylation values, using gene expression as features.
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.methylation.load_methylation import load_ge_pm_top_n_genes, filter_driver_genes, filter_driver_genes_std
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
#R_ge, R_pm, R_gm, genes, samples = filter_driver_genes()
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes_std()

X = R_gm.T
Y = R_ge.T

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
160 driver genes (std)
    10 folds, 1 estimator:
    Average MSE: 1.72045960849 +- 0.0970324082147. 
    Average R^2: -0.733827316493 +- 0.141559609388. 
    Average Rp:  0.153253973435 +- 0.0410164993776.

    10 folds, 10 estimators:
    Average MSE: 0.891336371606 +- 0.089921187122. 
    Average R^2: 0.10695710122 +- 0.0368714349521. 
    Average Rp:  0.353164624601 +- 0.0452165813287.
    
    10 folds, 100 estimators:
    Average MSE: 0.815575844119 +- 0.078010863646. 
    Average R^2: 0.182621792578 +- 0.0294428886982. 
    Average Rp:  0.430486656132 +- 0.0354654071343.

    10 folds, 1000 estimators:
    Average MSE: 0.806086136176 +- 0.0776691051079. 
    Average R^2: 0.192176172965 +- 0.0291540190585. 
    Average Rp:  0.443935327617 +- 0.034858649328.
"""