'''
Script for using a random forest regressor to predict the gene expression 
values, using promoter region methylation as features.
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

Y = R_ge.T
X = R_pm.T

''' Compute the folds '''
n = len(X)
n_folds = 10
shuffle = True
folds = KFold(n=n,n_folds=n_folds,shuffle=shuffle)

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
    10 folds, 1 estimator:
    Average MSE: 4.59619664101 +- 0.391542612006. 
    Average R^2: 0.245268284771 +- 0.0618955608979. 
    Average Rp:  0.624323630501 +- 0.0311250012654.

    10 folds, 10 estimators:
    Average MSE: 2.44648211214 +- 0.177148708446. 
    Average R^2: 0.598882261217 +- 0.0197071411403. 
    Average Rp:  0.775234934427 +- 0.0122348032184.

    10 folds, 100 estimators:
    Average MSE: 2.21053647099 +- 0.179784131312. 
    Average R^2: 0.638012016481 +- 0.0235131342244. 
    Average Rp:  0.799459653049 +- 0.0145718211889.

    10 folds, 1000 estimators:
    Average MSE: 2.17811147119 +- 0.185525207994. 
    Average R^2: 0.642477537706 +- 0.0317788162907. 
    Average Rp:  0.802426441531 +- 0.020128752759.
    
160 driver genes
    10 folds, 1 estimator:
    Average MSE: 0.795970953741 +- 0.0768369693165. 
    Average R^2: 0.3404355513 +- 0.0625050414068. 
    Average Rp:  0.672859373517 +- 0.0286886342839.

    10 folds, 10 estimators:
    Average MSE: 0.410015632458 +- 0.0339477463786. 
    Average R^2: 0.660683660383 +- 0.0252118225275. 
    Average Rp:  0.813861487885 +- 0.0147419218431.

    10 folds, 100 estimators:
    Average MSE: 0.374325092472 +- 0.0278174076743. 
    Average R^2: 0.690146579209 +- 0.0238359601365. 
    Average Rp:  0.830914237051 +- 0.0144322054166.

    10 folds, 1000 estimators:
    Average MSE: 0.371637074998 +- 0.0406966332475. 
    Average R^2: 0.692282939685 +- 0.0316354139571. 
    Average Rp:  0.832550657386 +- 0.019193002811.
"""