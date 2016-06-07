'''
Script for using a random forest regressor to predict the gene expression 
values, using promoter region methylation as features.
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.methylation.load_methylation import load_ge_pm_top_n_genes
from HMF.code.statistics.statistics import all_statistics_matrix

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold

import numpy

''' Model settings '''
no_genes = 100      #13966
n_estimators = 1000 # number of trees
max_depth = None    # until what depth of feature splits we go

''' Load in data '''
(R_ge_n, R_pm_n, genes, samples) = load_ge_pm_top_n_genes(no_genes)

Y = R_ge_n.T
X = R_pm_n.T

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
100 top genes, 10 folds, 1 estimator:
Average MSE: 4.59619664101 +- 0.391542612006. 
Average R^2: 0.245268284771 +- 0.0618955608979. 
Average Rp:  0.624323630501 +- 0.0311250012654.

100 top genes, 10 folds, 10 estimators:
Average MSE: 2.44648211214 +- 0.177148708446. 
Average R^2: 0.598882261217 +- 0.0197071411403. 
Average Rp:  0.775234934427 +- 0.0122348032184.

100 top genes, 10 folds, 100 estimators:
Average MSE: 2.21053647099 +- 0.179784131312. 
Average R^2: 0.638012016481 +- 0.0235131342244. 
Average Rp:  0.799459653049 +- 0.0145718211889.

100 top genes, 10 folds, 1000 estimators:
Average MSE: 2.17811147119 +- 0.185525207994. 
Average R^2: 0.642477537706 +- 0.0317788162907. 
Average Rp:  0.802426441531 +- 0.020128752759.
"""