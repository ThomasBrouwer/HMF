'''
Script for using a random forest regressor to predict the promoter region
methylation values, using gene expression as features.
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

X = R_ge.T
Y = R_pm.T

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
    Average MSE: 0.00534711085867 +- 0.000795111298234. 
    Average R^2: 0.880812613237 +- 0.0188552606185. 
    Average Rp:  0.940948181068 +- 0.0089296863681.
    
    10 folds, 10 estimators:
    Average MSE: 0.028719183557 +- 0.00305707071341. 
    Average R^2: 0.500430009301 +- 0.0667711584878. 
    Average Rp:  0.709585840895 +- 0.0462748536733.
    
    10 folds, 100 estimators:
    Average MSE: 0.0258352561042 +- 0.00413138224843. 
    Average R^2: 0.550010514462 +- 0.077807455449. 
    Average Rp:  0.743234096254 +- 0.0523117587712.
    
    10 folds, 1000 estimators:
    Average MSE: 0.025430596927 +- 0.00384329435704. 
    Average R^2: 0.556922385563 +- 0.0787808116692. 
    Average Rp:  0.747544577824 +- 0.0546832029041.
    
160 driver genes
    10 folds, 1 estimator:
    Average MSE: 0.00534711085867 +- 0.000795111298234. 
    Average R^2: 0.880812613237 +- 0.0188552606185. 
    Average Rp:  0.940948181068 +- 0.0089296863681.

    10 folds, 10 estimators:
    Average MSE: 0.00280977816245 +- 0.00038980574028. 
    Average R^2: 0.937420224185 +- 0.0087994674882. 
    Average Rp:  0.968420617516 +- 0.00428629892791.
    
    10 folds, 100 estimators:
    Average MSE: 0.00249555278225 +- 0.000468516637625. 
    Average R^2: 0.944459958245 +- 0.0103735633935. 
    Average Rp:  0.971891608224 +- 0.00535444759152.

    10 folds, 1000 estimators:
    
"""