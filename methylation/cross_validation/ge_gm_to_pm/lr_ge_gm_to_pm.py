'''
Script for using a linear regressor to predict values.
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.methylation.load_methylation import load_ge_pm_top_n_genes, filter_driver_genes, filter_driver_genes_std
from HMF.code.statistics.statistics import all_statistics_matrix

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

import numpy

''' Model settings '''
fit_intercept = True # whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations (e.g. data is expected to be already centered).
normalize = True # If True, the regressors X will be normalized before regression.

''' Load in data '''
#(R_ge, R_pm, genes, samples) = load_ge_pm_top_n_genes(no_genes)
#R_ge, R_pm, R_gm, genes, samples = filter_driver_genes()
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes_std()

X1 = R_ge.T
X2 = R_gm.T
X = numpy.concatenate((X1,X2),axis=1)
Y = R_pm.T

''' Compute the folds '''
n = len(X)
n_folds = 10
shuffle, random_state = True, None
folds = KFold(n=n,n_folds=n_folds,shuffle=shuffle,random_state=random_state)

''' Run the RF regression to predict Y from X '''
all_MSE, all_R2, all_Rp = numpy.zeros(n_folds), numpy.zeros(n_folds), numpy.zeros(n_folds)
for i, (train_index, test_index) in enumerate(folds):
    print "Training fold %s for the Linear Regressor." % (i+1)
    
    ''' Split into train and test '''
    X_train, Y_train = X[train_index], Y[train_index]
    X_test,  Y_test  = X[test_index],  Y[test_index]
    
    ''' Train and predict '''
    lr = LinearRegression(fit_intercept=fit_intercept,normalize=normalize)
    lr.fit(X=X_train,y=Y_train)
    Y_pred = lr.predict(X=X_test)
    
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
    10 folds, fit_intercept = True, normalize = True
    Average MSE: 2.23594850138 +- 0.351213645613. 
    Average R^2: -1.23141410102 +- 0.220686844869. 
    Average Rp:  0.320729887021 +- 0.0318639752432.
    
    10 folds, fit_intercept = False, normalize = False
    Average MSE: 2.03631096997 +- 0.281337727573. 
    Average R^2: -1.04865049263 +- 0.137760957457. 
    Average Rp:  0.344365679427 +- 0.0213080994096.
"""