'''
Script for using a linear regressor to predict the gene expression values, 
using promoter region methylation as features.
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
fit_intercept = False # whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations (e.g. data is expected to be already centered).
normalize = False # If True, the regressors X will be normalized before regression.

''' Load in data '''
#(R_ge, R_pm, genes, samples) = load_ge_pm_top_n_genes(no_genes)
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes()
#R_ge, R_pm, R_gm, genes, samples = filter_driver_genes_std()

Y = R_ge.T
X = R_pm.T

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
160 driver genes
    10 folds, fit_intercept = True, normalize = True
        Average MSE: 1.29633809393 +- 0.367329650754. 
        Average R^2: -0.0663462051036 +- 0.264683700056. 
        Average Rp:  0.613013392625 +- 0.0548107799898.
    
    10 folds, fit_intercept = False, normalize = False
        Average MSE: 1.2653466962 +- 0.348804187599. 
        Average R^2: -0.0409205848253 +- 0.24874410622. 
        Average Rp:  0.61643844203 +- 0.0523957195147.
    
160 driver genes (std)
    10 folds, fit_intercept = True, normalize = True
        Average MSE: 2.80711591728 +- 0.708927344999. 
        Average R^2: -1.79161393993 +- 0.540460954279. 
        Average Rp:  0.234206852656 +- 0.0346279113613.
    
    10 folds, fit_intercept = False, normalize = False
        Average MSE: 2.66679616085 +- 0.593243351099. 
        Average R^2: -1.65637807451 +- 0.442977384386. 
        Average Rp:  0.243964715654 +- 0.032620792825.
"""