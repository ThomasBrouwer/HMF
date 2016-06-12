'''
Script for using a support vector regressor to predict the promoter region 
methylation values, using gene expression as features.
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.methylation.load_methylation import load_ge_pm_top_n_genes, filter_driver_genes, filter_driver_genes_std
from HMF.code.statistics.statistics import all_statistics_matrix

from sklearn.svm import SVR
from sklearn.cross_validation import KFold

import numpy

''' Model settings '''
no_genes = 100      #13966
kernel = 'rbf' # kernel type

''' Load in data '''
#(R_ge, R_pm, genes, samples) = load_ge_pm_top_n_genes(no_genes)
#R_ge, R_pm, R_gm, genes, samples = filter_driver_genes()
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes_std()

Y = R_pm.T
X = R_ge.T

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
    
    ''' For each individual gene, do a SVR '''
    no_samples, no_genes = Y_test.shape
    Y_pred = numpy.zeros((no_samples,no_genes))
    for gene in range(0,no_genes):
        ''' Train and predict '''
        svr = SVR(kernel=kernel)
        svr.fit(X=X_train,y=Y_train[:,gene])
        Y_pred[:,gene] = svr.predict(X=X_test)
    
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
    10 folds, rbf kernel
    Average MSE: 0.00353847576284 +- 0.000397659490145. 
    Average R^2: 0.921171488375 +- 0.00921918368168. 
    Average Rp:  0.962651486587 +- 0.00456028666333.
    
160 driver genes (std)
    10 folds, rbf kernel
    Average MSE: 0.86247030115 +- 0.108768773829. 
    Average R^2: 0.135994087314 +- 0.0313651214782. 
    Average Rp:  0.377214919289 +- 0.0427727140672.
    
    10 folds, linear kernel
    Average MSE: 2.33763699281 +- 0.351188291974. 
    Average R^2: -1.3446488001 +- 0.234660743598. 
    Average Rp:  0.208960161916 +- 0.0311181872698.
"""