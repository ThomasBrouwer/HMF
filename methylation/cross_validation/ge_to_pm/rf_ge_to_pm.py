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

X = R_ge.T
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
    Average MSE: 0.00501976727869 +- 0.000779891992678. 
    Average R^2: 0.888191192909 +- 0.0176515453659. 
    Average Rp:  0.944368177057 +- 0.00873581220305.

    10 folds, 10 estimators:
    Average MSE: 0.00275716103362 +- 0.000502043984722. 
    Average R^2: 0.938585077828 +- 0.0113488416599. 
    Average Rp:  0.968991540379 +- 0.00573993395794.
    
    10 folds, 100 estimators:
    Average MSE: 0.00252754907006 +- 0.000481777494941. 
    Average R^2: 0.943694878894 +- 0.0109253096835. 
    Average Rp:  0.971638919881 +- 0.00548327888793.

    10 folds, 1000 estimators:
    Average MSE: 0.002489958894 +- 0.000460998115048. 
    Average R^2: 0.944537858178 +- 0.010415236763. 
    Average Rp:  0.972051225882 +- 0.00524679384447.
    
160 driver genes (std)
    10 folds, 1 estimator:
    Average MSE: 1.72890916031 +- 0.150943782571. 
    Average R^2: -0.744157811653 +- 0.130453191916. 
    Average Rp:  0.123811188986 +- 0.0363299796438.

    10 folds, 10 estimators:
    Average MSE: 0.972983046646 +- 0.120269451496. 
    Average R^2: 0.0245506862124 +- 0.0399945676361. 
    Average Rp:  0.252602861549 +- 0.0535116459343.
    
    10 folds, 100 estimators:
    Average MSE: 0.885485288481 +- 0.113348649299. 
    Average R^2: 0.112713025658 +- 0.0349633295554. 
    Average Rp:  0.340042345276 +- 0.0485527027101.

    10 folds, 1000 estimators:
    
"""