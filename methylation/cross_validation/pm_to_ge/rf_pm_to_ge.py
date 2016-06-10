'''
Script for using a random forest regressor to predict the gene expression 
values, using promoter region methylation as features.
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
    Average MSE: 0.790687460111 +- 0.07667765625. 
    Average R^2: 0.345455918463 +- 0.0565688133244. 
    Average Rp:  0.674278776242 +- 0.0287225908945.

    10 folds, 10 estimators:
    Average MSE: 0.404558945196 +- 0.046803743027. 
    Average R^2: 0.665325773613 +- 0.0328109679366. 
    Average Rp:  0.816350303954 +- 0.0200690699472.

    10 folds, 100 estimators:
    Average MSE: 0.370693119479 +- 0.0435100001681. 
    Average R^2: 0.693430706697 +- 0.0296732435931. 
    Average Rp:  0.833251950968 +- 0.0182825379061.

    10 folds, 1000 estimators:
    Average MSE: 0.367032074255 +- 0.0447927399464. 
    Average R^2: 0.696489183614 +- 0.0307290118114. 
    Average Rp:  0.835104496433 +- 0.0188865032742.
    
160 driver genes (std)
    10 folds, 1 estimator:
    Average MSE: 1.64112377437 +- 0.125490166349. 
    Average R^2: -0.655344003329 +- 0.175857747209. 
    Average Rp:  0.183674797044 +- 0.0525761740703.

    10 folds, 10 estimators:
    Average MSE: 0.902961209096 +- 0.0806886994898. 
    Average R^2: 0.0941673375573 +- 0.0393400229475. 
    Average Rp:  0.340095645263 +- 0.0444182381533.

    10 folds, 100 estimators:
    Average MSE: 0.825743938467 +- 0.0748849644407. 
    Average R^2: 0.171790265734 +- 0.0350726899469. 
    Average Rp:  0.416561174842 +- 0.0418755075308.

    10 folds, 1000 estimators:
    
"""