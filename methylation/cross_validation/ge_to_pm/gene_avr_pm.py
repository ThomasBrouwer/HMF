'''
Script for using the gene average to predict the promoter region methylation values.
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.methylation.load_methylation import load_ge_pm_top_n_genes, filter_driver_genes
from HMF.code.statistics.statistics import all_statistics_matrix

from sklearn.cross_validation import KFold

import numpy


''' Load in data '''
no_genes = 100      #13966
#(R_ge, R_pm, genes, samples) = load_ge_pm_top_n_genes(no_genes)
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes()

Y = R_pm.T

''' Compute the folds '''
n = len(Y)
n_folds = 10
shuffle = True
folds = KFold(n=n,n_folds=n_folds,shuffle=shuffle)

''' Do cross-validation to predict Y using the column (gene) average '''
all_MSE, all_R2, all_Rp = numpy.zeros(n_folds), numpy.zeros(n_folds), numpy.zeros(n_folds)
for i, (train_index, test_index) in enumerate(folds):
    print "Training fold %s for the gene average predictor." % (i+1)
    
    ''' Split into train and test '''
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    ''' Compute the column (gene) average '''
    _, no_samples = Y_train.shape
    gene_averages = numpy.average(Y_train,axis=0)
    print gene_averages
    Y_pred = [gene_averages for row in range(0,len(Y_test))]
    
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
    10 folds:
    Average MSE: 0.0384503969803 +- 0.0026974065991. 
    Average R^2: 0.333484597393 +- 0.0376417809001. 
    Average Rp:  0.579599647556 +- 0.0321463951749.

160 driver genes
    10 folds:
    Average MSE: 0.00321981001099 +- 0.000365744306856. 
    Average R^2: 0.928399624833 +- 0.00729832409707. 
    Average Rp:  0.963655118293 +- 0.00372453501049.
"""