"""
Run the cross validation for Support Vector Regression on the drug sensitivity datasets.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from HMF.drug_sensitivity.load_dataset import load_data_without_empty, load_data_filter
import HMF.code.generate_mask.mask as mask
from HMF.code.statistics.statistics import all_statistics_list

from sklearn.svm import SVR

import numpy, random, itertools


''' Model settings '''
kernel = 'rbf' # kernel type


''' Load datasets '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data =                 location+"data_row_01/"
location_features_drugs =       location+"features_drugs/"
location_features_cell_lines =  location+"features_cell_lines/"
location_kernels =              location+"kernels_features/"

R_main, M_main, cell_lines, drugs = load_data_without_empty(location_data+"ccle_ic50_row_01.txt")

R_cnv,      M_cnv =      load_data_filter(location_features_cell_lines+"cnv.txt",                 cell_lines)
#R_cnv_std,  M_cnv_std =  load_data_filter(location_features_cell_lines+"cnv_std.txt",             cell_lines)
R_mutation, M_mutation = load_data_filter(location_features_cell_lines+"mutation.txt",            cell_lines)
#R_ge,       M_ge =       load_data_filter(location_features_cell_lines+"gene_expression.txt",     cell_lines)
#R_ge_std,   M_ge_std =   load_data_filter(location_features_cell_lines+"gene_expression_std.txt", cell_lines)

R_fp,       M_fp =       load_data_filter(location_features_drugs+"drug_fingerprints.txt", drugs)
R_targets,  M_targets =  load_data_filter(location_features_drugs+"drug_targets.txt",      drugs)
R_1d2d,     M_1d2d =     load_data_filter(location_features_drugs+"drug_1d2d.txt",         drugs)
#R_1d2d_std, M_1d2d_std = load_data_filter(location_features_drugs+"drug_1d2d_std.txt",     drugs)

features_drugs = [R_fp, R_targets, R_1d2d]
features_cell_lines = [R_cnv, R_mutation]


''' Split the mask M into folds '''
no_folds = 10
I,J = R_main.shape
ATTEMPTS_GENERATE_M = 100

numpy.random.seed(0)
random.seed(0)
folds_test = mask.compute_folds_attempts(I=I,J=J,no_folds=no_folds,attempts=ATTEMPTS_GENERATE_M,M=M_main)
folds_training = mask.compute_Ms(folds_test)


''' Function for assembling features X '''
def assemble_X(Rs_rows,Rs_cols,M):
    indices = [(i,j) for i,j in itertools.product(range(0,I),range(0,J)) if M[i,j]]
    X = [[] for datapoint in range(0,len(indices))]
    for n,(i,j) in enumerate(indices):
        for R in Rs_rows:
            X[n] += list(R[i,:])
        for R in Rs_cols:
            X[n] += list(R[j,:])
    return numpy.array(X)

''' Function for assembling outcomes y '''
def assemble_y(R,M):
    I,J = R.shape
    y = [R[i,j] for i,j in itertools.product(range(0,I),range(0,J)) if M[i,j]]
    return numpy.array(y)


''' For each train, test fold, construct the labels y and features X '''
all_MSE, all_R2, all_Rp = numpy.zeros(no_folds), numpy.zeros(no_folds), numpy.zeros(no_folds)
for i, (M_train, M_test) in enumerate(zip(folds_training, folds_test)):
    print "Training fold %s for the Support Vector Regressor." % (i+1)
    
    ''' Assemble training and test matrices '''
    y_train, y_test = assemble_y(R_main,M_train), assemble_y(R_main,M_test)
    X_train = assemble_X(Rs_rows=features_cell_lines,Rs_cols=features_drugs,M=M_train)
    X_test = assemble_X(Rs_rows=features_cell_lines,Rs_cols=features_drugs,M=M_test)
    
    ''' Train the SVR and predict '''
    svr = SVR(kernel=kernel)
    svr.fit(X=X_train,y=y_train)
    y_pred = svr.predict(X=X_test)
    
    ''' Measure and store performance '''
    MSE, R2, Rp = all_statistics_list(R=y_test, R_pred=y_pred)
    all_MSE[i], all_R2[i], all_Rp[i] = MSE, R2, Rp
    print "MSE: %s. R^2: %s. Rp: %s." % (MSE,R2,Rp)
    
print "Average MSE: %s +- %s. \nAverage R^2: %s +- %s. \nAverage Rp:  %s +- %s." % \
    (all_MSE.mean(),all_MSE.std(),all_R2.mean(),all_R2.std(),all_Rp.mean(),all_Rp.std())


"""
10 folds, rbf kernel
Average MSE: 0.0915556188769 +- 0.00699479638215. 
Average R^2: 0.469785431883 +- 0.0518605257957. 
Average Rp:  0.702120900233 +- 0.0432055838993.
"""