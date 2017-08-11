"""
Run the cross validation for NMF-NP on the drug sensitivity datasets, where we 
record the predictions, real values, and indices of the predictions.
This then allows us to plot the performances grouped by how many observations
in that row or column we have.
We use the most common dimensionality from nested cross-validation.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

import HMF.code.generate_mask.mask as mask
from HMF.drug_sensitivity.load_dataset import load_data_without_empty
from HMF.code.models.nmf_np import nmf_np

import itertools


''' Load datasets '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data =                 location+"data_row_01/"

R_gdsc, M_gdsc, cell_lines, drugs = load_data_without_empty(location_data+"gdsc_ic50_row_01.txt")
I, J = R_gdsc.shape


''' Settings NMF-NP. '''
iterations = 2000
K = 2
init_UV = 'exponential'
expo_prior = 0.1

no_folds = 10
file_performance = 'results/nmf_np.txt'



''' Split the folds. For each, obtain a list for the test set of (i,j,real,pred) values. '''
i_j_real_pred = []
folds_test = mask.compute_folds_attempts(I=I,J=J,no_folds=no_folds,attempts=1000,M=M_gdsc)
folds_training = mask.compute_Ms(folds_test)

for i,(train,test) in enumerate(zip(folds_training,folds_test)):
    print "Fold %s." % (i+1)
    
    ''' Predict values. '''
    NMF_NP = nmf_np(R=R_gdsc,M=train,K=K)
    NMF_NP.train(iterations=iterations,init_UV=init_UV,expo_prior=expo_prior)
    R_pred = NMF_NP.return_R_predicted()
    
    ''' Add predictions to list. '''
    indices_test = [(i,j) for (i,j) in itertools.product(range(I),range(J)) if test[i,j]]
    for i,j in indices_test:
        i_j_real_pred.append((i,j,R_gdsc[i,j],R_pred[i,j]))
        
        
''' Store the performances. '''
with open(file_performance, 'w') as fout:
    fout.write('%s' % i_j_real_pred)