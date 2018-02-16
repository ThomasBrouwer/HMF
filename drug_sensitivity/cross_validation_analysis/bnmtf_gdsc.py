"""
Run the cross validation for BNMTF on the drug sensitivity datasets, where we 
record the predictions, real values, and indices of the predictions.
This then allows us to plot the performances grouped by how many observations
in that row or column we have.
We use the most common dimensionality from nested cross-validation.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

import HMF.code.generate_mask.mask as mask
from HMF.drug_sensitivity.load_dataset import load_data_without_empty
from HMF.code.models.bnmtf_gibbs import bnmtf_gibbs

import itertools


''' Load datasets '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data =                 location+"data_row_01/"

R_gdsc, M_gdsc, cell_lines, drugs = load_data_without_empty(location_data+"gdsc_ic50_row_01.txt")
I, J = R_gdsc.shape


''' Settings BNMF '''
iterations, burn_in, thinning = 1000, 900, 2
K,L = 7,7
init_FG, init_S = 'kmeans', 'random'
alpha, beta = 1., 1.
lambdaF, lambdaS, lambdaG = 1., 1., 1.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

no_folds = 10
file_performance = 'results/bnmtf.txt'



''' Split the folds. For each, obtain a list for the test set of (i,j,real,pred) values. '''
i_j_real_pred = []
folds_test = mask.compute_folds_attempts(I=I,J=J,no_folds=no_folds,attempts=1000,M=M_gdsc)
folds_training = mask.compute_Ms(folds_test)

for i,(train,test) in enumerate(zip(folds_training,folds_test)):
    print "Fold %s." % (i+1)
    
    ''' Predict values. '''
    BNMTF = bnmtf_gibbs(R=R_gdsc,M=train,K=K,L=L,priors=priors)
    BNMTF.initialise(init_S=init_S, init_FG=init_FG)
    BNMTF.run(iterations=iterations)
    R_pred = BNMTF.return_R_predicted(burn_in=burn_in,thinning=thinning)
    
    ''' Add predictions to list. '''
    indices_test = [(i,j) for (i,j) in itertools.product(range(I),range(J)) if test[i,j]]
    for i,j in indices_test:
        i_j_real_pred.append((i,j,R_gdsc[i,j],R_pred[i,j]))
        
        
''' Store the performances. '''
with open(file_performance, 'w') as fout:
    fout.write('%s' % i_j_real_pred)