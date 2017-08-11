"""
Run the cross validation for HMF (datasets, using MTF) on the drug sensitivity
datasets, where we record the predictions, real values, and indices of the 
predictions.
This then allows us to plot the performances grouped by how many observations
in that row or column we have.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

import HMF.code.generate_mask.mask as mask
from HMF.drug_sensitivity.load_dataset import load_data_without_empty, load_data_filter
from HMF.code.models.hmf_Gibbs import HMF_Gibbs

import itertools


''' Load datasets '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data =                 location+"data_row_01/"
location_features_drugs =       location+"features_drugs/"
location_features_cell_lines =  location+"features_cell_lines/"
location_kernels =              location+"kernels_features/"

R_gdsc,     M_gdsc,   cell_lines, drugs   = load_data_without_empty(location_data+"gdsc_ic50_row_01.txt")
R_ccle_ec,  M_ccle_ec                     = load_data_filter(location_data+"ccle_ec50_row_01.txt",cell_lines,drugs)
R_ctrp,     M_ctrp                        = load_data_filter(location_data+"ctrp_ec50_row_01.txt",cell_lines,drugs)
R_ccle_ic,  M_ccle_ic                     = load_data_filter(location_data+"ccle_ic50_row_01.txt",cell_lines,drugs)

I, J = R_gdsc.shape


''' Settings HMF '''
iterations, burn_in, thinning = 250, 200, 1
no_folds = 10

settings = {
    'priorF'  : 'exponential',
    'priorG'  : 'normal',
    'priorSn' : 'normal',
    'priorSm' : 'normal',
    'orderF'  : 'columns',
    'orderG'  : 'rows',
    'orderSn' : 'rows',
    'orderSm' : 'rows',
    'ARD'     : True
}
hyperparameters = {
    'alphatau' : 1.,
    'betatau'  : 1.,
    'alpha0'   : 0.001,
    'beta0'    : 0.001,
    'lambdaF'  : 0.1,
    'lambdaG'  : 0.1,
    'lambdaSn' : 0.1,
    'lambdaSm' : 0.1,
}
init = {
    'F'       : 'kmeans',
    'Sn'      : 'least',
    'Sm'      : 'least',
    'G'       : 'least',
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

alpha = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC
K = {'Cell_lines':10, 'Drugs':10}
file_performance = 'results/hmf_d_mtf.txt'
D, C = [], []
n = 0



''' Split the folds. For each, obtain a list for the test set of (i,j,real,pred) values. '''
i_j_real_pred = []
folds_test = mask.compute_folds_attempts(I=I,J=J,no_folds=no_folds,attempts=1000,M=M_gdsc)
folds_training = mask.compute_Ms(folds_test)

for i,(train,test) in enumerate(zip(folds_training,folds_test)):
    print "Fold %s." % (i+1)
    
    ''' Predict values. '''
    R = [(R_gdsc,    train,     'Cell_lines', 'Drugs', alpha[0]), 
         (R_ctrp,    M_ctrp,    'Cell_lines', 'Drugs', alpha[1]), 
         (R_ccle_ic, M_ccle_ic, 'Cell_lines', 'Drugs', alpha[2]),
         (R_ccle_ec, M_ccle_ec, 'Cell_lines', 'Drugs', alpha[3])]
    HMF = HMF_Gibbs(R=R,C=C,D=D,K=K,settings=settings,hyperparameters=hyperparameters)
    HMF.initialise(init=init)
    HMF.run(iterations=iterations)
    R_pred = HMF.return_Rn(n=n,burn_in=burn_in,thinning=thinning)
    
    ''' Add predictions to list. '''
    indices_test = [(i,j) for (i,j) in itertools.product(range(I),range(J)) if test[i,j]]
    for i,j in indices_test:
        i_j_real_pred.append((i,j,R_gdsc[i,j],R_pred[i,j]))
        
        
''' Store the performances. '''
with open(file_performance, 'w') as fout:
    fout.write('%s' % i_j_real_pred)