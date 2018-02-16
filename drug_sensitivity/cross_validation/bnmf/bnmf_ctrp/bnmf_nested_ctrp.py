"""
Run the nested cross validation using BNMF on the drug sensitivity datasets
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../../"
sys.path.append(project_location)

from HMF.code.models.bnmf_gibbs import bnmf_gibbs
from HMF.code.cross_validation.nested_matrix_cross_validation import MatrixNestedCrossValidation
from HMF.drug_sensitivity.load_dataset import load_data_without_empty

import numpy, random

''' Load datasets '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data =                 location+"data_row_01/"
location_features_drugs =       location+"features_drugs/"
location_features_cell_lines =  location+"features_cell_lines/"
location_kernels =              location+"kernels_features/"

R_gdsc,     M_gdsc,     _, _ = load_data_without_empty(location_data+"gdsc_ic50_row_01.txt")
R_ctrp,     M_ctrp,     _, _ = load_data_without_empty(location_data+"ctrp_ec50_row_01.txt")
R_ccle_ec,  M_ccle_ec,  _, _ = load_data_without_empty(location_data+"ccle_ec50_row_01.txt")
R_ccle_ic,  M_ccle_ic,  _, _ = load_data_without_empty(location_data+"ccle_ic50_row_01.txt")

R, M = R_ctrp, M_ctrp


''' Settings BNMF '''
no_folds, no_threads = 10, 5
iterations, burn_in, thinning = 1000, 900, 2
init_UV = 'random'

K_range = range(1,5+1)

quality_metric = 'AIC'
output_file = "results_nested.txt"
files_nested_performances = ["./results_nested_fold_%s.txt" % fold for fold in range(1,no_folds+1)]

alpha, beta = 1., 1.
lambdaU = 1.
lambdaV = 1.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

parameter_search = [ {'K':K, 'priors':priors} for K in K_range ]
train_config = {
    'iterations' : iterations,
    'init' : init_UV
}
predict_config = {
    'burn_in' : burn_in,
    'thinning' : thinning
}


''' Run the nested cross-validation '''
random.seed(0)
numpy.random.seed(0)
nested_crossval = MatrixNestedCrossValidation(
    method=bnmf_gibbs,
    X=R,
    M=M,
    K=no_folds,
    P=no_threads,
    parameter_search=parameter_search,
    train_config=train_config,
    predict_config=predict_config,
    file_performance=output_file,
    files_nested_performances=files_nested_performances
)
nested_crossval.run()

"""
Average performances: {'R^2': 0.4215563829608663, 'MSE': 0.091971319297911774, 'Rp': 0.64983779774157202}. 
All performances: {'R^2': [0.3804384661651281, 0.4051057576259287, 0.4497367473230903, 0.40776422327275885, 0.4375508710126166, 0.39745255383287037, 0.4327599572235762, 0.4248984668926443, 0.41886230323361395, 0.46099448302643564], 'MSE': [0.096444982587646597, 0.095200777590685431, 0.084195074798380343, 0.096826582465562652, 0.089129226338969955, 0.095186896961302087, 0.089892083917804802, 0.092469791594868067, 0.093815049406835285, 0.086552727317062386], 'Rp': [0.61960032857628533, 0.63709339170333834, 0.67111973756468468, 0.6395014157835629, 0.66158583794090886, 0.63143934006769897, 0.65796963100207861, 0.65286039608565549, 0.64808730337548115, 0.67912059531602598]}. 
"""