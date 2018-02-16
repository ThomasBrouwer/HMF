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

R, M = R_ccle_ic, M_ccle_ic


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
Average performances: {'R^2': 0.6541079139599442, 'MSE': 0.05944666085775914, 'Rp': 0.80951147802337964}. 
All performances: {'R^2': [0.6503598040243246, 0.6030155664678333, 0.7027014750101929, 0.5371169359706814, 0.7433026717648732, 0.6621160928462642, 0.6663679987744852, 0.5907872148997958, 0.6470990108866711, 0.7382123689543201], 'MSE': [0.065054705497576124, 0.068024503999169664, 0.049964975326256003, 0.072326224382758195, 0.047200617825985905, 0.059116646688002233, 0.053919226094391107, 0.065741468565059205, 0.061709705317128646, 0.051408534881264407], 'Rp': [0.80691042865637896, 0.77926873705584843, 0.83956165063279442, 0.73537795664478744, 0.86365035664825451, 0.81395397735794894, 0.8171909897663362, 0.76955983501596659, 0.8049507581901223, 0.86469009026535804]}. 
"""