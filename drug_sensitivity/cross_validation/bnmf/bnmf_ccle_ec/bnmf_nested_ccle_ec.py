"""
Run the nested cross validation using BNMF on the drug sensitivity datasets
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
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

R, M = R_ccle_ec, M_ccle_ec


''' Settings BNMF '''
no_folds, no_threads = 10, 5
iterations, burn_in, thinning = 1000, 900, 2
init_UV = 'random'

K_range = range(1,3+1)

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
Average performances: {'R^2': 0.13776394069419484, 'MSE': 0.13175510842095928, 'Rp': 0.42703094535537478}. 
All performances: {'R^2': [0.2784441415666905, 0.13299188273516316, 0.048808862237248785, 0.096478641218114, 0.08466140046116721, 0.10119755703484823, 0.18345520766061396, 0.18037681346750034, 0.0682520881544334, 0.20297281240616893], 'MSE': [0.095246956214583664, 0.13742433459159051, 0.1533074224477893, 0.14707650724642535, 0.12439325261946449, 0.13846598400007218, 0.12968030740881112, 0.13357203458119501, 0.15038660343953006, 0.10799768166013117], 'Rp': [0.54036463462243511, 0.4164009638437387, 0.36021737397661358, 0.39379602513169526, 0.42074800757911573, 0.36468555770326866, 0.46478112261617732, 0.46092448543593245, 0.37281851324059823, 0.47557276940417242]}. 
"""