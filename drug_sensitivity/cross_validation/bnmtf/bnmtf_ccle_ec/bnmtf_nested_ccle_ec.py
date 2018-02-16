"""
Run the cross validation with line search for model selection using BNMF on
the drug sensitivity datasets
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../../"
sys.path.append(project_location)

from HMF.code.models.bnmtf_gibbs import bnmtf_gibbs
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
iterations, burn_in, thinning = 500, 400, 2

init_S = 'random'
init_FG = 'kmeans'

KL_range = [(1,1),(2,2),(3,3),(4,4),(5,5)]

output_file = "results_nested.txt"
files_nested_performances = ["./results_nested_fold_%s.txt" % fold for fold in range(1,no_folds+1)]

alpha, beta = 1., 1.
lambdaF = 1.
lambdaS = 1.
lambdaG = 1.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

parameter_search = [ {'K':K, 'L':L, 'priors':priors} for K,L in KL_range ]
train_config = {
    'iterations' : iterations,
    'init_S' : init_S,
    'init_FG' : init_FG
}
predict_config = {
    'burn_in' : burn_in,
    'thinning' : thinning
}


''' Run the nested cross-validation '''
random.seed(0)
numpy.random.seed(0)
nested_crossval = MatrixNestedCrossValidation(
    method=bnmtf_gibbs,
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
Average performances: {'R^2': 0.15318151485364193, 'MSE': 0.12922058450231275, 'Rp': 0.4395563145297392}. 
All performances: {'R^2': [0.220409745193052, 0.13937687942353383, 0.08376061762028364, 0.14065397899274823, 0.11862831457022516, 0.08638591641023052, 0.2544626384534714, 0.20631083468281697, 0.06421300226051385, 0.2176132209295436], 'MSE': [0.10290762384791373, 0.13641228648754625, 0.14767410300749062, 0.13988558218065419, 0.11977719586232516, 0.14074780734155884, 0.11840319739608772, 0.12934562903835428, 0.15103852269887555, 0.10601389716232122], 'Rp': [0.51441793565088567, 0.41903218477244536, 0.37292092723770803, 0.42995918834537278, 0.44040861984606655, 0.35656488404536307, 0.51858070819700375, 0.48202012222758434, 0.37297963897877517, 0.48867893599618734]}. 
"""