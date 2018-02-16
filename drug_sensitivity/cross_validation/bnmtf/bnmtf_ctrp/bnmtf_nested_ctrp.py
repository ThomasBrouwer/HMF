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

R, M = R_ctrp, M_ctrp

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
Average performances: {'R^2': 0.4213883095980126, 'MSE': 0.091992316106664571, 'Rp': 0.64984109465811923}. 
All performances: {'R^2': [0.3765189486333157, 0.4135756000092672, 0.44902817876857515, 0.4082796647394621, 0.4361152633562243, 0.39408146480591766, 0.42247281481295607, 0.43296358520799605, 0.42270107084837716, 0.45814650479803454], 'MSE': [0.097055120208308462, 0.093845350821472662, 0.084303491964449417, 0.096742311238387363, 0.089356721757037211, 0.095719441752417034, 0.091522315564223244, 0.091173012214401147, 0.093195343998901406, 0.08701005154704762], 'Rp': [0.61736643161474647, 0.64350145048423779, 0.67054320306783577, 0.63965641772105808, 0.66061076914941708, 0.6299418342508547, 0.65044624229652881, 0.65867474422740313, 0.65074682984843812, 0.67692302392067316]}. 
"""