"""
Run the cross validation with line search for model selection using BNMF on
the drug sensitivity datasets
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
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

R, M = R_gdsc, M_gdsc

''' Settings BNMF '''
no_folds, no_threads = 10, 5
iterations, burn_in, thinning = 500, 400, 2

init_S = 'random'
init_FG = 'kmeans'

KL_range = [(4,4),(5,5),(6,6),(7,7)]

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
Average performances: {'R^2': 0.5992207178122909, 'MSE': 0.079927469725170369, 'Rp': 0.77466299650215587}. 
All performances: {'R^2': [0.6113250730876916, 0.5941629134381649, 0.614592192363363, 0.6170713874974054, 0.575403606393085, 0.585126700002935, 0.5980018840075433, 0.6045322307429992, 0.593371295915244, 0.5986198946744778], 'MSE': [0.076869062184235296, 0.082015229203092094, 0.076125940573670697, 0.076499983913185701, 0.086224765590910468, 0.082736215690290049, 0.079478589042972225, 0.078091082108673215, 0.08086096370083512, 0.080372865243838756], 'Rp': [0.78207546102853509, 0.77118075222982208, 0.78440505423459517, 0.78571007831196837, 0.75945478386902188, 0.76498130017372101, 0.77467453954668475, 0.77810195302298635, 0.7712427657747154, 0.774803276829509]}. 
"""