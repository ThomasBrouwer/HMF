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

R, M = R_ccle_ic, M_ccle_ic

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
Average performances: {'R^2': 0.6557733622757193, 'MSE': 0.059273555972201034, 'Rp': 0.81071543341446728}. 
All performances: {'R^2': [0.6549382033217122, 0.5839934735773817, 0.7416150434979959, 0.5630154631503392, 0.7126021288773382, 0.6502137245757811, 0.674633655881302, 0.60068610351328, 0.6389649294633799, 0.7370708968986818], 'MSE': [0.064202840004506281, 0.071283998136977583, 0.04342503205066689, 0.068279537792686182, 0.052845727581695992, 0.061199101889032249, 0.052583389505802255, 0.064151177405282644, 0.063132063947853923, 0.051632691407506204], 'Rp': [0.80962638008424215, 0.76607915580209762, 0.8618903319583392, 0.75192492337502104, 0.84726648115954828, 0.80700881458981144, 0.82228941718134518, 0.77595967883800598, 0.79977566782483578, 0.86533348333142579]}. 
"""