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

R, M = R_gdsc, M_gdsc


''' Settings BNMF '''
no_folds, no_threads = 10, 5
iterations, burn_in, thinning = 1000, 900, 2
init_UV = 'random'

K_range = range(1,5+1)

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
Average performances: {'R^2': 0.5964646844068926, 'MSE': 0.080469632602836308, 'Rp': 0.77268356736825905}. 
All performances: {'R^2': [0.6066435271851296, 0.5941194023183445, 0.6077812489605965, 0.6199820865830092, 0.5775396987585044, 0.5881627362481963, 0.5991150207706128, 0.5849868306509036, 0.5836810332210771, 0.6026352593725519], 'MSE': [0.077794941416943264, 0.082024022323738552, 0.077471241479506209, 0.075918495808210765, 0.085790979373547196, 0.08213075337291928, 0.079258512043053425, 0.081950616467962048, 0.082787940257318574, 0.079568823485163545], 'Rp': [0.77894235158091407, 0.7711192765136311, 0.77986690333384712, 0.78764781891445834, 0.76085761710966726, 0.76710208745995712, 0.77453605847564777, 0.7650600507992642, 0.76460981006494388, 0.77709369943025963]}. 
"""