"""
Run the cross validation with line search for model selection using BNMF on
the drug sensitivity datasets
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../../"
sys.path.append(project_location)

from HMF.code.models.bnmtf_gibbs import bnmtf_gibbs
from HMF.code.cross_validation.greedy_search_cross_validation_bnmtf import GreedySearchCrossValidation
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
iterations, burn_in, thinning = 500, 400, 2

init_S = 'random'
init_FG = 'kmeans'

K_range = range(1,5+1)
L_range = range(1,5+1)
no_folds = 10
restarts = 2

quality_metric = 'AIC'
output_file = "results.txt"

alpha, beta = 1., 1.
lambdaF = 1.
lambdaS = 1.
lambdaG = 1.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

# Run the cross-validation framework
random.seed(0)
numpy.random.seed(0)
nested_crossval = GreedySearchCrossValidation(
    classifier=bnmtf_gibbs,
    R=R,
    M=M,
    values_K=K_range,
    values_L=L_range,
    folds=no_folds,
    priors=priors,
    init_S=init_S,
    init_FG=init_FG,
    iterations=iterations,
    restarts=restarts,
    quality_metric=quality_metric,
    file_performance=output_file
)
nested_crossval.run(burn_in=burn_in,thinning=thinning)

"""
Performances 10 folds, 2 restarts, 500 iterations 400 burnin, 2 thinning.
Average performance: {'R^2': 0.5921287898608034, 'MSE': 0.081329129719171681, 'Rp': 0.76973013223220677}. 
Performances test: {'R^2': [0.6042727683903288, 0.5964153363261631, 0.6094530541897409, 0.6181852063532448, 0.5848340058265824, 0.5781364620523171, 0.5811457352839599, 0.5833259971848013, 0.5822864890084756, 0.5832328439924217], 'MSE': [0.078263811396977953, 0.081560039212974431, 0.077141025684696482, 0.0762774695285931, 0.084309690491768757, 0.084130245710570595, 0.082811198983049769, 0.082278573112352843, 0.083065255134063931, 0.083453987936668889], 'Rp': [0.77756713100416441, 0.77269544973072002, 0.78098941235001407, 0.78656214030147487, 0.76554515505090925, 0.76038148499493163, 0.76240940734532858, 0.76376662970978182, 0.76352174733881872, 0.76386276449592461]}.
"""