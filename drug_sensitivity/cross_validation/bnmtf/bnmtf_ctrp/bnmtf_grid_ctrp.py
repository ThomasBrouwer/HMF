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

R, M = R_ctrp, M_ctrp

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
Average performance: {'R^2': 0.4217131786890202, 'MSE': 0.091946411978032966, 'Rp': 0.64985570256510061}. 
Performances test: {'R^2': [0.38354300498098504, 0.40891406587373136, 0.45028272401125735, 0.4082101474707025, 0.4386263786930019, 0.3955977839067174, 0.42619955671212517, 0.42719362280462014, 0.4195754885596944, 0.45898901387736646], 'MSE': [0.095961709860586372, 0.094591334969339871, 0.084111535605321794, 0.096753676846175418, 0.088958794627844465, 0.095479902590934557, 0.090931728563523523, 0.092100756604278827, 0.093699917456223997, 0.086874762656100807], 'Rp': [0.62119744949187239, 0.63998720715748036, 0.67153391666810858, 0.64000188619225007, 0.66229939854528375, 0.63011787242490414, 0.6531338046813524, 0.65427000405224633, 0.64848768418302261, 0.67752780225448606]}.
"""