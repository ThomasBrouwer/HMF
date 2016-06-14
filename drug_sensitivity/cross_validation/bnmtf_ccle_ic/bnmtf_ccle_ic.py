"""
Run the cross validation with line search for model selection using BNMF on
the drug sensitivity datasets
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
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

R, M = R_ccle_ic, M_ccle_ic

''' Settings BNMTF '''
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
Average performance: {'R^2': 0.6533120143716802, 'MSE': 0.059714538373972871, 'Rp': 0.80913527676203889}. 
Performances test: {'R^2': [0.6349069053209118, 0.5949513502773882, 0.7499802183103329, 0.5721050568219471, 0.72268168219017, 0.6647626944148144, 0.6657293984818569, 0.5825531050496062, 0.612325322458489, 0.733124410391286], 'MSE': [0.067929900586141789, 0.069406332252777134, 0.042019153050384041, 0.066859274139663069, 0.050992334143413517, 0.058653593531165414, 0.054022432122097362, 0.06706430718004755, 0.067790374206961782, 0.052407682527077067], 'Rp': [0.79805778839453545, 0.77285967679232026, 0.86649708914679879, 0.75791489580819338, 0.85359336274548092, 0.81564697851168533, 0.8166666761235859, 0.7645532826688699, 0.78321482305426016, 0.86234819437465848]}.
"""