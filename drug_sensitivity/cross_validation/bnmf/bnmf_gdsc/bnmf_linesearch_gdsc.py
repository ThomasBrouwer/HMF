"""
Run the cross validation with line search for model selection using BNMF on
the drug sensitivity datasets
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.code.models.bnmf_gibbs import bnmf_gibbs
from HMF.code.cross_validation.line_search_cross_validation import LineSearchCrossValidation
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
iterations, burn_in, thinning = 1000, 900, 2
init_UV = 'random'

K_range = range(1,5+1)
no_folds = 10
restarts = 2

quality_metric = 'AIC'
output_file = "results.txt"

alpha, beta = 1., 1.
lambdaU = 1.
lambdaV = 1.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

# Run the cross-validation framework
random.seed(0)
numpy.random.seed(0)
nested_crossval = LineSearchCrossValidation(
    classifier=bnmf_gibbs,
    R=R,
    M=M,
    values_K=K_range,
    folds=no_folds,
    priors=priors,
    init_UV=init_UV,
    iterations=iterations,
    restarts=restarts,
    quality_metric=quality_metric,
    file_performance=output_file
)
nested_crossval.run(burn_in=burn_in,thinning=thinning)

"""
Performances 10 folds, 2 restarts, 500 iterations 400 burnin, 2 thinning.
Average performance: {'R^2': 0.5988664368043916, 'MSE': 0.079991325095272431, 'Rp': 0.7740914673609377}. 
Performances test: {'R^2': [0.6101287878367061, 0.5997026285217406, 0.6133875802178568, 0.6198835459391767, 0.5829367422581575, 0.5870554369648593, 0.5982072709951504, 0.5950056210149767, 0.5861429191617078, 0.5962138351335841], 'MSE': [0.077105654047977604, 0.080895713472904959, 0.076363876159785224, 0.075938181873502972, 0.084694976633890473, 0.082351576820320002, 0.079437982215891711, 0.079972255039367418, 0.082298377007894274, 0.080854657681189573], 'Rp': [0.78120489149083583, 0.7746100235091895, 0.7834720846270592, 0.7876152318721874, 0.76410585552206511, 0.76632681455625229, 0.77361257688925678, 0.77143326092355813, 0.76587691690164861, 0.77265701731732428]}.

Performances 10 folds, 2 restarts, 1000 iterations 900 burnin, 2 thinning.
Average performance: {'R^2': 0.5984248042453831, 'MSE': 0.08007964483382865, 'Rp': 0.7738184253234317}. 
Performances test: {'R^2': [0.6078767359753459, 0.5980624405058887, 0.6128082335054388, 0.6178758404458005, 0.5831309288949382, 0.5836131388336798, 0.6003944429274495, 0.596157689230025, 0.5850802182057089, 0.599248373929556], 'MSE': [0.077551047106769092, 0.081227177502463915, 0.076478309008646078, 0.076339273442351613, 0.084655542249874483, 0.083038058019890018, 0.079005558947577714, 0.079744761775516454, 0.082509702530575435, 0.080247017754621577], 'Rp': [0.77976532110941776, 0.77359590997873373, 0.78311601445982404, 0.78629008107551646, 0.76425751517448459, 0.76412127501516636, 0.77505014386119431, 0.77218176187109611, 0.76516137811821716, 0.77464485257066618]}.
"""