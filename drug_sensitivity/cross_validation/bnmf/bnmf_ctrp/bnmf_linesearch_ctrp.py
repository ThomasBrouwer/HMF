"""
Run the cross validation with line search for model selection using BNMF on
the drug sensitivity datasets
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../../"
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

R, M = R_ctrp, M_ctrp

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
Average performance: {'R^2': 0.4222098290995741, 'MSE': 0.09186113010245639, 'Rp': 0.65059709677813615}. 
Performances test: {'R^2': [0.37781134698067875, 0.40893168329116725, 0.4438087309255375, 0.40564847706189533, 0.44287989152919727, 0.3939108356601021, 0.43361075836582175, 0.42568592198001676, 0.42675096674935487, 0.4630596784519694], 'MSE': [0.096853937066197909, 0.094588515658411018, 0.085102112987047174, 0.097172492798934362, 0.088284756232593492, 0.09574639673998632, 0.089757255129446073, 0.092343177764739298, 0.09254155542838384, 0.086221101218824439], 'Rp': [0.61794065136797771, 0.64020785561371607, 0.66770684606382003, 0.63790829561521467, 0.66550611807522586, 0.62990618912924168, 0.65870682757178822, 0.65362616532109263, 0.65383884328049735, 0.68062317574278708]}.

Performances 10 folds, 2 restarts, 1000 iterations 900 burnin, 2 thinning. 
Average performance: {'R^2': 0.42205931880500913, 'MSE': 0.091885982369427299, 'Rp': 0.65046182204080583}. 
Performances test: {'R^2': [0.3782739165418936, 0.40523651772209857, 0.4451444649838837, 0.4049197106736159, 0.4393865801585597, 0.3963160792632351, 0.4321313154567926, 0.42811837968692534, 0.4278076857824824, 0.46325853778060344], 'MSE': [0.096781930476310382, 0.095179852085029965, 0.084897734031327166, 0.097291641221852715, 0.088838328322546942, 0.09536643052077054, 0.089991706501177715, 0.091952066205697125, 0.092370965654473766, 0.086189168675086553], 'Rp': [0.61824392437346853, 0.63738062618068625, 0.6689163591334315, 0.6374101120857244, 0.66288832374635787, 0.63157752580605853, 0.65753585379468893, 0.65536388388448685, 0.65454433100078968, 0.68075728040236427]}.
"""