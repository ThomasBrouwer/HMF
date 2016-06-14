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

R, M = R_ccle_ic, M_ccle_ic

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
Average performance: {'R^2': 0.6560108943892284, 'MSE': 0.05921677183856041, 'Rp': 0.81075178669417958}. 
Performances test: {'R^2': [0.6676638961737752, 0.5750185349280311, 0.7419228525740715, 0.5721792073981666, 0.7054527708809106, 0.6411288197502336, 0.6637724398228861, 0.595072843192759, 0.6611636507855919, 0.7367339283858583], 'MSE': [0.061835073911613553, 0.07282188148573511, 0.043373300637294283, 0.066847687992684438, 0.054160326829029951, 0.062788609697469105, 0.054338701832445352, 0.065052967355024, 0.059250305059468246, 0.051698863584840091], 'Rp': [0.81740028049712832, 0.76059065852948105, 0.86208055688210672, 0.75763451333506826, 0.842038957868534, 0.80132793963087146, 0.81653535361572871, 0.77286279237082167, 0.8133492952307082, 0.86369751898134861]}.

Performances 10 folds, 2 restarts, 1000 iterations 900 burnin, 2 thinning.
Average performance: {'R^2': 0.6530476787141359, 'MSE': 0.059781226903239927, 'Rp': 0.80926419424451146}. 
Performances test: {'R^2': [0.6632170234371099, 0.5842080218960926, 0.746389175855457, 0.5773858997354577, 0.6675344830112254, 0.6533814044924358, 0.6656946728631841, 0.5909660488309902, 0.6424534935615509, 0.7392465634578566], 'MSE': [0.062662467328041829, 0.071247234622513547, 0.042622675545692622, 0.066034133927861158, 0.061132610594731462, 0.060644880126797349, 0.05402804422908232, 0.065712738267047849, 0.062522039410889796, 0.051205444979741166], 'Rp': [0.81467938585578303, 0.76614626516560524, 0.86510311767376236, 0.76116551789391174, 0.82213473704093054, 0.80885697073014606, 0.81672464213138141, 0.77068263803295312, 0.80184710461441977, 0.86530156330622165]}.
"""