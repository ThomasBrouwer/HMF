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

R, M = R_ccle_ec, M_ccle_ec

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
Average performance: {'R^2': 0.1220124576141151, 'MSE': 0.13413001781251538, 'Rp': 0.41487910931046335}. 
Performances test: {'R^2': [0.22499442587403895, 0.1258239589814072, -0.013981851399431422, 0.04303749876148699, 0.08196143848790627, 0.08840069046828569, 0.20763789561879953, 0.1977293227680733, 0.06208333316532311, 0.20243786341526115], 'MSE': [0.10230243594045504, 0.1385604798394236, 0.1634276623019349, 0.15577573333486669, 0.12476017372600584, 0.14043741914149332, 0.12583971172096867, 0.13074413755430345, 0.1513822569832288, 0.10807016758247326], 'Rp': [0.49475252490278226, 0.40972422228134681, 0.31775571935219737, 0.35624301360546334, 0.40446488659680652, 0.35735321066943376, 0.48285302028469895, 0.4736854228350168, 0.37036079151990825, 0.48159828105697922]}.

Performances 10 folds, 2 restarts, 1000 iterations 900 burnin, 2 thinning.
Average performance: {'R^2': 0.12152229589487322, 'MSE': 0.13422958319512507, 'Rp': 0.41445401263596227}. 
Performances test: {'R^2': [0.2264877632342973, 0.13243553661621255, -0.0187217930878667, 0.04167690553973313, 0.08637924126394714, 0.08662382169234895, 0.19523800163421268, 0.1907796008629773, 0.06948313368204717, 0.2048407475108226], 'MSE': [0.10210531213291649, 0.13751251772814591, 0.16419161837128601, 0.15599721265784416, 0.12415980042478168, 0.14071115658560557, 0.12780901221598615, 0.13187671714684082, 0.15018790940091337, 0.10774457528693053], 'Rp': [0.49588520192371993, 0.41807381871630778, 0.31761049322182144, 0.35582258071350437, 0.40110844275439245, 0.35233181500154709, 0.47423809346964385, 0.46780329385979663, 0.37847405726819366, 0.48319232943069601]}.
"""