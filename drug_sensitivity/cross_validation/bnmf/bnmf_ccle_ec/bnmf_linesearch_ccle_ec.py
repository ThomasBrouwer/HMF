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

R, M = R_ccle_ec, M_ccle_ec

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
Average performance: {'R^2': 0.1254066072297933, 'MSE': 0.13359605864431459, 'Rp': 0.41744680688473323}. 
Performances test: {'R^2': [0.2237495071595682, 0.1457641180096102, -0.006278673099161747, 0.048746182474687205, 0.07522100196637982, 0.10359166531775199, 0.20677624764592994, 0.19422079093238176, 0.05286436886284196, 0.20941086302794343], 'MSE': [0.10246676794178529, 0.13539988303354128, 0.16218610909250555, 0.15484646558336163, 0.12567618974828454, 0.1380971570550748, 0.1259765551312117, 0.13131591461403849, 0.15287022246300572, 0.10712532178033678], 'Rp': [0.49801452199832563, 0.42595854718897297, 0.32290316617590237, 0.36120502477268407, 0.399421533015411, 0.36712636907475749, 0.48310703458384002, 0.47245217747247303, 0.36292163973749403, 0.48135805482747185]}.

Performances 10 folds, 2 restarts, 1000 iterations 900 burnin, 2 thinning.
Average performance: {'R^2': 0.12440809035229616, 'MSE': 0.13374740582925398, 'Rp': 0.4163175593352057}. 
Performances test: {'R^2': [0.23872884522554805, 0.14820930240889663, -0.010863657480792543, 0.052602338063412635, 0.06472801571550457, 0.09437969276995495, 0.19731639441595228, 0.19759182868918468, 0.05870446118294592, 0.20268368253235458], 'MSE': [0.10048946245639757, 0.13501231129997349, 0.16292509004975508, 0.15421875502635482, 0.12710217210071162, 0.13951631746501356, 0.12747893036697935, 0.13076654463619397, 0.15192761595256657, 0.10803685893859377], 'Rp': [0.50789959351711844, 0.42733616656047696, 0.32180912189068933, 0.36474567050551299, 0.38853358124273174, 0.36170047929464633, 0.47450149650925227, 0.47399124413156901, 0.36824212044954618, 0.47441611925051352]}.
"""