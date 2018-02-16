"""
Run nested cross-validation on the GDSC dataset with non-probabilistic NMF.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../../"
sys.path.append(project_location)

from HMF.code.models.nmtf_np import nmtf_np
from HMF.code.cross_validation.nested_matrix_cross_validation import MatrixNestedCrossValidation
from HMF.drug_sensitivity.load_dataset import load_data_without_empty

import numpy, random, itertools

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

''' Settings DI-MMTF '''
no_folds, no_threads = 10, 5
output_file = "./results.txt"
files_nested_performances = ["./fold_%s.txt" % fold for fold in range(1,no_folds+1)]

train_config = {
    'iterations' : 1000,
    'init_FG' : 'kmeans',
    'init_S' : 'exponential',
    'expo_prior' : 0.1
}
K_range = range(1,5+1)
L_range = range(1,5+1)
parameter_search = [{'K':K,'L':L} for (K,L) in itertools.product(K_range,L_range)]

R, M = R_ctrp, M_ctrp

''' Run the nested cross-validation '''
random.seed(0)
numpy.random.seed(0)
nested_crossval = MatrixNestedCrossValidation(
    method=nmtf_np,
    X=R,
    M=M,
    K=no_folds,
    P=no_threads,
    parameter_search=parameter_search,
    train_config=train_config,
    file_performance=output_file,
    files_nested_performances=files_nested_performances
)
nested_crossval.run()

''' 
Results

10 folds, 1000 iterations
Average performances: {'R^2': 0.3997974014066007, 'MSE': 0.095424720846599684, 'Rp': 0.63730431035376234}. 
All performances: {'R^2': [0.35400394003126756, 0.38662458297699964, 0.40970450684820037, 0.37954040608995787, 0.4260753189692357, 0.3816457600790576, 0.42793964315155464, 0.40854923594057746, 0.38769619954431933, 0.4361944204348368], 'MSE': [0.1005599530521822, 0.098158315371427607, 0.090320356588020637, 0.10144098752066634, 0.090947714487899192, 0.097683961147539453, 0.090655972297319518, 0.095098562154225588, 0.098846300302619289, 0.090535085544097019], 'Rp': [0.60490224200990306, 0.62693775597706636, 0.64583535128563296, 0.6224496844336348, 0.65509566103486017, 0.62410642541940198, 0.65616900771896292, 0.64523943814134321, 0.62927008872931145, 0.66303744878750726]}. 
'''