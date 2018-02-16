"""
Run nested cross-validation on the GDSC dataset with non-probabilistic NMF.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../../"
sys.path.append(project_location)

from HMF.code.models.nmf_np import nmf_np
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

''' Settings DI-MMTF '''
no_folds, no_threads = 10, 5
output_file = "./results.txt"
files_nested_performances = ["./fold_%s.txt" % fold for fold in range(1,no_folds+1)]

train_config = {
    'iterations' : 1000,
    'init_UV' : 'exponential',
    'expo_prior' : 0.1
}
K_range = range(1,5+1)
parameter_search = [{'K':K} for K in K_range]

R, M = R_ctrp, M_ctrp

''' Run the nested cross-validation '''
random.seed(0)
numpy.random.seed(0)
nested_crossval = MatrixNestedCrossValidation(
    method=nmf_np,
    X=R,
    M=M,
    K=no_folds,
    P=no_threads,
    parameter_search=parameter_search,
    train_config=train_config,
    predict_config={},
    file_performance=output_file,
    files_nested_performances=files_nested_performances
)
nested_crossval.run()

''' 
Results

10 folds, 1000 iterations
Average performances: {'R^2': 0.39195892884290867, 'MSE': 0.096685372795682359, 'Rp': 0.63097698640518429}. 
All performances: {'R^2': [0.3540128174408864, 0.3845516398167005, 0.4093180246371402, 0.35582843721484536, 0.4256529882160528, 0.38146883141303467, 0.4043448907281303, 0.3821415830221866, 0.3876844346556515, 0.43458564128445765], 'MSE': [0.1005585711367963, 0.098490047949598508, 0.090379491735625261, 0.10531773559960644, 0.091014639675174563, 0.097711911296211318, 0.094395097367695177, 0.0993446126710434, 0.098848199548904253, 0.090793420976168243], 'Rp': [0.60490895898789521, 0.62558290285463569, 0.64557722216816904, 0.6028882654287111, 0.65485492742890483, 0.62404470112704036, 0.63712676831280723, 0.62362315467964191, 0.62926055403512238, 0.66190240902891562]}. 
'''