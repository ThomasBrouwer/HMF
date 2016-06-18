"""
Run nested cross-validation on the CCLE EC50 dataset with non-probabilistic NMF.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
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

R, M = R_ccle_ic, M_ccle_ic

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
Average performances: {'R^2': 0.5676009256216047, 'MSE': 0.074578741958192307, 'Rp': 0.7578788216732778}. 
All performances: {'R^2': [0.5556591730163889, 0.4810602951144014, 0.6280137718318963, 0.48352506193318634, 0.6280966272316919, 0.5656885930454449, 0.5835698034715205, 0.560366529943118, 0.5171847338330584, 0.6728446667953403], 'MSE': [0.082674881128310798, 0.088921914938153379, 0.062517238229700497, 0.080700041028775898, 0.068384307257585097, 0.075987738551334894, 0.067300480279688615, 0.07062865825371295, 0.084427045309881446, 0.064245114604779441], 'Rp': [0.74959173430768944, 0.70125122168982545, 0.79559402834969462, 0.71190297354739485, 0.79497397577934636, 0.75589218257406487, 0.76733201835125464, 0.75167304025943293, 0.72607742900055472, 0.8244996128735208]}. 
'''