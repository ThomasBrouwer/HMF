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

R, M = R_ccle_ic, M_ccle_ic

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
Average performances: {'R^2': 0.5667028774926035, 'MSE': 0.074719063424955962, 'Rp': 0.75674095303607802}. 
All performances: {'R^2': [0.5556591730163896, 0.48106029511440207, 0.6280137718318964, 0.47454458064317406, 0.6280966272316917, 0.5656885930454448, 0.5835698034715201, 0.5603665299431175, 0.5171847338330582, 0.6728446667953405], 'MSE': [0.082674881128310659, 0.088921914938153254, 0.06251723822970047, 0.082103255696412664, 0.068384307257585153, 0.075987738551334921, 0.067300480279688685, 0.070628658253713034, 0.084427045309881488, 0.064245114604779399], 'Rp': [0.74959173430769044, 0.70125122168982668, 0.79559402834969351, 0.70052428717539594, 0.79497397577934503, 0.7558921825740641, 0.76733201835125553, 0.75167304025943382, 0.72607742900055494, 0.82449961287352103]}. 
'''