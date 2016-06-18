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

''' Settings NMF '''
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

R, M = R_ccle_ec, M_ccle_ec

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
Average performances: {'R^2': -0.013962969375202649, 'MSE': 0.15353431475091153, 'Rp': 0.36847401676513203}. 
All performances: {'R^2': [0.0024283451581081916, 0.06889808439425615, -0.17775443721108286, -0.013325322046825683, -0.47650124364389135, 0.006981663956445838, 0.15916143664613835, 0.17673773162693662, 0.05889974821283772, 0.05484429915505051], 'MSE': [0.13168164684566463, 0.14758346391581598, 0.18982356949830334, 0.1649505544306499, 0.20065448161596522, 0.15298051546986743, 0.13353854485375344, 0.13416508706365096, 0.1518960961039596, 0.12806918771148462], 'Rp': [0.40516878786957944, 0.3541438082593753, 0.2724195515769306, 0.34624160103853108, 0.31127678878328141, 0.30754272946045241, 0.44914589576631803, 0.45523923246270165, 0.36646633340554302, 0.41709543902860663]}. 

Massive overfitting due to sparse dataset. Only reasonable performance with K=1.
'''