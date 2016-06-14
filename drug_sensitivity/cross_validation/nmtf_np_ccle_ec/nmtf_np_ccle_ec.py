"""
Run nested cross-validation on the GDSC dataset with non-probabilistic NMF.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.code.models.nmtf_np import nmtf_np
from HMF.code.cross_validation.nested_matrix_cross_validation import MatrixNestedCrossValidation
from DI_MMTF.drug_sensitivity.load_dataset import load_data_without_empty

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

R, M = R_ccle_ec, M_ccle_ec

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
Average performances: {'R^2': 0.012388392036882124, 'MSE': 0.15059895641873849, 'Rp': 0.37225680139056982}. 
All performances: {'R^2': [0.0024283451581086357, 0.0688980843942566, -0.43324080802650133, -0.013325322046825905, 0.04249874129237452, 0.006981663956445505, 0.15916143664613847, 0.17673773162693673, 0.0588997482128375, 0.05484429915505051], 'MSE': [0.13168164684566458, 0.14758346391581592, 0.23100136797146498, 0.16495055443064993, 0.1301230998210732, 0.15298051546986752, 0.13353854485375341, 0.13416508706365096, 0.15189609610395963, 0.12806918771148462], 'Rp': [0.40516878786957944, 0.35414380825937536, 0.2163989571482037, 0.34624160103853108, 0.40512522946638624, 0.30754272946045241, 0.44914589576631814, 0.4552392324627017, 0.36646633340554313, 0.41709543902860691]}. 
'''