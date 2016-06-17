"""
Run nested cross-validation on the GDSC dataset with non-probabilistic NMF.
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

R, M = R_gdsc, M_gdsc

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
    file_performance=output_file,
    files_nested_performances=files_nested_performances
)
nested_crossval.run()

''' 
Results

5 folds, 1000 iterations
Average performances: {'R^2': 0.5283139947182672, 'MSE': 0.094087457940215308, 'Rp': 0.73033339456613477}. 
All performances: {'R^2': [0.5303182005140784, 0.5353655987813535, 0.5185442643999467, 0.5428180007774619, 0.5145239091184959], 'MSE': [0.093996366009549845, 0.092301905943197649, 0.096919272240521659, 0.090342698359619888, 0.096877047148187539], 'Rp': [0.73067844527487114, 0.73527228465975369, 0.72666536548098948, 0.73930936319322138, 0.71974151422183841]}. 

10 folds, 1000 iterations
Average performances: {'R^2': 0.5506427646610004, 'MSE': 0.089610700855510803, 'Rp': 0.74696250504011596}. 
All performances: {'R^2': [0.576748431760556, 0.5443926889438082, 0.568968464260885, 0.577609128574648, 0.5381185987328203, 0.5110393309883685, 0.5433215595903147, 0.5544736660787886, 0.5408943657700396, 0.5508614119097752], 'MSE': [0.08370735765498287, 0.092073246334979, 0.085137561888703972, 0.084383600008205148, 0.093796405609449865, 0.097511108513613803, 0.090289373621810554, 0.087975901523361821, 0.091296368533244993, 0.089936084866756055], 'Rp': [0.76038694734680801, 0.74211000045938491, 0.76126639064196255, 0.76287291359623666, 0.7411826165482116, 0.72664869663819276, 0.73970204411170815, 0.74991356049919411, 0.73774028869552799, 0.74780159186393325]}. 
'''