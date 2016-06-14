"""
Run nested cross-validation on the GDSC dataset with non-probabilistic NMF.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from DI_MMTF.code.models.nmtf_np import nmtf_np
from DI_MMTF.code.cross_validation.nested_matrix_cross_validation import MatrixNestedCrossValidation
from DI_MMTF.drug_sensitivity.load_dataset import load_data_without_empty

import numpy, random, itertools

''' Load datasets '''
location = project_location+"DI_MMTF/data/datasets_drug_sensitivity/overlap/"
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

R, M = R_gdsc, M_gdsc

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

Average performances: {'R^2': 0.5592294249844292, 'MSE': 0.087890298286996141, 'Rp': 0.75336699618135516}. 
All performances: {'R^2': [0.5770204298372774, 0.5779383638731931, 0.5571590852395782, 0.589643870947444, 0.5376625306148366, 0.5356771949454411, 0.5808491323870373, 0.5496139060926615, 0.5227744662075997, 0.5639552696992234], 'MSE': [0.083653564020184826, 0.085294032928402555, 0.087470156267371882, 0.081979346139836956, 0.093889021484569679, 0.092597696089833342, 0.082869840003531342, 0.088935534508922456, 0.094899637377925047, 0.087314154049383294], 'Rp': [0.7643040775367469, 0.76491408729557553, 0.74958569904872108, 0.77445427874441874, 0.73967847290348732, 0.7446629555937827, 0.76643636114140434, 0.74355716779512093, 0.72873384856265144, 0.75734301319164243]}. 
'''