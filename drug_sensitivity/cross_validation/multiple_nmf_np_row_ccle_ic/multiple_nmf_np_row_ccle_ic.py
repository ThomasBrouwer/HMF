"""
Run nested cross-validation on the CCLE IC dataset with multiple non-probabilistic 
NMF, where we concatenate the rows of the datasets (so R=[R1;..;R4]).
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.code.models.nmf_np import nmf_np
from HMF.code.cross_validation.multiple_nmf_nested_matrix_cross_validation import MultipleNMFNestedCrossValidation
from HMF.drug_sensitivity.load_dataset import load_data_without_empty, load_data_filter

import numpy, random

''' Load datasets '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data =                 location+"data_row_01/"
location_features_drugs =       location+"features_drugs/"
location_features_cell_lines =  location+"features_cell_lines/"
location_kernels =              location+"kernels_features/"

''' Concatenate the datasets by ROWS. We remove the columns of the other datasets '''
R_ccle_ic,  M_ccle_ic, cell_lines, drugs = load_data_without_empty(location_data+"ccle_ic50_row_01.txt")
R_ctrp,     M_ctrp                       = load_data_filter(location_data+"ctrp_ec50_row_01.txt",rows=None,columns=drugs)
R_gdsc,     M_gdsc                       = load_data_filter(location_data+"gdsc_ic50_row_01.txt",rows=None,columns=drugs)
R_ccle_ec,  M_ccle_ec                    = load_data_filter(location_data+"ccle_ec50_row_01.txt",rows=None,columns=drugs)

R_concat = numpy.concatenate((R_ccle_ic,R_gdsc,R_ctrp,R_ccle_ec),axis=0) #rows
M_concat = numpy.concatenate((M_ccle_ic,M_gdsc,M_ctrp,M_ccle_ec),axis=0) #rows
no_rows, _ = R_ccle_ic.shape

''' Remove entirely empty rows, due to the other three datasets that we concatenate '''
def remove_empty_rows(R,M):
    new_R, new_M = [], []
    for i,sum_row in enumerate(M.sum(axis=1)):
        if sum_row > 0:
            new_R.append(R[i])
            new_M.append(M[i])
    return numpy.array(new_R), numpy.array(new_M)
        
R_concat, M_concat = remove_empty_rows(R_concat,M_concat)

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


''' Run the nested cross-validation '''
random.seed(0)
numpy.random.seed(0)
nested_crossval = MultipleNMFNestedCrossValidation(
    method=nmf_np,
    X=R_concat,
    M=M_concat,
    K=no_folds,
    P=no_threads,
    parameter_search=parameter_search,
    train_config=train_config,
    file_performance=output_file,
    files_nested_performances=files_nested_performances
)
nested_crossval.run(rows_M=no_rows)

''' 
Results

10 folds, 1000 iterations
Average performances: {'R^2': 0.4497282604232765, 'MSE': 0.094741745806284847, 'Rp': 0.6976867463135431}. 
All performances: {'R^2': [0.4617874828604077, 0.4225768737081358, 0.49009736158936046, 0.33951363523192424, 0.491890374922771, 0.45988876045422755, 0.39032549285930485, 0.43921101929421313, 0.3909840914745327, 0.6110075118378886], 'MSE': [0.10014082248158106, 0.098943229119011003, 0.08569592717573668, 0.10320205842947494, 0.093429442339218668, 0.094498627026704016, 0.098531248422675605, 0.090092715792526895, 0.10649500400374622, 0.076388383272173468], 'Rp': [0.69770625859192781, 0.67280632953830954, 0.72796280350239262, 0.63678242619623382, 0.72843764090252805, 0.69424176231089607, 0.67933576710495058, 0.6882655184891826, 0.66301829251757749, 0.78831066398143235]}. 
'''