"""
Run nested cross-validation on the GDSC dataset with multiple non-probabilistic 
NMF, where we concatenate the columns of the datasets (so R=[R1;..;R4]).
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.code.models.nmf_np import nmf_np
from HMF.code.cross_validation.multiple_nmf_nested_matrix_cross_validation import MultipleNMFNestedCrossValidation
from HMF.drug_sensitivity.load_dataset import load_data_without_empty, load_data_filter

import numpy, random

''' Load datasets '''
location = project_location+"HMF/datasets_drug_sensitivity/data/overlap/"
location_data =                 location+"data_row_01/"
location_features_drugs =       location+"features_drugs/"
location_features_cell_lines =  location+"features_cell_lines/"
location_kernels =              location+"kernels_features/"

''' Concatenate the datasets by ROWS. We remove the columns of the other datasets '''
R_gdsc,     M_gdsc,     cell_lines, drugs = load_data_without_empty(location_data+"gdsc_ic50_row_01.txt")
R_ctrp,     M_ctrp                        = load_data_filter(location_data+"ctrp_ec50_row_01.txt",rows=cell_lines,columns=None)
R_ccle_ec,  M_ccle_ec                     = load_data_filter(location_data+"ccle_ec50_row_01.txt",rows=cell_lines,columns=None)
R_ccle_ic,  M_ccle_ic                     = load_data_filter(location_data+"ccle_ic50_row_01.txt",rows=cell_lines,columns=None)

R_concat = numpy.concatenate((R_gdsc,R_ctrp,R_ccle_ec,R_ccle_ic),axis=1) #columns
M_concat = numpy.concatenate((M_gdsc,M_ctrp,M_ccle_ec,M_ccle_ic),axis=1) #columns
_, no_columns = R_gdsc.shape

''' Remove entirely empty columns, due to the other three datasets that we concatenate '''
def remove_empty_columns(R,M):
    new_R, new_M = [], []
    for j,sum_column in enumerate(M.sum(axis=0)):
        if sum_column > 0:
            new_R.append(R[:,j])
            new_M.append(M[:,j])
    return numpy.array(new_R).T, numpy.array(new_M).T
        
R_concat, M_concat = remove_empty_columns(R_concat,M_concat)

''' Settings DI-MMTF '''
no_folds, no_threads = 10, 5
output_file = "./results.txt"
files_nested_performances = ["./fold_%s.txt" % fold for fold in range(1,no_folds+1)]

train_config = {
    'iterations' : 1000,
    'init_UV' : 'exponential',
    'expo_prior' : 0.1
}
K_range = range(1,10+1)
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
nested_crossval.run(columns_M=no_columns)

''' 
Results

10 folds, 1000 iterations
Average performances: {'R^2': 0.5689876364691212, 'MSE': 0.08593917235039715, 'Rp': 0.75650834662287758}. 
All performances: {'R^2': [0.5842454979076124, 0.5771378908897286, 0.5656134626510327, 0.5928854119237827, 0.5618628659942957, 0.5549335622601177, 0.5763267658627742, 0.5481801266248671, 0.5611819451207196, 0.5675088354562807], 'MSE': [0.08222464702984543, 0.085455799749088915, 0.085800243464207948, 0.081331763732966883, 0.088974546758137887, 0.088757490032779746, 0.083763951931385097, 0.089218655913115813, 0.087262041391619444, 0.086602583500824434], 'Rp': [0.76488759606247725, 0.76334629725936076, 0.75488196974045596, 0.77131657576357249, 0.75328533849615098, 0.74831695785113794, 0.76165409773999904, 0.74169683495062677, 0.7502045773570859, 0.75549322100790839]}. 
'''