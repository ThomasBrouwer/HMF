"""
Run nested cross-validation on the CCLE EC dataset with multiple non-probabilistic 
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
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data =                 location+"data_row_01/"
location_features_drugs =       location+"features_drugs/"
location_features_cell_lines =  location+"features_cell_lines/"
location_kernels =              location+"kernels_features/"

''' Concatenate the datasets by ROWS. We remove the columns of the other datasets '''
R_ccle_ec,  M_ccle_ec, cell_lines, drugs = load_data_without_empty(location_data+"ccle_ec50_row_01.txt")
R_ctrp,     M_ctrp                       = load_data_filter(location_data+"ctrp_ec50_row_01.txt",rows=cell_lines,columns=None)
R_gdsc,     M_gdsc                       = load_data_filter(location_data+"gdsc_ic50_row_01.txt",rows=cell_lines,columns=None)
R_ccle_ic,  M_ccle_ic                    = load_data_filter(location_data+"ccle_ic50_row_01.txt",rows=cell_lines,columns=None)

R_concat = numpy.concatenate((R_ccle_ec,R_gdsc,R_ctrp,R_ccle_ic),axis=1) #columns
M_concat = numpy.concatenate((M_ccle_ec,M_gdsc,M_ctrp,M_ccle_ic),axis=1) #columns
_, no_columns = R_ccle_ec.shape

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
Average performances: {'R^2': 0.241637913885654, 'MSE': 0.11566788017486926, 'Rp': 0.50466323345609299}. 
All performances: {'R^2': [0.2976464920048776, 0.2537608345809964, 0.20852229788698373, 0.13514118013089393, 0.27898328196072997, 0.24233826609646147, 0.3009615207501801, 0.2870922314944424, 0.21840261932236682, 0.19353041462860743], 'MSE': [0.092712203831900139, 0.11828196150851605, 0.12756574532563669, 0.14078296351411454, 0.097985177064644352, 0.11672239917156285, 0.11101843491033662, 0.11618087759435422, 0.12615190637188645, 0.10927713245574076], 'Rp': [0.55840024476649042, 0.505210057831051, 0.48131281912516694, 0.41160089984560116, 0.54924136071364305, 0.49307326894853803, 0.55949535049961141, 0.54330993550383044, 0.47203156425314341, 0.47295683307385389]}. 
'''