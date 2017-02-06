"""
Run nested cross-validation on the CCLE IC dataset with multiple non-probabilistic 
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
R_ccle_ic,  M_ccle_ic, cell_lines, drugs = load_data_without_empty(location_data+"ccle_ic50_row_01.txt")
R_ctrp,     M_ctrp                       = load_data_filter(location_data+"ctrp_ec50_row_01.txt",rows=cell_lines,columns=None)
R_gdsc,     M_gdsc                       = load_data_filter(location_data+"gdsc_ic50_row_01.txt",rows=cell_lines,columns=None)
R_ccle_ec,  M_ccle_ec                    = load_data_filter(location_data+"ccle_ec50_row_01.txt",rows=cell_lines,columns=None)

R_concat = numpy.concatenate((R_ccle_ic,R_gdsc,R_ctrp,R_ccle_ec),axis=1) #columns
M_concat = numpy.concatenate((M_ccle_ic,M_gdsc,M_ctrp,M_ccle_ec),axis=1) #columns
_, no_columns = R_ccle_ic.shape

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
Average performances: {'R^2': 0.6142501937761284, 'MSE': 0.066641140584958472, 'Rp': 0.79140855604251015}. 
All performances: {'R^2': [0.5609868700656795, 0.5740253038849876, 0.6267846181056385, 0.5747265581429164, 0.6187900639623587, 0.6672896689790916, 0.667892272239147, 0.5554699572966357, 0.5827431191847974, 0.713793505900032], 'MSE': [0.081683599901177462, 0.072992074680612945, 0.062723813878223317, 0.066449660335455893, 0.070095565957351502, 0.058211470484334511, 0.053672883881207459, 0.071415309816020373, 0.072963238843123723, 0.056203788072077446], 'Rp': [0.76526403013971511, 0.76665169195615068, 0.80079022734757488, 0.76247966086293373, 0.80092365801532672, 0.81817060712793432, 0.82412776964201795, 0.74998743070787521, 0.77456827265787553, 0.85112221196769733]}. 
'''