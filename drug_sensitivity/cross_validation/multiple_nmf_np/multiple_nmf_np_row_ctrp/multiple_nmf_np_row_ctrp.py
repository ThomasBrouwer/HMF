"""
Run nested cross-validation on the CCLE EC dataset with multiple non-probabilistic 
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
R_ctrp,     M_ctrp,   cell_lines, drugs = load_data_without_empty(location_data+"ctrp_ec50_row_01.txt")
R_ccle_ec,  M_ccle_ec                   = load_data_filter(location_data+"ccle_ec50_row_01.txt",rows=None,columns=drugs)
R_gdsc,     M_gdsc                      = load_data_filter(location_data+"gdsc_ic50_row_01.txt",rows=None,columns=drugs)
R_ccle_ic,  M_ccle_ic                   = load_data_filter(location_data+"ccle_ic50_row_01.txt",rows=None,columns=drugs)

R_concat = numpy.concatenate((R_ctrp,R_ccle_ec,R_gdsc,R_ccle_ic),axis=0) #rows
M_concat = numpy.concatenate((M_ctrp,M_ccle_ec,M_gdsc,M_ccle_ic),axis=0) #rows
no_rows, _ = R_ctrp.shape

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
Average performances: {'R^2': 0.3587569725843171, 'MSE': 0.1019470143184497, 'Rp': 0.60443242077582515}. 
All performances: {'R^2': [0.30731842462484193, 0.35546784237974005, 0.38148445999631986, 0.327843412536585, 0.3720279804878843, 0.3221439761220971, 0.3756086091421492, 0.35148615946483275, 0.3906114319810552, 0.4035774291076658], 'MSE': [0.10782732436976318, 0.10314432081705649, 0.094638269775843387, 0.10989309967974434, 0.099512395658614181, 0.10708370255305506, 0.098949014653169381, 0.10427365643881886, 0.098375684342569575, 0.095772674895862567], 'Rp': [0.57326118377364255, 0.60825555579102031, 0.62478573525629033, 0.5760123920010336, 0.61058404346749395, 0.56976091313113519, 0.61317223250350561, 0.59583672554860168, 0.62985217965196938, 0.64280324663355859]}. 
'''