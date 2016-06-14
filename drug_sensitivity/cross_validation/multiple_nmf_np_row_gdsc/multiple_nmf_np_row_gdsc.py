"""
Run nested cross-validation on the GDSC dataset with multiple non-probabilistic 
NMF, where we concatenate the rows of the datasets (so R=[R1,..,R4].T).
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
R_gdsc,     M_gdsc,     cell_lines, drugs = load_data_without_empty(location_data+"gdsc_ic50_row_01.txt")
R_ctrp,     M_ctrp                        = load_data_filter(location_data+"ctrp_ec50_row_01.txt",rows=None,columns=drugs)
R_ccle_ec,  M_ccle_ec                     = load_data_filter(location_data+"ccle_ec50_row_01.txt",rows=None,columns=drugs)
R_ccle_ic,  M_ccle_ic                     = load_data_filter(location_data+"ccle_ic50_row_01.txt",rows=None,columns=drugs)

R_concat = numpy.concatenate((R_gdsc,R_ctrp,R_ccle_ec,R_ccle_ic),axis=0) #rows
M_concat = numpy.concatenate((M_gdsc,M_ctrp,M_ccle_ec,M_ccle_ic),axis=0) #rows
no_rows, _ = R_gdsc.shape

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
nested_crossval.run(rows_M=no_rows)

''' 
Results

10 folds, 1000 iterations
Average performances: {'R^2': 0.529957568835441, 'MSE': 0.09371093418782471, 'Rp': 0.73154251028440664}. 
All performances: {'R^2': [0.5416370018390801, 0.5384885866712273, 0.5148942046583935, 0.5601362883081549, 0.5395234229558483, 0.5108883589228213, 0.5304457285221458, 0.5087676042378362, 0.5240411919805708, 0.5307533002583319], 'MSE': [0.090651419396892682, 0.093266400724161314, 0.09581833636057327, 0.087874255852825087, 0.093511121416859258, 0.097541216157027849, 0.092835039497699709, 0.097001253538247248, 0.094647740092469845, 0.093962558841490823], 'Rp': [0.73803074748796982, 0.73688687766878158, 0.72219709458012404, 0.75200569438921128, 0.738015387578149, 0.71964990732612677, 0.73268175227171717, 0.71837582256461963, 0.72654039499873058, 0.73104142397863614]}. 
'''