"""
Run nested cross-validation on the CTRP dataset with multiple non-probabilistic 
NMF, where we concatenate the columns of the datasets (so R=[R1;..;R4]).
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../../"
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
R_ccle_ec,  M_ccle_ec                   = load_data_filter(location_data+"ccle_ec50_row_01.txt",rows=cell_lines,columns=None)
R_gdsc,     M_gdsc                      = load_data_filter(location_data+"gdsc_ic50_row_01.txt",rows=cell_lines,columns=None)
R_ccle_ic,  M_ccle_ic                   = load_data_filter(location_data+"ccle_ic50_row_01.txt",rows=cell_lines,columns=None)

R_concat = numpy.concatenate((R_ctrp,R_ccle_ec,R_gdsc,R_ccle_ic),axis=1) #columns
M_concat = numpy.concatenate((M_ctrp,M_ccle_ec,M_gdsc,M_ccle_ic),axis=1) #columns
_, no_columns = R_ctrp.shape

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
nested_crossval.run(columns_M=no_columns)

''' 
Results

10 folds, 1000 iterations
Average performances: {'R^2': 0.41624329735544385, 'MSE': 0.09280951172763327, 'Rp': 0.64633329675217577}. 
All performances: {'R^2': [0.3899304645636511, 0.3790407521111314, 0.4269696430024711, 0.4134523384870097, 0.42332526145063476, 0.40241311277187597, 0.4476871573973421, 0.41214226893803996, 0.43071748565496293, 0.4367544891773195], 'MSE': [0.094967396310459293, 0.099371953937949342, 0.08767864024072429, 0.095896613729276825, 0.091383505897980777, 0.094403257074350141, 0.08752652960307368, 0.094520843276180419, 0.091901226691884136, 0.090445150514453718], 'Rp': [0.62641601836306637, 0.61814158117639073, 0.65558359539736255, 0.64460034734634408, 0.65258422272438665, 0.63555896609360474, 0.66938306817536197, 0.64297944145369879, 0.65662778518680665, 0.66145794160473481]}. 
'''