"""
Run nested cross-validation on the CCLE EC dataset with multiple non-probabilistic 
NMF, where we concatenate the rows of the datasets (so R=[R1;..;R4]).
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
R_ccle_ec,  M_ccle_ec, cell_lines, drugs = load_data_without_empty(location_data+"ccle_ec50_row_01.txt")
R_ctrp,     M_ctrp                       = load_data_filter(location_data+"ctrp_ec50_row_01.txt",rows=None,columns=drugs)
R_gdsc,     M_gdsc                       = load_data_filter(location_data+"gdsc_ic50_row_01.txt",rows=None,columns=drugs)
R_ccle_ic,  M_ccle_ic                    = load_data_filter(location_data+"ccle_ic50_row_01.txt",rows=None,columns=drugs)

R_concat = numpy.concatenate((R_ccle_ec,R_gdsc,R_ctrp,R_ccle_ic),axis=0) #rows
M_concat = numpy.concatenate((M_ccle_ec,M_gdsc,M_ctrp,M_ccle_ic),axis=0) #rows
no_rows, _ = R_ccle_ec.shape

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
Average performances: {'R^2': 0.040519376329985445, 'MSE': 0.14633211495734219, 'Rp': 0.34728414274876818}. 
All performances: {'R^2': [0.11002957803160784, 0.026323843897260613, -0.08427026430695483, -0.004430826361998541, 0.026303357930229865, 0.03036495341090062, 0.07917093883291859, 0.13496941781611616, 0.03146571311616253, 0.05526705093361162], 'MSE': [0.1174780480579144, 0.15433165525859069, 0.17475633745774363, 0.16350269562097391, 0.13232403007229535, 0.14937817748245821, 0.14624230886464518, 0.14097196948036247, 0.156324022696956, 0.12801190458148234], 'Rp': [0.4097461478809819, 0.30023097963543383, 0.27946636099008404, 0.31133128497200024, 0.36214876932294338, 0.28471707251490774, 0.36324927542368168, 0.4133697926722773, 0.34148019739736329, 0.4071015466780083]}. 
'''