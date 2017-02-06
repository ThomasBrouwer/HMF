"""
Run the cross validation for HMF (features) on the drug sensitivity datasets.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.code.cross_validation.cross_validation_hmf import CrossValidation
from HMF.drug_sensitivity.load_dataset import load_data_without_empty, load_data_filter

import numpy, random


''' Load datasets '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data =                 location+"data_row_01/"
location_features_drugs =       location+"features_drugs/"
location_features_cell_lines =  location+"features_cell_lines/"
location_kernels =              location+"kernels_features/"

R_ctrp, M_ctrp, cell_lines, drugs = load_data_without_empty(location_data+"ctrp_ec50_row_01.txt")

R_cnv,      M_cnv =      load_data_filter(location_features_cell_lines+"cnv.txt",                 cell_lines)
R_cnv_std,  M_cnv_std =  load_data_filter(location_features_cell_lines+"cnv_std.txt",             cell_lines)
R_mutation, M_mutation = load_data_filter(location_features_cell_lines+"mutation.txt",            cell_lines)
#R_ge,       M_ge =       load_data_filter(location_features_cell_lines+"gene_expression.txt",     cell_lines)
#R_ge_std,   M_ge_std =   load_data_filter(location_features_cell_lines+"gene_expression_std.txt", cell_lines)

R_fp,       M_fp =       load_data_filter(location_features_drugs+"drug_fingerprints.txt", drugs)
R_targets,  M_targets =  load_data_filter(location_features_drugs+"drug_targets.txt",      drugs)
R_1d2d,     M_1d2d =     load_data_filter(location_features_drugs+"drug_1d2d.txt",         drugs)
R_1d2d_std, M_1d2d_std = load_data_filter(location_features_drugs+"drug_1d2d_std.txt",     drugs)


''' Settings HMF '''
iterations, burn_in, thinning = 200, 180, 2
no_folds = 10

settings = {
    'priorF'  : 'exponential',
    'orderG'  : 'normal',
    'priorSn' : 'normal',
    'priorSm' : 'normal',
    'orderF'  : 'columns',
    'orderG'  : 'rows',
    'orderSn' : 'rows',
    'orderSm' : 'rows',
    'ARD'     : True
}
hyperparameters = {
    'alphatau' : 1.,
    'betatau'  : 1.,
    'alpha0'   : 0.001,
    'beta0'    : 0.001,
    'lambdaF'  : 0.1,
    'lambdaG'  : 0.1,
    'lambdaSn' : 0.1,
    'lambdaSm' : 0.1,
}
init = {
    'F'       : 'kmeans',
    'Sn'      : 'least',
    'Sm'      : 'least',
    'G'       : 'least',
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

K = {'Cell_lines':10, 'Drugs':10}
alpha_n = [2.] # main dataset
alpha_l = [1., 1.]#, 1., 1., 1.]


''' Assemble R, C, D. '''
R = [(R_ctrp,     M_ctrp,     'Cell_lines', 'Drugs', alpha_n[0])]
C = []
D = [#(R_cnv_std,  M_cnv_std,  'Cell_lines', alpha_l[0]),     
     (R_mutation, M_mutation, 'Cell_lines', alpha_l[0]),
     #(R_fp,       M_fp,       'Drugs',      alpha_l[1]),]
     (R_targets,  M_targets,  'Drugs',      alpha_l[1]),]
     #(R_1d2d_std, M_1d2d_std, 'Drugs',      alpha_l[1])]

main_dataset = 'R'
index_main = 0 # CTRP
file_performance = 'results_mut_target_211.txt'


''' Run the cross-validation framework '''
random.seed(0)
numpy.random.seed(0)
crossval = CrossValidation(
    folds=no_folds,
    main_dataset=main_dataset,
    index_main=index_main,
    R=R,
    C=C,
    D=D,
    K=K,
    settings=settings,
    hyperparameters=hyperparameters,
    init=init,
    file_performance=file_performance
)
crossval.run(iterations=iterations,burn_in=burn_in,thinning=thinning)


"""

"""