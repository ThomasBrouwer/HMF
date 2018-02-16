"""
Run the cross validation for HMF (features) on the drug sensitivity datasets.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../../"
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

R_gdsc, M_gdsc, cell_lines, drugs = load_data_without_empty(location_data+"gdsc_ic50_row_01.txt")

C_cnv_std,  M_cnv_std =  load_data_filter(location_kernels+"cnv_std.txt",            cell_lines,cell_lines)
C_ge_std,   M_ge_std =   load_data_filter(location_kernels+"gene_expression_std.txt",cell_lines,cell_lines)
C_mutation, M_mutation = load_data_filter(location_kernels+"mutation.txt",           cell_lines,cell_lines)

C_1d2d_std, M_1d2d_std = load_data_filter(location_kernels+"drug_1d2d_std.txt",      drugs,drugs)
C_fp,       M_fp =       load_data_filter(location_kernels+"drug_fingerprints.txt",  drugs,drugs)
C_targets,  M_targets =  load_data_filter(location_kernels+"drug_targets.txt",       drugs,drugs)


''' Settings HMF '''
iterations, burn_in, thinning = 100, 80, 2
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
    'lambdaSn' : 0.01,
    'lambdaSm' : 0.01,
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
alpha_n = [3.] # main dataset
alpha_m = [.1, .1, .1, .1, .1, .1]


''' Assemble R, C, D. '''
R = [(R_gdsc,     M_gdsc,     'Cell_lines', 'Drugs', alpha_n[0])]
C = [(C_1d2d_std, M_1d2d_std, 'Drugs',      alpha_m[0]),
     (C_fp,       M_fp,       'Drugs',      alpha_m[1]),
     (C_targets,  M_targets,  'Drugs',      alpha_m[2]),
     (C_cnv_std,  M_cnv_std,  'Cell_lines', alpha_m[3]),
     (C_ge_std,   M_ge_std,   'Cell_lines', alpha_m[4]),
     (C_mutation, M_mutation, 'Cell_lines', alpha_m[5])]
D = []

main_dataset = 'R'
index_main = 0 # CTRP
file_performance = 'results_all_3_01_div4.txt'


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