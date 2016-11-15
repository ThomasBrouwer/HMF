"""
Run the cross validation for HMF (datasets, using MF) on the drug sensitivity
datasets.
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

R_gdsc,     M_gdsc,   cell_lines, drugs   = load_data_without_empty(location_data+"gdsc_ic50_row_01.txt")
R_ccle_ec,  M_ccle_ec                     = load_data_filter(location_data+"ccle_ec50_row_01.txt",cell_lines,drugs)
R_ctrp,     M_ctrp                        = load_data_filter(location_data+"ctrp_ec50_row_01.txt",cell_lines,drugs)
R_ccle_ic,  M_ccle_ic                     = load_data_filter(location_data+"ccle_ic50_row_01.txt",cell_lines,drugs)


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
alpha_l = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC


''' Assemble R, C, D. '''
D = [(R_gdsc,    M_gdsc,    'Cell_lines', alpha_l[0]), 
     (R_ctrp,    M_ctrp,    'Cell_lines', alpha_l[1]), 
     (R_ccle_ic, M_ccle_ic, 'Cell_lines', alpha_l[2]),
     (R_ccle_ec, M_ccle_ec, 'Cell_lines', alpha_l[3])]
R, C = [], []

main_dataset = 'D'
index_main = 0 # GDSC
file_performance = 'results.txt'


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
iterations, burn_in, thinning = 200, 180, 2
K = {'Cell_lines':10, 'Drugs':10}
alpha_n = [1., 1., 1., 1.]

Performance fold 1: {'R^2': 0.6244730773213181, 'MSE': 0.074268753584289901, 'Rp': 0.79039626126089346}.
Performance fold 2: {'R^2': 0.6128744432981739, 'MSE': 0.078233834005304217, 'Rp': 0.78408867191647624}.
Performance fold 3: {'R^2': 0.6237736678012971, 'MSE': 0.074312411009096455, 'Rp': 0.7905944265488799}.
Performance fold 4: {'R^2': 0.6294022906038208, 'MSE': 0.074036564209155539, 'Rp': 0.7944287045070062}.
Performance fold 5: {'R^2': 0.5883439829728234, 'MSE': 0.083596903098326417, 'Rp': 0.76901164187442284}.
Performance fold 6: {'R^2': 0.6150403456905339, 'MSE': 0.07677067912356085, 'Rp': 0.78566832300571876}.
Performance fold 7: {'R^2': 0.6200501459609491, 'MSE': 0.075119452317715954, 'Rp': 0.78777732496951181}.
Performance fold 8: {'R^2': 0.5838628010809797, 'MSE': 0.082172573077984701, 'Rp': 0.7657302864460076}.
Performance fold 9: {'R^2': 0.6104844167149992, 'MSE': 0.077457899859310092, 'Rp': 0.78246492503861742}.
Performance fold 10: {'R^2': 0.6066701066121667, 'MSE': 0.078760880517472029, 'Rp': 0.77997618323330276}.
Average performance: {'R^2': 0.6114975278057061, 'MSE': 0.07747299508022161, 'Rp': 0.78301367488008367}.
"""