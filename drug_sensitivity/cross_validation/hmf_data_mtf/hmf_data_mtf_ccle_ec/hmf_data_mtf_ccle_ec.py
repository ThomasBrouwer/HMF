"""
Run the cross validation for HMF (datasets, using MTF) on the drug sensitivity
datasets.
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

R_ccle_ec,  M_ccle_ec, cell_lines, drugs  = load_data_without_empty(location_data+"ccle_ec50_row_01.txt")
R_ctrp,     M_ctrp                        = load_data_filter(location_data+"ctrp_ec50_row_01.txt",cell_lines,drugs)
R_gdsc,     M_gdsc                        = load_data_filter(location_data+"gdsc_ic50_row_01.txt",cell_lines,drugs)
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
alpha_n = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC
alpha_m = []


''' Assemble R, C, D. '''
R = [(R_gdsc,    M_gdsc,    'Cell_lines', 'Drugs', alpha_n[0]), 
     (R_ctrp,    M_ctrp,    'Cell_lines', 'Drugs', alpha_n[1]), 
     (R_ccle_ic, M_ccle_ic, 'Cell_lines', 'Drugs', alpha_n[2]),
     (R_ccle_ec, M_ccle_ec, 'Cell_lines', 'Drugs', alpha_n[3])]
C, D = [], []

main_dataset = 'R'
index_main = 3 # CCLE EC
file_performance = 'results_1010_1111.txt'


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

Performance fold 1: {'R^2': 0.3560353165280301, 'MSE': 0.085004750905313259, 'Rp': 0.6024340820372972}.
Performance fold 2: {'R^2': 0.30387959187257285, 'MSE': 0.11033793337982849, 'Rp': 0.56327706717299408}.
Performance fold 3: {'R^2': 0.21811818645073444, 'MSE': 0.12601913614962504, 'Rp': 0.51028629608486309}.
Performance fold 4: {'R^2': 0.25639547924786954, 'MSE': 0.12104501417910252, 'Rp': 0.52653559252884563}.
Performance fold 5: {'R^2': 0.29016142980477766, 'MSE': 0.096466082197146238, 'Rp': 0.57540839717725345}.
Performance fold 6: {'R^2': 0.32734780811439956, 'MSE': 0.10362616208738566, 'Rp': 0.57922155109525908}.
Performance fold 7: {'R^2': 0.3616271313886389, 'MSE': 0.10138377051648338, 'Rp': 0.6038136794286274}.
Performance fold 8: {'R^2': 0.30534232270502537, 'MSE': 0.11320670378577408, 'Rp': 0.56654076807280718}.
Performance fold 9: {'R^2': 0.2969546421044683, 'MSE': 0.11347339993326638, 'Rp': 0.54971831306965913}.
Performance fold 10: {'R^2': 0.24444090590144008, 'MSE': 0.10237872909481779, 'Rp': 0.51516642241145527}.
Average performance: {'R^2': 0.2960302814117957, 'MSE': 0.10729416822287427, 'Rp': 0.55924021690790604}.
"""