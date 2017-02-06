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

R_ctrp,     M_ctrp,   cell_lines, drugs   = load_data_without_empty(location_data+"ctrp_ec50_row_01.txt")
R_ccle_ec,  M_ccle_ec                     = load_data_filter(location_data+"ccle_ec50_row_01.txt",cell_lines,drugs)
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
alpha_l = [1., 3., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC


''' Assemble R, C, D. '''
D = [(R_gdsc,    M_gdsc,    'Cell_lines', alpha_l[0]), 
     (R_ctrp,    M_ctrp,    'Cell_lines', alpha_l[1]), 
     (R_ccle_ic, M_ccle_ic, 'Cell_lines', alpha_l[2]),
     (R_ccle_ec, M_ccle_ec, 'Cell_lines', alpha_l[3])]
R, C = [], []

main_dataset = 'D'
index_main = 1 # CTRP
file_performance = 'results_1010_1311.txt'


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

Performance fold 1: {'R^2': 0.39011896612091124, 'MSE': 0.094938052930641698, 'Rp': 0.63085086509336052}.
Performance fold 2: {'R^2': 0.41175664870450446, 'MSE': 0.094136437146198113, 'Rp': 0.64416163486365408}.
Performance fold 3: {'R^2': 0.43009029587237446, 'MSE': 0.087201153146096222, 'Rp': 0.66091157956987712}.
Performance fold 4: {'R^2': 0.4130143305989229, 'MSE': 0.095968225085029044, 'Rp': 0.6460125496872734}.
Performance fold 5: {'R^2': 0.44066312953784836, 'MSE': 0.08863603827942014, 'Rp': 0.66524773249073599}.
Performance fold 6: {'R^2': 0.39460968529708784, 'MSE': 0.095635996590072228, 'Rp': 0.63217310378784441}.
Performance fold 7: {'R^2': 0.45678552811330697, 'MSE': 0.086084685864554034, 'Rp': 0.67648307217086368}.
Performance fold 8: {'R^2': 0.4014056164805184, 'MSE': 0.096247174989833165, 'Rp': 0.63799346535118873}.
Performance fold 9: {'R^2': 0.4306914826638708, 'MSE': 0.091905424443824185, 'Rp': 0.65763150356847688}.
Performance fold 10: {'R^2': 0.4496949651545358, 'MSE': 0.088367187574671921, 'Rp': 0.67163407008757359}.
Average performance: {'R^2': 0.42188306485438815, 'MSE': 0.091912037605034067, 'Rp': 0.65230995766708477}.
"""