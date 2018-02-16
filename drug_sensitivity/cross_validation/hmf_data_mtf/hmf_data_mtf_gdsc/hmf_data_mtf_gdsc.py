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
alpha_n = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC
alpha_m = []


''' Assemble R, C, D. '''
R = [(R_gdsc,    M_gdsc,    'Cell_lines', 'Drugs', alpha_n[0]), 
     (R_ctrp,    M_ctrp,    'Cell_lines', 'Drugs', alpha_n[1]), 
     (R_ccle_ic, M_ccle_ic, 'Cell_lines', 'Drugs', alpha_n[2]),
     (R_ccle_ec, M_ccle_ec, 'Cell_lines', 'Drugs', alpha_n[3])]
C, D = [], []

main_dataset = 'R'
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

Performance fold 1: {'R^2': 0.6181263464635397, 'MSE': 0.075523959966777354, 'Rp': 0.78642346279208186}.
Performance fold 2: {'R^2': 0.6101908013003439, 'MSE': 0.078776168653464915, 'Rp': 0.78210723275350602}.
Performance fold 3: {'R^2': 0.6302263959802499, 'MSE': 0.073037864951243536, 'Rp': 0.79424525646462441}.
Performance fold 4: {'R^2': 0.634564652831846, 'MSE': 0.073005247628195435, 'Rp': 0.79762266955156425}.
Performance fold 5: {'R^2': 0.5999605048893351, 'MSE': 0.081237881933016717, 'Rp': 0.77591661778199317}.
Performance fold 6: {'R^2': 0.6067077024447547, 'MSE': 0.078432418668761536, 'Rp': 0.78034874397011533}.
Performance fold 7: {'R^2': 0.611905173659589, 'MSE': 0.07672978550225841, 'Rp': 0.78304778132185593}.
Performance fold 8: {'R^2': 0.5955037907817349, 'MSE': 0.079873883897180378, 'Rp': 0.7727316117207943}.
Performance fold 9: {'R^2': 0.6188688109832338, 'MSE': 0.075790604378772913, 'Rp': 0.78698673867817115}.
Performance fold 10: {'R^2': 0.6204423212188201, 'MSE': 0.07600311466409955, 'Rp': 0.78805231335084358}.
Average performance: {'R^2': 0.6146496500553447, 'MSE': 0.076841093024377097, 'Rp': 0.78474824283855482}.
"""