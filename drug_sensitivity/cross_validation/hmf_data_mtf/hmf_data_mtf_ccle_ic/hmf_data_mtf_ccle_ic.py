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

R_ccle_ic,  M_ccle_ic, cell_lines, drugs  = load_data_without_empty(location_data+"ccle_ic50_row_01.txt")
R_ctrp,     M_ctrp                        = load_data_filter(location_data+"ctrp_ec50_row_01.txt",cell_lines,drugs)
R_gdsc,     M_gdsc                        = load_data_filter(location_data+"gdsc_ic50_row_01.txt",cell_lines,drugs)
R_ccle_ec,  M_ccle_ec                     = load_data_filter(location_data+"ccle_ec50_row_01.txt",cell_lines,drugs)


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
index_main = 2 # CCLE IC
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

Performance fold 1: {'R^2': 0.6943553511528819, 'MSE': 0.056868812128919664, 'Rp': 0.83500851358340322}.
Performance fold 2: {'R^2': 0.6290465590378098, 'MSE': 0.063564013338557296, 'Rp': 0.79772470597837275}.
Performance fold 3: {'R^2': 0.7455447165588269, 'MSE': 0.042764598173534771, 'Rp': 0.8649879407927028}.
Performance fold 4: {'R^2': 0.627666902298345, 'MSE': 0.058177646283018136, 'Rp': 0.79478838675479269}.
Performance fold 5: {'R^2': 0.7201813105960235, 'MSE': 0.051452093833355278, 'Rp': 0.85022132642616455}.
Performance fold 6: {'R^2': 0.6691841207590581, 'MSE': 0.057880014519215768, 'Rp': 0.81899525489725844}.
Performance fold 7: {'R^2': 0.7055028749520937, 'MSE': 0.047594526338236243, 'Rp': 0.84396337793391485}.
Performance fold 8: {'R^2': 0.572537894659678, 'MSE': 0.068673285841016563, 'Rp': 0.7633774184229789}.
Performance fold 9: {'R^2': 0.6821503525418906, 'MSE': 0.055580484852351272, 'Rp': 0.82706464313567241}.
Performance fold 10: {'R^2': 0.7171081382600711, 'MSE': 0.055552877283744952, 'Rp': 0.85115757271730474}.
Average performance: {'R^2': 0.676327822081668, 'MSE': 0.055810835259194988, 'Rp': 0.82472891406425641}.
"""