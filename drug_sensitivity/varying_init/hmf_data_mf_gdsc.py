"""
Measure the convergence of the drug sensitivity datasets for HMF (datasets,
using MF). We take out 10% of the values and measure both the train and test
data convergence.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.code.cross_validation.cross_validation_hmf import CrossValidation
from HMF.drug_sensitivity.load_dataset import load_data_without_empty, load_data_filter


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
iterations, burn_in, thinning = 500, 400, 2
no_folds = 10

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
settings = {
    'priorF'  : 'exponential',
    'priorG'  : 'normal',
    'priorSn' : 'normal',
    'priorSm' : 'normal',
    'orderF'  : 'columns',
    'orderG'  : 'rows',
    'orderSn' : 'rows',
    'orderSm' : 'rows',
    'ARD'     : True
},

alpha_l = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC
K = {'Cell_lines':10, 'Drugs':10}

values_init = [
    # All expectation
    {
        'F'       : 'exp',
        'Sn'      : 'exp',
        'Sm'      : 'exp',
        'G'       : 'exp',
        'lambdat' : 'exp',
        'tau'     : 'exp'
    },
    # All random
    {
        'F'       : 'random',
        'Sn'      : 'random',
        'Sm'      : 'random',
        'G'       : 'random',
        'lambdat' : 'random',
        'tau'     : 'random'
    },
    # Kmeans and expectation
    {
        'F'       : 'kmeans',
        'Sn'      : 'exp',
        'Sm'      : 'exp',
        'G'       : 'exp',
        'lambdat' : 'exp',
        'tau'     : 'exp'
    },
    # Kmeans and random
    {
        'F'       : 'kmeans',
        'Sn'      : 'random',
        'Sm'      : 'random',
        'G'       : 'random',
        'lambdat' : 'exp',
        'tau'     : 'exp'
    },
    # Kmeans and least squares
    {
        'F'       : 'kmeans',
        'Sn'      : 'least',
        'Sm'      : 'least',
        'G'       : 'least',
        'lambdat' : 'exp',
        'tau'     : 'exp'
    },
]


R, C = [], []
main_dataset = 'D'
index_main = 0 # GDSC
file_performance = 'results.txt'


''' Run the methods with different inits and measure the convergence. '''
for init in values_init:
    ''' Split data into test and train. '''
    M_gdsc_train, M_gdsc_test = 
    D = [(R_gdsc,    M_gdsc,    'Cell_lines', alpha_l[0]), 
         (R_ctrp,    M_ctrp,    'Cell_lines', alpha_l[1]), 
         (R_ccle_ic, M_ccle_ic, 'Cell_lines', alpha_l[2]),
         (R_ccle_ec, M_ccle_ec, 'Cell_lines', alpha_l[3])]
    