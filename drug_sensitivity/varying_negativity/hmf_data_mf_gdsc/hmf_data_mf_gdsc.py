"""
Run the cross validation for HMF (datasets, using MF) on the drug sensitivity
datasets, where we vary the negativity constraints.
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
init = {
    'F'       : 'kmeans',
    'Sn'      : 'least',
    'Sm'      : 'least',
    'G'       : 'least',
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

alpha_l = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC
K = {'Cell_lines':10, 'Drugs':10}

values_settings = [
    # Nonnegative
    {
        'priorF'  : 'exponential',
        'priorG'  : 'exponential',
        'priorSn' : 'exponential',
        'priorSm' : 'exponential',
        'orderF'  : 'columns',
        'orderG'  : 'columns',
        'orderSn' : 'individual',
        'orderSm' : 'individual',
        'ARD'     : True
    },
    # Semi-nonnegative
    {
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
    # Semi-nonnegative, with column draws
    {
        'priorF'  : 'exponential',
        'priorG'  : 'normal',
        'priorSn' : 'normal',
        'priorSm' : 'normal',
        'orderF'  : 'columns',
        'orderG'  : 'columns',
        'orderSn' : 'individual',
        'orderSm' : 'individual',
        'ARD'     : True
    },
    # Real-valued
    {
        'priorF'  : 'normal',
        'priorG'  : 'normal',
        'priorSn' : 'normal',
        'priorSm' : 'normal',
        'orderF'  : 'rows',
        'orderG'  : 'rows',
        'orderSn' : 'rows',
        'orderSm' : 'rows',
        'ARD'     : True
    },
    # Real-valued, with column draws
    {
        'priorF'  : 'normal',
        'priorG'  : 'normal',
        'priorSn' : 'normal',
        'priorSm' : 'normal',
        'orderF'  : 'columns',
        'orderG'  : 'columns',
        'orderSn' : 'individual',
        'orderSm' : 'individual',
        'ARD'     : True
    },
]


''' Assemble R, C, D. '''
D = [(R_gdsc,    M_gdsc,    'Cell_lines', alpha_l[0]), 
     (R_ctrp,    M_ctrp,    'Cell_lines', alpha_l[1]), 
     (R_ccle_ic, M_ccle_ic, 'Cell_lines', alpha_l[2]),
     (R_ccle_ec, M_ccle_ec, 'Cell_lines', alpha_l[3])]
R, C = [], []

main_dataset = 'D'
index_main = 0 # GDSC
file_performance = 'results.txt'


''' Run the cross-validation framework for each negativity setting '''
for settings in values_settings:
    print settings
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
        file_performance=file_performance,
        append=True,
    )
    crossval.run(iterations=iterations,burn_in=burn_in,thinning=thinning)
    