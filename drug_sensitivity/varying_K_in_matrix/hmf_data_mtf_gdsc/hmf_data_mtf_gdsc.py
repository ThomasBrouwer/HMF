"""
Run the cross validation for HMF (datasets, using MTF) on the drug sensitivity
datasets, where we vary the values for K.
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
    'ARD'     : False#True
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

alpha_n = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC
alpha_m = []
values_K = [
    {'Cell_lines':1,  'Drugs':1},
    {'Cell_lines':2,  'Drugs':2},
    {'Cell_lines':3,  'Drugs':3},
    {'Cell_lines':4,  'Drugs':4},
    {'Cell_lines':5,  'Drugs':5},
    {'Cell_lines':6,  'Drugs':6},
    {'Cell_lines':7,  'Drugs':7},
    {'Cell_lines':8,  'Drugs':8},
    {'Cell_lines':9,  'Drugs':9},
    {'Cell_lines':10, 'Drugs':10},
    {'Cell_lines':12, 'Drugs':12},
    {'Cell_lines':14, 'Drugs':14},
    {'Cell_lines':16, 'Drugs':16},
    {'Cell_lines':18, 'Drugs':18},
    {'Cell_lines':20, 'Drugs':20},
    {'Cell_lines':25, 'Drugs':25},
    {'Cell_lines':30, 'Drugs':30},
]


''' Assemble R, C, D. '''
R = [(R_gdsc,    M_gdsc,    'Cell_lines', 'Drugs', alpha_n[0]), 
     (R_ctrp,    M_ctrp,    'Cell_lines', 'Drugs', alpha_n[1]), 
     (R_ccle_ic, M_ccle_ic, 'Cell_lines', 'Drugs', alpha_n[2]),
     (R_ccle_ec, M_ccle_ec, 'Cell_lines', 'Drugs', alpha_n[3])]
C, D = [], []

main_dataset = 'R'
index_main = 0 # GDSC
file_performance = 'results_no_ARD.txt'


''' Run the cross-validation framework for each value of K '''
all_performances = []
for K in values_K:
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
    all_performances.append(crossval.average_performance)
    
''' Combine all average performances into a list for MSE, one for R^2, one for Rp. '''
measures = ['MSE', 'R^2', 'Rp']
performances = {measure: [] for measure in measures}
for performance in all_performances:
    for measure in measures:
        performances[measure].append(performance[measure])
print "perf_hmf_mtf_gdsc = %s" % performances