'''
Measure the performance of different hybrid combinations of factorisations of
the four drug sensitivity datasets, doing in-matrix predictions on the GDSC 
dataset.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/" # "/home/tab43/Documents/Projects/libraries/"
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

R_gdsc,     M_gdsc,    cell_lines, drugs = load_data_without_empty(location_data+"gdsc_ic50_row_01.txt")
R_ctrp,     M_ctrp                       = load_data_filter(location_data+"ctrp_ec50_row_01.txt",cell_lines,drugs)
R_ccle_ic,  M_ccle_ic                    = load_data_filter(location_data+"ccle_ic50_row_01.txt",cell_lines,drugs)
R_ccle_ec,  M_ccle_ec                    = load_data_filter(location_data+"ccle_ec50_row_01.txt",cell_lines,drugs)


''' Settings HMF '''
iterations, burn_in, thinning = 5, 2, 1#500, 400, 2
no_folds = 10 #10

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
}
init = {
    'F'       : 'random', #'kmeans' gives error
    'Sn'      : 'random',
    'Sm'      : 'random',
    'G'       : 'random',
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

alpha = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC
K = {'Cell_lines':10, 'Drugs':10}

file_performance = 'results_gdsc.txt'


''' The different factorisations, and a method to construct the matrices. '''
values_factorisation = [
    { 'GDSC': 'R', 'CTRP': 'R', 'CCLE_IC': 'R', 'CCLE_EC': 'R' },
    { 'GDSC': 'D', 'CTRP': 'R', 'CCLE_IC': 'R', 'CCLE_EC': 'R' },
    { 'GDSC': 'R', 'CTRP': 'D', 'CCLE_IC': 'R', 'CCLE_EC': 'R' },
    { 'GDSC': 'R', 'CTRP': 'R', 'CCLE_IC': 'D', 'CCLE_EC': 'R' },
    { 'GDSC': 'R', 'CTRP': 'R', 'CCLE_IC': 'R', 'CCLE_EC': 'D' },
    { 'GDSC': 'D', 'CTRP': 'D', 'CCLE_IC': 'R', 'CCLE_EC': 'R' },
    { 'GDSC': 'D', 'CTRP': 'R', 'CCLE_IC': 'D', 'CCLE_EC': 'R' },
    { 'GDSC': 'D', 'CTRP': 'R', 'CCLE_IC': 'R', 'CCLE_EC': 'D' },
    { 'GDSC': 'R', 'CTRP': 'D', 'CCLE_IC': 'D', 'CCLE_EC': 'R' },
    { 'GDSC': 'R', 'CTRP': 'D', 'CCLE_IC': 'R', 'CCLE_EC': 'D' },
    { 'GDSC': 'R', 'CTRP': 'R', 'CCLE_IC': 'D', 'CCLE_EC': 'D' },
    { 'GDSC': 'D', 'CTRP': 'D', 'CCLE_IC': 'D', 'CCLE_EC': 'D' },
]

def construct_RCD(factorisation, dataset_to_predict):
    ''' Given a dictionary {'X1':..,'X2':..,'Y':..}, all giving either 'R' or 
        'D', construct the (R,C,D) lists. For the Y entry, use M_Y as the mask. 
        Note that the Y entry always has index 0. 
        Return a tuple (index_main,R,C,D), where index_main gives the index of
        dataset_to_predict in either R or D. '''
    R, C, D = [], [], []
        
    if factorisation['CTRP'] == 'R':
        R.append((R_ctrp, M_ctrp, 'Cell_lines', 'Drugs', alpha[1]))
    else: # factorisation['CTRP'] == 'D'
        D.append((R_ctrp, M_ctrp, 'Cell_lines', alpha[1]))
        
    if factorisation['CCLE_IC'] == 'R':
        R.append((R_ccle_ic, M_ccle_ic, 'Cell_lines', 'Drugs', alpha[2]))
    else: # factorisation['CCLE_IC'] == 'D'
        D.append((R_ccle_ic, M_ccle_ic, 'Cell_lines', alpha[2]))
        
    if factorisation['CCLE_EC'] == 'R':
        R.append((R_ccle_ec, M_ccle_ec, 'Cell_lines', 'Drugs', alpha[3]))
    else: # factorisation['CCLE_EC'] == 'D'
        D.append((R_ccle_ec, M_ccle_ec, 'Cell_lines', alpha[3]))
    
    if factorisation['GDSC'] == 'R':
        R.append((R_gdsc, M_gdsc, 'Cell_lines', 'Drugs', alpha[0]))
    else: # factorisation['GDSC'] == 'D'
        D.append((R_gdsc, M_gdsc, 'Cell_lines', alpha[0]))
        
    index_main = len(R) - 1 if factorisation[dataset_to_predict] == 'R' else len(D) - 1
    return (index_main, R, C, D)


''' Run the methods with different factorisation combinations, and measure performances. '''
for factorisation in values_factorisation:
    main_dataset = factorisation['GDSC']    
    index_main, R, C, D = construct_RCD(factorisation=factorisation, dataset_to_predict='GDSC')
    
    log = open(file_performance,'a')
    log.write('Trying factorisation: %s. \n' % factorisation)
    log.close()
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
    