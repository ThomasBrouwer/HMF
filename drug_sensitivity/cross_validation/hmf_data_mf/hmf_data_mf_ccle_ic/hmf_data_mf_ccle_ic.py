"""
Run the cross validation for HMF (datasets, using MF) on the drug sensitivity
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
alpha_l = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC


''' Assemble R, C, D. '''
D = [(R_gdsc,    M_gdsc,    'Cell_lines', alpha_l[0]), 
     (R_ctrp,    M_ctrp,    'Cell_lines', alpha_l[1]), 
     (R_ccle_ic, M_ccle_ic, 'Cell_lines', alpha_l[2]),
     (R_ccle_ec, M_ccle_ec, 'Cell_lines', alpha_l[3])]
R, C = [], []

main_dataset = 'D'
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
alpha_l = [1., 1., 1., 1.]

Performance fold 1: {'R^2': 0.6270304796535086, 'MSE': 0.06939540300280933, 'Rp': 0.79943136133165671}.
Performance fold 2: {'R^2': 0.599174024624373, 'MSE': 0.068682764012461275, 'Rp': 0.78059317053180111}.
Performance fold 3: {'R^2': 0.7544979411752707, 'MSE': 0.0412598895744377, 'Rp': 0.87145586607471981}.
Performance fold 4: {'R^2': 0.6389341018388808, 'MSE': 0.056417128205211595, 'Rp': 0.80239565120038525}.
Performance fold 5: {'R^2': 0.7411586293305594, 'MSE': 0.047594856941135723, 'Rp': 0.86279488269062821}.
Performance fold 6: {'R^2': 0.5901730164039578, 'MSE': 0.071703909181544595, 'Rp': 0.77641815858088337}.
Performance fold 7: {'R^2': 0.6851413781201525, 'MSE': 0.050885206330769595, 'Rp': 0.8327971333830867}.
Performance fold 8: {'R^2': 0.5491401569834853, 'MSE': 0.072432214427659472, 'Rp': 0.74953273269244125}.
Performance fold 9: {'R^2': 0.6599887369258886, 'MSE': 0.059455755285712925, 'Rp': 0.81478633347820817}.
Performance fold 10: {'R^2': 0.7252492728125195, 'MSE': 0.053954162333230034, 'Rp': 0.85426404254547239}.
Average performance: {'R^2': 0.6570487737868597, 'MSE': 0.059178128929497233, 'Rp': 0.81444693325092832}.
"""