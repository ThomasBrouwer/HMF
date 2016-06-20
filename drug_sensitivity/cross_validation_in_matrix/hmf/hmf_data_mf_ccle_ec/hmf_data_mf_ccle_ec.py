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
alpha_l = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC


''' Assemble R, C, D. '''
D = [(R_gdsc,    M_gdsc,    'Cell_lines', alpha_l[0]), 
     (R_ctrp,    M_ctrp,    'Cell_lines', alpha_l[1]), 
     (R_ccle_ic, M_ccle_ic, 'Cell_lines', alpha_l[2]),
     (R_ccle_ec, M_ccle_ec, 'Cell_lines', alpha_l[3])]
R, C = [], []

main_dataset = 'D'
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
alpha_l = [1., 1., 1., 1.]

Performance fold 1: {'R^2': 0.3768608765518632, 'MSE': 0.082255731296434917, 'Rp': 0.61733572719189134}.
Performance fold 2: {'R^2': 0.2712100093272868, 'MSE': 0.11551619590502188, 'Rp': 0.52939483192313197}.
Performance fold 3: {'R^2': 0.2599974832157974, 'MSE': 0.11926927612035797, 'Rp': 0.52300607507411689}.
Performance fold 4: {'R^2': 0.20544862781458317, 'MSE': 0.12933821598466069, 'Rp': 0.47968854154252411}.
Performance fold 5: {'R^2': 0.3184423325905976, 'MSE': 0.092622746532818934, 'Rp': 0.58480304698748364}.
Performance fold 6: {'R^2': 0.32518935590612286, 'MSE': 0.10395868478052631, 'Rp': 0.57420480887375003}.
Performance fold 7: {'R^2': 0.39419706312559977, 'MSE': 0.096211147043088488, 'Rp': 0.62895302617619397}.
Performance fold 8: {'R^2': 0.2902504212217545, 'MSE': 0.11566619495188876, 'Rp': 0.55558700651611392}.
Performance fold 9: {'R^2': 0.31875496598885233, 'MSE': 0.10995476938827239, 'Rp': 0.568908629988639}.
Performance fold 10: {'R^2': 0.2805543536104159, 'MSE': 0.097485334377508212, 'Rp': 0.54452342365438833}.
Average performance: {'R^2': 0.30409054893528736, 'MSE': 0.10622782963805788, 'Rp': 0.56064051179282326}.
"""