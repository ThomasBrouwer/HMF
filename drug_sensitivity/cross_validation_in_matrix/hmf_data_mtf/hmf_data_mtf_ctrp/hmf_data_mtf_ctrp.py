"""
Run the cross validation for HMF (datasets, using MTF) on the drug sensitivity
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
alpha_n = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC
alpha_m = []


''' Assemble R, C, D. '''
R = [(R_gdsc,    M_gdsc,    'Cell_lines', 'Drugs', alpha_n[0]), 
     (R_ctrp,    M_ctrp,    'Cell_lines', 'Drugs', alpha_n[1]), 
     (R_ccle_ic, M_ccle_ic, 'Cell_lines', 'Drugs', alpha_n[2]),
     (R_ccle_ec, M_ccle_ec, 'Cell_lines', 'Drugs', alpha_n[3])]
C, D = [], []

main_dataset = 'R'
index_main = 1 # CTRP
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

Performance fold 1: {'R^2': 0.3979923431790232, 'MSE': 0.093712431790840642, 'Rp': 0.63480368954443955}.
Performance fold 2: {'R^2': 0.4168485688081066, 'MSE': 0.093321578438946159, 'Rp': 0.64679126897044259}.
Performance fold 3: {'R^2': 0.4432611452201122, 'MSE': 0.08518589837377348, 'Rp': 0.66783193799996021}.
Performance fold 4: {'R^2': 0.43087893236243824, 'MSE': 0.093047482360858719, 'Rp': 0.65814074081686835}.
Performance fold 5: {'R^2': 0.44755548121486055, 'MSE': 0.087543832885246656, 'Rp': 0.66960423375237987}.
Performance fold 6: {'R^2': 0.40101189445140806, 'MSE': 0.094624613292419099, 'Rp': 0.63768543278847034}.
Performance fold 7: {'R^2': 0.4394017296458057, 'MSE': 0.088839544042411456, 'Rp': 0.66375351626018042}.
Performance fold 8: {'R^2': 0.43010451523248994, 'MSE': 0.091632718178601469, 'Rp': 0.65712723686620433}.
Performance fold 9: {'R^2': 0.4356781303605963, 'MSE': 0.091100412821542784, 'Rp': 0.66123722702073018}.
Performance fold 10: {'R^2': 0.44640798024741635, 'MSE': 0.088895006862976481, 'Rp': 0.66853814259730249}.
Average performance: {'R^2': 0.42891407207222565, 'MSE': 0.090790351904761674, 'Rp': 0.65655134266169779}.
"""