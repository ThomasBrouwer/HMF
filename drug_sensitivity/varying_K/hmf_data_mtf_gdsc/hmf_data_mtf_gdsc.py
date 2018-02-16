"""
Run the cross validation for HMF (datasets, using MTF) on the drug sensitivity
datasets, where we vary the values for K.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
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

'''
perf_hmf_mtf_gdsc_ARD = {'R^2': [0.5087871046603342, 0.5405982951199403, 0.5652606999681304, 0.5778745352237907, 0.5841729768645889, 0.5973615852417401, 0.6014992078098289, 0.6071197087358132, 0.6088170003967341, 0.6099720127546693, 0.6107244132760659, 0.6101133521196453, 0.6062368475380094, 0.6030565777712709, 0.6075537720283692, 0.5921403839514018, 0.5786920017302524], 'MSE': [0.097880852148347935, 0.091581028077929935, 0.086649408529322486, 0.084123504489505632, 0.08292821255114842, 0.080240212610094536, 0.079378349411783566, 0.078319419448748456, 0.077979735413061579, 0.077722877667606285, 0.077560494615294323, 0.077677077066387684, 0.078495354932438505, 0.079098908232854565, 0.078247431685996668, 0.081315327001849652, 0.08393351100768523], 'Rp': [0.71343352870722698, 0.73566640734330024, 0.7522106707838242, 0.76055351020782291, 0.76468840716568309, 0.77353682263301038, 0.77639316548043169, 0.78014484167757081, 0.78123252238557783, 0.78192009582992472, 0.78295303586020326, 0.78290172828340754, 0.78089948492215089, 0.77982497886089797, 0.78250326212486931, 0.77476445436477592, 0.76809996564793226]}
perf_hmf_mtf_gdsc_no_ARD = {'R^2': [0.5084157621428307, 0.5403638252030959, 0.5651194119339944, 0.5774662109212993, 0.5794649408627909, 0.5868785787961086, 0.5883374762968328, 0.5952588776588243, 0.5935662920709083, 0.5934768619981077, 0.5892755921509203, 0.5819142550599491, 0.5774149932937953, 0.566217514944143, 0.5478738958508538, 0.5329529878963819, 0.47753873671748004], 'MSE': [0.097999354426903898, 0.091577996918977403, 0.086673318951935155, 0.084246832655848564, 0.083840197160398186, 0.082384375620752737, 0.082022896390710268, 0.08065393903753966, 0.081017452247734198, 0.080933849063874991, 0.081882681554151279, 0.083330401984145902, 0.084223525383876063, 0.086468356936769478, 0.090109863647932262, 0.093120959444953272, 0.10408610144845129], 'Rp': [0.71330422066813837, 0.73578479067725422, 0.75239492148233345, 0.76051158538330754, 0.76183125514560479, 0.76666803526575611, 0.76801692224259244, 0.77229090455971183, 0.77153349809404448, 0.77196012324580665, 0.76999048018581107, 0.76685931821413322, 0.76555242646280541, 0.75965877272033344, 0.75177149948928645, 0.74602871524520997, 0.72407130438754963]}
'''