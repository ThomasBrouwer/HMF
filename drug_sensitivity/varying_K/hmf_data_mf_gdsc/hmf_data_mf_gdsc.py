"""
Run the cross validation for HMF (datasets, using MF) on the drug sensitivity
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

alpha_l = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC
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
D = [(R_gdsc,    M_gdsc,    'Cell_lines', alpha_l[0]), 
     (R_ctrp,    M_ctrp,    'Cell_lines', alpha_l[1]), 
     (R_ccle_ic, M_ccle_ic, 'Cell_lines', alpha_l[2]),
     (R_ccle_ec, M_ccle_ec, 'Cell_lines', alpha_l[3])]
R, C = [], []

main_dataset = 'D'
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
print "perf_hmf_mf_gdsc = %s" % performances

'''
perf_hmf_mf_gdsc_ARD = {'R^2': [0.5378895897404576, 0.5588179784583696, 0.5775754357991849, 0.5873917418246475, 0.5967456327721383, 0.6095091882778506, 0.6074754829120068, 0.6050982525383638, 0.6098315847434679, 0.611131820078387, 0.6082729251853476, 0.6140249635786372, 0.6069252831431039, 0.6083592162220847, 0.6019826370374308, 0.5917142836519045, 0.5739435402252778], 'MSE': [0.092136703486547483, 0.087905642199881034, 0.084164693137332741, 0.082259050400168274, 0.080401058765981775, 0.077825310883253979, 0.078174668859023189, 0.078696435988965113, 0.077705288040566881, 0.077498733288756283, 0.07804560567351479, 0.076928521167593494, 0.078374012701426674, 0.0780458543346009, 0.079341708581703488, 0.081416782601892793, 0.084949076252284456], 'Rp': [0.73356781208514232, 0.74800716339225581, 0.76044620352630921, 0.76679009228605166, 0.77298798892714338, 0.78126235797111454, 0.78032086351937813, 0.77899753778874326, 0.78209406846316987, 0.78295799672380328, 0.78134473535901727, 0.78535239371595467, 0.78142139740958672, 0.78250240626119072, 0.779086954732916, 0.77393268283739303, 0.76480257894949644]}
perf_hmf_mf_gdsc_no_ARD = {'R^2': [0.5363060417340935, 0.5614975454009382, 0.5791607097910151, 0.5868235238467349, 0.5948259587638552, 0.6055236879704063, 0.602006153186475, 0.6038681652766151, 0.6038486571826207, 0.6029574355307981, 0.5934197217857531, 0.5948126746691593, 0.5935278744198079, 0.5790847651634309, 0.5790906673749082, 0.5594027022281074, 0.5242266045833859], 'MSE': [0.092428563410528339, 0.087388634628372958, 0.083799369125948139, 0.082339345177863094, 0.080699584354268897, 0.078621891227560053, 0.079340555083425413, 0.078957528972110003, 0.078975654330226089, 0.079119037550295784, 0.080999147730606277, 0.080781475508719183, 0.080999020442368175, 0.083870073938298512, 0.083870826059296771, 0.087821360529603515, 0.094849007932430535], 'Rp': [0.7326684385887976, 0.74973170381131715, 0.76165930546992955, 0.76674695090264211, 0.77239877140582169, 0.77918775821311015, 0.77734384558978054, 0.77888754337254507, 0.77910536600665758, 0.77898878506090585, 0.77418628571332992, 0.77616794075932372, 0.77630781168525664, 0.76830027371813125, 0.76975722287856208, 0.76114635861732816, 0.74501999039888989]}
'''