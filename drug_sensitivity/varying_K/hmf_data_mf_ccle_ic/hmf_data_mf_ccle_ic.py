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
    'ARD'     : True#False#
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
index_main = 2 # CCLE IC
file_performance = 'results_ARD_rerun.txt'


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
print "perf_hmf_mf_ccle_ic = %s" % performances

'''
perf_hmf_mf_ccle_ic_ARD = {'R^2': [0.5910922382221295, 0.6531908838142564, 0.6629382033090828, 0.6797758870849914, 0.6817821366180813, 0.6843181437032306, 0.6762967198598966, 0.6661954618598349, 0.6773871404090466, 0.6512353353949608, 0.6504416024816939, 0.6394313984979657, 0.5865500355709877, 0.584331170403394, 0.5516416926137764, 0.6343167857989174, 0.6317481418362226], 'MSE': [0.071035912401119519, 0.060296826294145754, 0.058798734466451563, 0.055765316137568063, 0.055397070568064088, 0.054833788673537845, 0.056087323930680166, 0.058004186083619014, 0.055993851091518686, 0.060733798706769392, 0.06060710501145411, 0.062501913340019158, 0.071943528884849797, 0.072254370174732283, 0.078114498077536051, 0.063622753521215963, 0.064012712501347227], 'Rp': [0.77071051660718881, 0.80881157457886688, 0.81542469131322337, 0.82576887459029957, 0.8270272573255959, 0.82959606579522338, 0.82469378030229712, 0.81957530654148203, 0.8261845040836816, 0.81238094284337414, 0.81260144670517231, 0.80648610575371349, 0.78074642009213968, 0.77814678578549568, 0.76313668279044755, 0.80223619792802514, 0.80042254096762799]}
perf_hmf_mf_ccle_ic_no_ARD = {'R^2': [0.5918603429896213, 0.6530251694597508, 0.6531941799605078, 0.675733229733389, 0.6840381447618619, 0.6839886405866593, 0.6806215115260226, 0.6745988359141192, 0.6633073685548363, 0.6637351303385091, 0.673653494439065, 0.651861728470893, 0.6391279706220506, 0.6463859342301752, 0.6383641072487218, 0.5785437210469763, 0.5675628364959946], 'MSE': [0.07095956654633287, 0.060417544928583568, 0.060511519428802764, 0.056432659637867442, 0.054872522084523992, 0.054883332399356867, 0.055583435963792907, 0.056703127415110965, 0.05850708302254122, 0.058687850499123728, 0.056610334326547683, 0.060291121255975702, 0.06263250506739873, 0.061481660633877756, 0.063059457049180576, 0.073350890085052356, 0.075069711955655843], 'Rp': [0.7715131559709103, 0.80868261200269509, 0.80962777739240754, 0.82365390152977225, 0.82883050981581774, 0.82952956785649568, 0.8271245645798585, 0.82414274865365567, 0.81804523363338255, 0.81944640838011595, 0.82421757506261173, 0.81287332778952592, 0.80739760668110294, 0.81077506101767849, 0.80710936901109065, 0.7776903673704807, 0.77406066850604682]}
'''