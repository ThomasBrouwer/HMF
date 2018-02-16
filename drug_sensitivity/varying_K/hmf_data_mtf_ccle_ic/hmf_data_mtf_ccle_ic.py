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
index_main = 2 # CCLE IC
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
print "perf_hmf_mtf_ccle_ic = %s" % performances

'''
perf_hmf_mtf_ccle_ic_ARD = {'R^2': [0.5637778089516734, 0.6229008287739488, 0.6459166259801155, 0.671001063028316, 0.6775223245124332, 0.6769281050910366, 0.685745034895577, 0.6910036735319031, 0.6694839518093924, 0.6769815503342866, 0.6756972749942373, 0.6681651933732523, 0.6507771956109533, 0.6214759646358363, 0.6086625531827311, 0.609331026295461, 0.5998979402312878], 'MSE': [0.075668159556016051, 0.065716273156219904, 0.061614113283628501, 0.057161702436326935, 0.056240449874023976, 0.055908668389967095, 0.054612399141030388, 0.05384105995762134, 0.057502333576918127, 0.056176597222810487, 0.056452199749285573, 0.057859381544366317, 0.06065747412056921, 0.065580171633972964, 0.06823222059373478, 0.067825831293541208, 0.069657805031371894], 'Rp': [0.75283516182071364, 0.79003622116100714, 0.80570880614354878, 0.82028510435435842, 0.8246625884728791, 0.82443035932769904, 0.82998817926748958, 0.83341589076298761, 0.82184402570861192, 0.82499482242939481, 0.82544429431384159, 0.82196315078744164, 0.81386472665021348, 0.79826751638699256, 0.79158984845298297, 0.79112916167406067, 0.78583155390421155]}
perf_hmf_mtf_ccle_ic_no_ARD = {'R^2': [0.5649123845153683, 0.6220918014405556, 0.6465493397052937, 0.6644637807262898, 0.6683309891889193, 0.6690493753950661, 0.6692245659238865, 0.674504714587984, 0.6674779453573751, 0.6641336463160181, 0.6567276890559317, 0.6341744192965353, 0.5926801430461524, 0.5802284419552813, 0.5674486510313731, 0.5174154287879993, 0.45935385138470225], 'MSE': [0.075526044520224322, 0.065592212450125101, 0.061289893938475018, 0.058524973461807225, 0.057726800228028022, 0.057695903459691701, 0.057245871259827316, 0.056670345042860795, 0.057626590084049426, 0.058285005123450674, 0.059763100478976802, 0.063676679241231895, 0.071011090049675291, 0.072801236873157815, 0.075278841815490494, 0.084076492874460479, 0.094114844928637781], 'Rp': [0.75345440489284243, 0.7903485977118323, 0.80523204817908778, 0.81673814324704819, 0.81891625774327148, 0.81992583671359243, 0.81995362231816737, 0.8238621132261349, 0.8198394718925085, 0.81923222304939358, 0.81605875138132533, 0.80466614767932731, 0.78686207637379846, 0.78507283894636093, 0.77807883650100784, 0.75473405331883514, 0.73534458094347988]}
'''