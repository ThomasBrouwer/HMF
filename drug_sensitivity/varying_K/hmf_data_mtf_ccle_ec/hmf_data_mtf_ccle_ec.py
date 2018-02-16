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
index_main = 3 # CCLE EC
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
print "perf_hmf_mtf_ccle_ec = %s" % performances

'''
perf_hmf_mtf_ccle_ec_ARD = {'R^2': [0.22130433351572662, 0.23436904393066277, 0.25349891921848744, 0.26197569125518816, 0.2695342307689645, 0.26672055317967447, 0.2695523170280055, 0.27947810132490886, 0.27374285234417195, 0.29692246728355265, 0.28685299652010743, 0.23076954830631666, 0.23895444545974942, 0.21803856442602637, 0.1932270062669479, 0.11728626954616406, -0.16273974309204747], 'MSE': [0.11877093472158848, 0.11651397899509874, 0.1142164623900664, 0.11277792208548303, 0.11156726251650757, 0.11204750847533337, 0.11081313894382958, 0.10995226389667438, 0.11092767452564808, 0.10675481780212559, 0.10875522168261728, 0.1171615963816293, 0.11630766413022293, 0.11917230518406814, 0.12340850150942455, 0.13462390362626769, 0.17593769413576021], 'Rp': [0.47477119689468861, 0.48984275355508877, 0.50713142942048095, 0.5185244223836083, 0.52547953155413762, 0.52820352499744094, 0.53372142402441558, 0.54179922363152044, 0.53991045058884624, 0.56115486279552296, 0.55281225076709706, 0.52115182920861958, 0.53106497540558872, 0.52041814802453978, 0.51711485509254196, 0.47940193100960266, 0.39336437720174977]}
perf_hmf_mtf_ccle_ec_no_ARD = {'R^2': [0.2212591099422283, 0.24245412075972955, 0.24138068681462718, 0.2601499181402914, 0.2556600025427874, 0.26326331348983417, 0.26305567538160946, 0.2660931910565833, 0.2512935853641924, 0.23141794068714222, 0.10593963630086436, -0.0694506879467203, -0.5035858204876258, -0.6684481052589197, -0.8584740263217391, -1.4822266454787023, -2.1288269646561555], 'MSE': [0.11870203382953415, 0.11568500926021244, 0.11543534321813162, 0.11313843188986002, 0.11290774265798406, 0.11261723936857553, 0.11258832313308295, 0.11239589688428135, 0.11424161449350427, 0.11688217593537782, 0.13582588022762682, 0.16359665984640484, 0.22777045166804011, 0.25393011149106648, 0.28389736206377941, 0.37494946042187738, 0.47950423649302404], 'Rp': [0.47751202602083309, 0.49679133242870088, 0.50238666644029029, 0.5141792105905697, 0.51905088865421556, 0.52263850457840055, 0.52363257393437546, 0.53105614019907121, 0.52778925036877566, 0.52291476601462461, 0.47953954258187459, 0.42678909945283705, 0.35206175805623413, 0.34712954436047522, 0.28874899420671041, 0.26146649628806784, 0.24387754937227563]}
'''