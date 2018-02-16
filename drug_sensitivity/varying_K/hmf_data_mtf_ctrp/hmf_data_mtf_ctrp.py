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
index_main = 1 # CTRP
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
print "perf_hmf_mtf_ctrp = %s" % performances

'''
perf_hmf_mtf_ctrp_ARD = {'R^2': [0.3114859267983666, 0.3996953050043409, 0.41421115646920786, 0.4210598859457206, 0.42466197565851516, 0.4257553930600973, 0.4314929401225439, 0.43233038499026766, 0.4396501443499569, 0.43687247130072493, 0.4259789764685179, 0.4319039970198606, 0.4154414540287063, 0.40434344892359275, 0.40006041213895305, 0.37444980124988164, 0.3612508071486233], 'MSE': [0.10945597652504177, 0.095392146845027337, 0.093093061239668845, 0.092055421894656111, 0.091459107314937116, 0.091241628111785211, 0.090339478793706393, 0.090252066324524954, 0.089099913007402123, 0.089561589635702141, 0.091268907779504388, 0.090289168857795601, 0.092943876108323512, 0.094658672695154739, 0.095376042382170095, 0.09943687402463354, 0.10154029147326064], 'Rp': [0.5584430276023955, 0.63253696195457998, 0.64436187091805108, 0.64932835755369234, 0.65241418160972109, 0.65325890023775246, 0.6581403457688888, 0.65831424165300523, 0.66399831011984722, 0.66197464052423616, 0.65504600109450717, 0.6604818034585962, 0.64962044273299757, 0.64422909747388779, 0.64170833633469948, 0.62921389264167549, 0.62373844398837408]}
perf_hmf_mtf_ctrp_no_ARD = {'R^2': [0.3110618798108806, 0.3988439003093299, 0.4136861051167612, 0.4135156673635027, 0.4217855415326481, 0.42310536538001353, 0.42591893083966426, 0.4227012761803909, 0.4186881962791363, 0.4140807750917661, 0.4159988246087997, 0.40229888452364443, 0.3946053894965488, 0.37008092487402183, 0.35195130089799964, 0.28571352875139333, 0.23319814636216182], 'MSE': [0.10948175534122757, 0.095531884754890162, 0.09314601732413294, 0.093255979053441365, 0.09194713848420713, 0.091724835147022066, 0.091242598430674809, 0.091814172059005814, 0.092362436763050768, 0.093189529609858199, 0.0928069539482866, 0.095026569405836175, 0.096256020756904939, 0.10016597577624213, 0.10300286077055469, 0.11356702749724021, 0.12196301892382377], 'Rp': [0.55829227471886234, 0.63224408610563754, 0.64402999233148572, 0.64340013723489464, 0.64970353796495406, 0.65113700287025522, 0.65314075481630884, 0.6508821021607073, 0.64869881692449938, 0.64524559808156379, 0.64823018488791417, 0.64006146967609612, 0.63736253021099798, 0.62394984464560355, 0.61526217721265353, 0.58572432883200576, 0.56868048632834411]}
'''