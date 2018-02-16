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
print "perf_hmf_mf_ccle_ec = %s" % performances

'''
perf_hmf_mf_ccle_ec_ARD = {'R^2': [0.25509182283203824, 0.27467220596281444, 0.28272765073797185, 0.28094958077098975, 0.30174518714127907, 0.3034467392428315, 0.3082742440419226, 0.3007785173717338, 0.28363089246112894, 0.28248729665977257, 0.29609316347695913, 0.3109570466965005, 0.29769154684482796, 0.26578517147055075, 0.27194882176697954, 0.16484352467687335, 0.09962742853469989], 'MSE': [0.11359247654578122, 0.11063934263594347, 0.1095591967468779, 0.1096023011459903, 0.10653084493403842, 0.10636807319637409, 0.10562032316624659, 0.10635686929594852, 0.10929225610121121, 0.10972399290107955, 0.10738318269230016, 0.10512259286671644, 0.10736657436285242, 0.11217905190606056, 0.11101883629643547, 0.12786265314410566, 0.13736070220627677], 'Rp': [0.50892634225894839, 0.52754537694652859, 0.53550311958875862, 0.53926371951085628, 0.55763465896803377, 0.5600068412305157, 0.56240224193314192, 0.55847065624565784, 0.54711894409704054, 0.54747329229558239, 0.55640705901053822, 0.56512937206073199, 0.55797870395089844, 0.54097320445109409, 0.54798363116243698, 0.48915320545845553, 0.4689156576197745]}
perf_hmf_mf_ccle_ec_no_ARD = {'R^2': [0.25771225398817404, 0.2667584745643762, 0.27847678119325375, 0.27110169078299806, 0.29060610701921846, 0.2871689388747002, 0.23192227075781924, 0.23897375252374733, 0.18087553775326953, 0.1792694497161687, 0.08746331133090707, 0.08413484696576205, -0.10291411676502552, -0.22526204672395247, -0.2839181659107333, -0.4336969312068434, -0.5248572467132993], 'MSE': [0.11332152927641445, 0.11189956819710183, 0.11018915353347676, 0.11098632678055587, 0.10832980881655549, 0.10893243302511371, 0.11706490245793269, 0.11590210542119303, 0.12399939415034593, 0.1253054086958639, 0.13941626462617154, 0.14005498990551821, 0.168868163638384, 0.18639280409045975, 0.19465634753961542, 0.21860006215651845, 0.23320807119250206], 'Rp': [0.51125560284470084, 0.52247655309807073, 0.53404815350954116, 0.53259481327561187, 0.55492749833019817, 0.5544241633412883, 0.52432464405917945, 0.53610280213258554, 0.50585152618219964, 0.49570628817364437, 0.4681265295334972, 0.46002881276532132, 0.41898813692180026, 0.38781711166649252, 0.38860630928821049, 0.3789506397454242, 0.36316974189041129]}
'''