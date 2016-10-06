"""
Run the cross validation for HMF (datasets, using MF) on the drug sensitivity
datasets, where we vary the values for K.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
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
index_main = 1 # CTRP
file_performance = 'results.txt'


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
print "perf_hmf_mf_ctrp = %s" % performances

'''
perf_hmf_mf_ctrp = {'R^2': [0.39601265774478034, 0.3994055427150653, 0.41697298985793835, 0.4199332641133404, 0.4239125135383608, 0.43345064722477183, 0.43064198375497487, 0.42885498067133854, 0.4209372112517296, 0.42530373766106067, 0.41837792304243965, 0.40854229150490384, 0.40295856898275983, 0.40917061887450784, 0.39386098442445266, 0.3948215300394192, 0.3748921724032922], 'MSE': [0.096003862679150481, 0.095432948334725182, 0.092692831976889628, 0.092210195716046509, 0.091545077328923988, 0.090122527387984, 0.090565933834273485, 0.090824809084094599, 0.091922310874882493, 0.091358687126241706, 0.092413073259747916, 0.094040671687050173, 0.094853731005173275, 0.093920726863974333, 0.096309353785351298, 0.09622223088188489, 0.099335159256396355], 'Rp': [0.62973458340270771, 0.63221332140062014, 0.64634352844001275, 0.64881982892103818, 0.65196915239762465, 0.65889801585245733, 0.65732073920634593, 0.6568800547670961, 0.65164347852607407, 0.65456382799755986, 0.65069182017929816, 0.64505029140092218, 0.64262544342424077, 0.64705869778389791, 0.63863724277750444, 0.6398273837756413, 0.62573156714127065]}
'''