"""
Measure the convergence of the drug sensitivity datasets for HMF (datasets,
using MTF). We run ten repeats and average across them.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

from HMF.code.models.hmf_Gibbs import HMF_Gibbs
from HMF.drug_sensitivity.load_dataset import load_data_without_empty, load_data_filter

import numpy


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
iterations = 500
n_repeats = 10
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
settings = {
    'priorF'  : 'exponential',
    'priorG'  : 'normal',
    'priorSn' : 'normal',
    'priorSm' : 'normal',
    'orderF'  : 'columns',
    'orderG'  : 'rows',
    'orderSn' : 'rows',
    'orderSm' : 'rows',
    'ARD'     : True
}

alpha_n = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC
K = {'Cell_lines':10, 'Drugs':10}

values_init = [
    # All expectation
    {
        'F'       : 'exp',
        'Sn'      : 'exp',
        'Sm'      : 'exp',
        'G'       : 'exp',
        'lambdat' : 'exp',
        'tau'     : 'exp'
    },
    # All random
    {
        'F'       : 'random',
        'Sn'      : 'random',
        'Sm'      : 'random',
        'G'       : 'random',
        'lambdat' : 'exp',
        'tau'     : 'random'
    },
    # Kmeans and expectation
    {
        'F'       : 'kmeans',
        'Sn'      : 'exp',
        'Sm'      : 'exp',
        'G'       : 'exp',
        'lambdat' : 'exp',
        'tau'     : 'exp'
    },
    # Kmeans and random
    {
        'F'       : 'kmeans',
        'Sn'      : 'random',
        'Sm'      : 'random',
        'G'       : 'random',
        'lambdat' : 'exp',
        'tau'     : 'exp'
    },
    # Exp and least squares
    {
        'F'       : 'exp',
        'Sn'      : 'least',
        'Sm'      : 'least',
        'G'       : 'least',
        'lambdat' : 'exp',
        'tau'     : 'exp'
    },
    # Random and least squares
    {
        'F'       : 'random',
        'Sn'      : 'least',
        'Sm'      : 'least',
        'G'       : 'least',
        'lambdat' : 'exp',
        'tau'     : 'exp'
    },
    # Kmeans and least squares
    {
        'F'       : 'kmeans',
        'Sn'      : 'least',
        'Sm'      : 'least',
        'G'       : 'least',
        'lambdat' : 'exp',
        'tau'     : 'exp'
    },
]


D, C = [], []
R = [(R_gdsc,    M_gdsc,    'Cell_lines', 'Drugs', alpha_n[0]), 
     (R_ctrp,    M_ctrp,    'Cell_lines', 'Drugs', alpha_n[1]), 
     (R_ccle_ic, M_ccle_ic, 'Cell_lines', 'Drugs', alpha_n[2]),
     (R_ccle_ec, M_ccle_ec, 'Cell_lines', 'Drugs', alpha_n[3])]
index_main = 1 # CTRP
file_performances = './results/hmf_data_mtf_ctrp_ec.txt'
    


''' Run the methods with different inits and measure the convergence. '''
all_init_performances = []
for init in values_init:
    all_performances = []
    for n in range(n_repeats):
        print "Repeat %s of initialisation experiment, with init %s." % (n, init)
        
        HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
        HMF.initialise(init)
        HMF.run(iterations)
        all_performances.append(HMF.all_performances_Rn['MSE'][index_main])
        
    average_performances = list(numpy.mean(all_performances, axis=0))
    all_init_performances.append(average_performances)
        
    
''' Store performances in file. '''
with open(file_performances,'w') as fout:
    fout.write("%s" % all_init_performances)