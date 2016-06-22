"""
Test the performance of HMF for recovering the CTRP dataset, where we vary the 
fraction of entries that are missing.
We repeat this 10 times per fraction and average that.

GDSC has 0.7356934001670844 observed entries
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from HMF.code.models.hmf_Gibbs import HMF_Gibbs
from HMF.code.generate_mask.mask import try_generate_M_from_M
from HMF.drug_sensitivity.load_dataset import load_data_without_empty,load_data_filter

import numpy, random


''' Settings '''
metrics = ['MSE', 'R^2', 'Rp']

fractions_unknown = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
repeats = 10

iterations, burn_in, thinning = 200, 180, 2
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


''' Load data '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data = location+"data_row_01/"

R_ctrp,     M_ctrp,   cell_lines, drugs   = load_data_without_empty(location_data+"ctrp_ec50_row_01.txt")
R_ccle_ec,  M_ccle_ec                     = load_data_filter(location_data+"ccle_ec50_row_01.txt",cell_lines,drugs)
R_gdsc,     M_gdsc                        = load_data_filter(location_data+"gdsc_ic50_row_01.txt",cell_lines,drugs)
R_ccle_ic,  M_ccle_ic                     = load_data_filter(location_data+"ccle_ic50_row_01.txt",cell_lines,drugs)


''' Seed all of the methods the same '''
numpy.random.seed(0)
random.seed(0)

''' Generate matrices M - one list of (M_train,M_test)'s for each fraction '''
M_attempts = 1000
all_Ms_train_test = [ 
    [try_generate_M_from_M(M=M_ctrp,fraction=fraction,attempts=M_attempts) for r in range(0,repeats)]
    for fraction in fractions_unknown
]

''' Make sure each M has no empty rows or columns '''
def check_empty_rows_columns(M,fraction):
    sums_columns = M.sum(axis=0)
    sums_rows = M.sum(axis=1)
    for i,c in enumerate(sums_rows):
        assert c != 0, "Fully unobserved row in M, row %s. Fraction %s." % (i,fraction)
    for j,c in enumerate(sums_columns):
        assert c != 0, "Fully unobserved column in M, column %s. Fraction %s." % (j,fraction)
        
for Ms_train_test,fraction in zip(all_Ms_train_test,fractions_unknown):
    for (M_train,M_test) in Ms_train_test:
        check_empty_rows_columns(M_train,fraction)

''' Run the method on each of the M's for each fraction '''
all_performances = {metric:[] for metric in metrics} 
average_performances = {metric:[] for metric in metrics} # averaged over repeats
for (fraction,Ms_train_test) in zip(fractions_unknown,all_Ms_train_test):
    print "Trying fraction %s." % fraction
    
    # Run the algorithm <repeats> times and store all the performances
    for metric in metrics:
        all_performances[metric].append([])
    for repeat,(M_train,M_test) in zip(range(0,repeats),Ms_train_test):
        print "Repeat %s of fraction %s." % (repeat+1, fraction)
     
        R = [(R_ctrp,    M_train,   'Cell_lines', 'Drugs', alpha_n[1]),
             (R_gdsc,    M_gdsc,    'Cell_lines', 'Drugs', alpha_n[0]),  
             (R_ccle_ic, M_ccle_ic, 'Cell_lines', 'Drugs', alpha_n[2]),
             (R_ccle_ec, M_ccle_ec, 'Cell_lines', 'Drugs', alpha_n[3])]
        C, D = [], []

        HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
        HMF.initialise(init)
        HMF.run(iterations)
        
        # Measure the performances
        performances = HMF.predict_Rn(n=0,M_pred=M_test,burn_in=burn_in,thinning=thinning)
        for metric in metrics:
            # Add this metric's performance to the list of <repeat> performances for this fraction
            all_performances[metric][-1].append(performances[metric])
            
    # Compute the average across attempts
    for metric in metrics:
        average_performances[metric].append(sum(all_performances[metric][-1])/repeats)
    
 
print "repeats=%s \nfractions_unknown = %s \nall_performances = %s \naverage_performances = %s" % \
    (repeats,fractions_unknown,all_performances,average_performances)

'''

'''