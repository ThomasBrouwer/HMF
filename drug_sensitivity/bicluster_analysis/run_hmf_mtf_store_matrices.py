"""
Run the HMF D-MTF method on the drug sensitivity datasets, and store the factor
matrices F, Sn - both the thinned out draws, and the expectation.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.code.models.hmf_Gibbs import HMF_Gibbs
from HMF.drug_sensitivity.load_dataset import load_data, load_names

import numpy


''' Load datasets '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data =                 location+"data_row_01/"
location_features_drugs =       location+"features_drugs/"
location_features_cell_lines =  location+"features_cell_lines/"
location_kernels =              location+"kernels_features/"

cell_lines, drugs = load_names()
R_gdsc,     M_gdsc    = load_data(location_data+"gdsc_ic50_row_01.txt")
R_ccle_ec,  M_ccle_ec = load_data(location_data+"ccle_ec50_row_01.txt")
R_ctrp,     M_ctrp    = load_data(location_data+"ctrp_ec50_row_01.txt")
R_ccle_ic,  M_ccle_ic = load_data(location_data+"ccle_ic50_row_01.txt")


''' Settings HMF '''
iterations, burn_in, thinning = 1000, 800, 10
indices_thinning = range(burn_in,iterations,thinning)  

hyperparameters = {
    'alphatau' : 1.,
    'betatau'  : 1.,
    'alpha0'   : 0.001,
    'beta0'    : 0.001,
    'alphaS'   : 0.001,
    'betaS'    : 0.001,
    'lambdaF'  : 0.1,
    'lambdaG'  : 0.1,
    'lambdaSn' : 0.1,
    'lambdaSm' : 0.1,
}
settings = {
    'priorF'           : 'exponential',
    'priorG'           : 'normal',
    'priorSn'          : 'normal',
    'priorSm'          : 'normal',
    'orderF'           : 'columns',
    'orderG'           : 'rows',
    'orderSn'          : 'rows',
    'orderSm'          : 'rows',
    'ARD'              : True,
    'element_sparsity' : True,
}
init = {
    'F'       : 'random',
    'Sn'      : 'least',
    'Sm'      : 'least',
    'G'       : 'least',
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

alpha_n = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC
alpha_m = []
K = {'Cell_lines':5, 'Drugs':5}

C, D = [], []
R = [(R_gdsc,    M_gdsc,    'Cell_lines', 'Drugs', alpha_n[0]), 
     (R_ctrp,    M_ctrp,    'Cell_lines', 'Drugs', alpha_n[1]), 
     (R_ccle_ic, M_ccle_ic, 'Cell_lines', 'Drugs', alpha_n[2]),
     (R_ccle_ec, M_ccle_ec, 'Cell_lines', 'Drugs', alpha_n[3])]


''' Run the model. '''
HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
HMF.initialise(init)
HMF.run(iterations)
    
"""
''' Extract all factor matrices (F, S^n), and store in files - only store the burned-in and thinned out draws. '''
E_drugs, E_cell_lines = 'Drugs', 'Cell_lines'
n_gdsc, n_ctrp, n_ccle_ic, n_ccle_ec = 0, 1, 2, 3
folder = project_location+'HMF/drug_sensitivity/bicluster_analysis/matrices/'

thinned_F_drugs = numpy.array(HMF.iterations_all_Ft[E_drugs])[indices_thinning]
thinned_F_cell_lines = numpy.array(HMF.iterations_all_Ft[E_cell_lines])[indices_thinning]
thinned_S_gdsc = numpy.array(HMF.iterations_all_Sn[n_gdsc])[indices_thinning]
thinned_S_ctrp = numpy.array(HMF.iterations_all_Sn[n_ctrp])[indices_thinning]
thinned_S_ccle_ic = numpy.array(HMF.iterations_all_Sn[n_ccle_ic])[indices_thinning]
thinned_S_ccle_ec = numpy.array(HMF.iterations_all_Sn[n_ccle_ec])[indices_thinning]

numpy.savetxt(fname=folder+'thinned_F_drugs', X=thinned_F_drugs)
numpy.savetxt(fname=folder+'thinned_F_cell_lines', X=thinned_F_cell_lines)
numpy.savetxt(fname=folder+'thinned_S_gdsc', X=thinned_S_gdsc)
numpy.savetxt(fname=folder+'thinned_S_ctrp', X=thinned_S_ctrp)
numpy.savetxt(fname=folder+'thinned_S_ccle_ic', X=thinned_S_ccle_ic)
numpy.savetxt(fname=folder+'thinned_S_ccle_ec', X=thinned_S_ccle_ec)
"""

''' Store the mean of the matrices. '''
E_drugs, E_cell_lines = 'Drugs', 'Cell_lines'
n_gdsc, n_ctrp, n_ccle_ic, n_ccle_ec = 0, 1, 2, 3
folder = project_location+'HMF/drug_sensitivity/bicluster_analysis/matrices/'

exp_F_drugs = HMF.approx_expectation_Ft(E=E_drugs, burn_in=burn_in, thinning=thinning)
exp_F_cell_lines = HMF.approx_expectation_Ft(E=E_cell_lines, burn_in=burn_in, thinning=thinning)
exp_S_gdsc = HMF.approx_expectation_Sn(n=n_gdsc, burn_in=burn_in, thinning=thinning)
exp_S_ctrp = HMF.approx_expectation_Sn(n=n_ctrp, burn_in=burn_in, thinning=thinning)
exp_S_ccle_ic = HMF.approx_expectation_Sn(n=n_ccle_ic, burn_in=burn_in, thinning=thinning)
exp_S_ccle_ec = HMF.approx_expectation_Sn(n=n_ccle_ec, burn_in=burn_in, thinning=thinning)

numpy.savetxt(fname=folder+'F_drugs', X=exp_F_drugs)
numpy.savetxt(fname=folder+'F_cell_lines', X=exp_F_cell_lines)
numpy.savetxt(fname=folder+'S_gdsc', X=exp_S_gdsc)
numpy.savetxt(fname=folder+'S_ctrp', X=exp_S_ctrp)
numpy.savetxt(fname=folder+'S_ccle_ic', X=exp_S_ccle_ic)
numpy.savetxt(fname=folder+'S_ccle_ec', X=exp_S_ccle_ec)