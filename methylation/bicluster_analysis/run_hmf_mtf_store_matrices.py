"""
Run the HMF D-MTF method on the methylation datasets, and store the expectation
of the factor matrices F, Sn.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.code.models.hmf_Gibbs import HMF_Gibbs
from HMF.methylation.load_methylation import filter_driver_genes_std

import numpy


''' Settings HMF '''
iterations, burn_in, thinning = 1000, 800, 10
indices_thinning = range(burn_in,iterations,thinning)  

settings = {
    'priorF'           : 'exponential',
    'priorSn'          : ['normal','normal','normal'],
    'orderF'           : 'columns',
    'orderSn'          : ['rows','rows','rows'],
    'ARD'              : True,
    'element_sparsity' : True,
}
hyperparameters = {
    'alphatau' : 1,
    'betatau'  : 1,
    'alpha0'   : 0.001,
    'beta0'    : 0.001,
    'alphaS'   : 0.001,
    'betaS'    : 0.001,
    'lambdaF'  : 0.1,
}
init = {
    'F'       : 'kmeans',
    'Sn'      : ['least','least','least'],
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

iterations, burn_in, thinning = 200, 150, 2

E = ['genes','samples']
K = {'genes':20, 'samples':20}
alpha_n = [1., 1., 1.] # GE, PM, GM


''' Load in data '''
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes_std()
M_ge, M_pm, M_gm = numpy.ones(R_ge.shape), numpy.ones(R_pm.shape), numpy.ones(R_gm.shape)

R = [
    (R_ge, M_ge, 'genes', 'samples', alpha_n[0]),
    (R_pm, M_pm, 'genes', 'samples', alpha_n[1]),
    (R_gm, M_gm, 'genes', 'samples', alpha_n[1]),
]
C, D = [], []


''' Run the Gibbs sampler '''
HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
HMF.initialise(init)
HMF.run(iterations)
    

''' Store the mean of the matrices. '''
folder = project_location+'HMF/methylation/bicluster_analysis/matrices/'

E_drugs, E_cell_lines = 'genes', 'samples'
n_ge, n_pm, n_gm = 0, 1, 2

exp_F_genes = HMF.approx_expectation_Ft(E=E_drugs, burn_in=burn_in, thinning=thinning)
exp_F_samples = HMF.approx_expectation_Ft(E=E_cell_lines, burn_in=burn_in, thinning=thinning)
exp_S_ge = HMF.approx_expectation_Sn(n=n_ge, burn_in=burn_in, thinning=thinning)
exp_S_pm = HMF.approx_expectation_Sn(n=n_pm, burn_in=burn_in, thinning=thinning)
exp_S_gm = HMF.approx_expectation_Sn(n=n_gm, burn_in=burn_in, thinning=thinning)

numpy.savetxt(fname=folder+'F_genes',   X=exp_F_genes)
numpy.savetxt(fname=folder+'F_samples', X=exp_F_samples)
numpy.savetxt(fname=folder+'S_ge',      X=exp_S_ge)
numpy.savetxt(fname=folder+'S_pm',      X=exp_S_pm)
numpy.savetxt(fname=folder+'S_gm',      X=exp_S_gm)
