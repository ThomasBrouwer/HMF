"""
Run the HMF model on the gene expression and methylation data.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.code.models.hmf_Gibbs import HMF_Gibbs
from HMF.methylation.load_methylation import filter_driver_genes_std

import numpy, matplotlib.pyplot as plt

##########

''' Model settings '''
settings = {
    'priorF'              : 'exponential',
    'priorSn'             : ['normal','normal','normal'], #GE,PM
    'orderF'              : 'columns',
    'orderSn'             : ['rows','rows','rows'],
    'ARD'                 : True,
    'importance_learning' : True,
}
hyperparameters = {
    'alphatau' : 1,
    'betatau'  : 1,
    'alpha0'   : 0.001,
    'beta0'    : 0.001,
    'alphaA'   : 1000.,
    'betaA'    : 1000.,
    'lambdaF'  : 0.1,
    'lambdaSn' : 0.1,
}
init = {
    'F'       : 'kmeans',
    'Sn'      : ['least','least','least'],
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

iterations, burn_in, thinning = 100, 80, 2

E = ['genes','samples']
K = {'genes':5, 'samples':5}
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


'''Plot tau and importance against iterations to see that it converges '''
iterations_all_taun = HMF.iterations_all_taun
iterations_all_alphan = HMF.iterations_all_alphan

f, axarr = plt.subplots(3, 2, sharex=True)
x = range(1,len(iterations_all_taun[0])+1)

axarr[0][0].set_title('Convergence of tau')
axarr[0][0].plot(x, iterations_all_taun[0])    
axarr[0][0].yaxis.set_label_position("right")   
axarr[0][0].set_ylabel("tau GE")
axarr[1][0].plot(x, iterations_all_taun[1]) 
axarr[1][0].yaxis.set_label_position("right")   
axarr[1][0].set_ylabel("tau PM")
axarr[2][0].plot(x, iterations_all_taun[2]) 
axarr[2][0].yaxis.set_label_position("right")   
axarr[2][0].set_ylabel("tau GM")
axarr[2][0].set_xlabel("Iterations")

axarr[0][1].set_title('Convergence of alpha')
axarr[0][1].plot(x, iterations_all_alphan[0])
axarr[0][1].yaxis.set_label_position("right")     
axarr[0][1].set_ylabel("alpha GE")
axarr[1][1].plot(x, iterations_all_alphan[1]) 
axarr[1][1].yaxis.set_label_position("right") 
axarr[1][1].set_ylabel("alpha PM")
axarr[2][1].plot(x, iterations_all_alphan[2]) 
axarr[2][1].yaxis.set_label_position("right") 
axarr[2][1].set_ylabel("alpha GM")
axarr[2][1].set_xlabel("Iterations")


''' Print the importance values found. '''
exp_alpha_ge = HMF.approx_expectation_alphan(n=0, burn_in=burn_in, thinning=thinning)
exp_alpha_pm = HMF.approx_expectation_alphan(n=1, burn_in=burn_in, thinning=thinning)
exp_alpha_gm = HMF.approx_expectation_alphan(n=2, burn_in=burn_in, thinning=thinning)

print "Importances. GE: %s. PM: %s. GM: %s." % (exp_alpha_ge,exp_alpha_pm,exp_alpha_gm)
