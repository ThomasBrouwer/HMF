"""
Run the HMF model on the gene expression and methylation data.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.code.models.hmf_Gibbs import HMF_Gibbs
from HMF.methylation.load_methylation import load_all_top_n_genes, load_ge_pm_top_n_genes

import numpy, matplotlib.pyplot as plt

##########

''' Model settings '''
settings = {
    'priorF'  : 'exponential',
    'priorSn' : 'normal',
    'orderF'  : 'columns',
    'orderSn' : 'rows',
    'ARD'     : True
}
hyperparameters = {
    'alphatau' : 1.,
    'betatau'  : 1.,
    'alpha0'   : 0.001,
    'beta0'    : 0.001,
    'lambdaF'  : 0.1,
    'lambdaSn' : 0.1,
}
init = {
    'F'       : 'kmeans',
    'Sn'      : 'least',
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

iterations, burn_in, thinning = 100, 80, 2
no_genes = 100 #13966

E = ['genes','samples']
I = {'genes':no_genes, 'samples':254}
K = {'genes':20, 'samples':20}
alpha_n = [1., 1.] # GE, PM


''' Load in data '''
(R_ge_n, R_pm_n, genes, samples) = load_ge_pm_top_n_genes(no_genes)
M_ge = numpy.ones((I['genes'],I['samples']))
M_pm = numpy.ones((I['genes'],I['samples']))

R = [
    (R_ge_n, M_ge, 'genes', 'samples', alpha_n[0]),
    (R_pm_n, M_pm, 'genes', 'samples', alpha_n[1])
]
C, D = [], []


''' Give the same random initialisation '''
numpy.random.seed(3)


''' Run the Gibbs sampler '''
HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
HMF.initialise(init)
HMF.run(iterations)


'''Plot tau against iterations to see that it converges '''
iterations_all_taun = HMF.iterations_all_taun

f, axarr = plt.subplots(2, sharex=True)
x = range(1,len(iterations_all_taun[0])+1)
axarr[0].set_title('Convergence of values')
axarr[0].plot(x, iterations_all_taun[0])    
axarr[0].set_ylabel("tau GE")
axarr[1].plot(x, iterations_all_taun[1]) 
axarr[1].set_ylabel("tau PM")
axarr[1].set_xlabel("Iterations")

# Extract the performances across all iterations
print "all_performances_Rn = %s" % HMF.all_performances_Rn
print "hmf_data = %s" % HMF.all_performances_Rn['MSE'][0]

'''

'''

plt.figure()
plt.plot(HMF.all_performances_Rn['MSE'][0],label='R1')
plt.plot(HMF.all_performances_Rn['MSE'][1],label='R2')
plt.legend()