'''
Plot the distributions of the methylation and gene expression datasets, after
we select the cancer driver genes; both before and after preprocessing.
Also plot the similarity kernels.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from HMF.methylation.load_methylation import filter_driver_genes
from HMF.methylation.load_methylation import filter_driver_genes_std
from HMF.methylation.load_methylation import load_kernels

import numpy
import matplotlib.pyplot as plt

''' Method for plotting the distributions '''
def plot_distribution(matrix, plot_title, plot_location, bins=100, dpi=300):
    values = matrix[~numpy.isnan(matrix)]
    plt.figure()    
    plt.hist(values,bins=bins)
    plt.title(plot_title)
    plt.show()
    plt.savefig(plot_location,dpi=dpi)

''' Load the data '''
(R_ge, R_pm, R_gm, genes, samples) = filter_driver_genes()
(R_ge_std, R_pm_std, R_gm_std, genes_std, samples_std) = filter_driver_genes_std()
K_ge, K_pm, K_gm = load_kernels()

''' Plot the data '''
plot_distribution(R_ge,"Gene expression (unstandardised)","ge")
plot_distribution(R_pm,"Promoter region methylation (unstandardised)","pm")
plot_distribution(R_gm,"Gene body methylation (unstandardised)","gm")

plot_distribution(R_ge_std,"Gene expression (standardised)","ge_std")
plot_distribution(R_pm_std,"Promoter region methylation (standardised)","pm_std")
plot_distribution(R_gm_std,"Gene body methylation (standardised)","gm_std")

plot_distribution(K_ge,"Gene expression kernel","ge_kernel")
plot_distribution(K_pm,"Promoter region methylation kernel","pm_kernel")
plot_distribution(K_gm,"Gene body methylation kernel","gm_kernel")