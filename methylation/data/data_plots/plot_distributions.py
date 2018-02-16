'''
Pretty plot the row-normalised datasets.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from HMF.methylation.load_methylation import filter_driver_genes_std

import numpy
import matplotlib.pyplot as plt

''' Method for plotting the distributions '''
def plot_distribution(matrix, plot_location, binsize=0.2, dpi=600):
    values = matrix[~numpy.isnan(matrix)]
    fig = plt.figure(figsize=(2, 1.8))
    fig.subplots_adjust(left=0.065, right=0.935, bottom=0.11, top=0.99)   
    plt.hist(values,bins=numpy.arange(-5, 5.01, binsize))
    plt.xlim(-5, 5.0)
    plt.xticks(fontsize=8)
    plt.yticks([],fontsize=8)
    plt.show()
    plt.savefig(plot_location, dpi=dpi, bbox_inches='tight')

''' Load the data '''
(R_ge_std, R_pm_std, R_gm_std, genes_std, samples_std) = filter_driver_genes_std()

''' Plot the data '''
plot_distribution(R_ge_std, "pretty_ge_std")
plot_distribution(R_pm_std, "pretty_pm_std")
plot_distribution(R_gm_std, "pretty_gm_std")
