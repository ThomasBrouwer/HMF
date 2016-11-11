"""
Load the factor matrices, and plot heatmaps of the biggest biclusters.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from plot_datasets import plot_heatmap

import numpy


''' Load in factor matrices. '''
folder_matrices = project_location+'HMF/methylation/bicluster_analysis/matrices/'

F_genes = numpy.loadtxt(folder_matrices+'F_genes')
F_samples = numpy.loadtxt(folder_matrices+'F_samples')
S_ge = numpy.loadtxt(folder_matrices+'S_ge')
S_pm = numpy.loadtxt(folder_matrices+'S_pm')
S_gm = numpy.loadtxt(folder_matrices+'S_gm')

''' Also load in list of genes and samples. '''
folder_data = project_location+'HMF/methylation/bicluster_analysis/data_reordered/'

genes = numpy.loadtxt(folder_data+'genes_reordered', dtype='str')
samples = numpy.loadtxt(folder_data+'samples_reordered', dtype='str')

''' Extract the most interesting biclusters, and plot them. '''
folder_biclusters = project_location+'HMF/methylation/bicluster_analysis/plots_biclusters/'

biclusters = [ # list of S matrix, bicluster index (k,l), and dataset name
    (S_ge, (2,18), 'ge'),
    (S_ge, (1,18), 'ge'),
    (S_gm, (4,11), 'gm'),
    (S_pm, (4,19), 'pm'),
    (S_pm, (16,19), 'pm'),
]

for S, (k,l), name in biclusters:
    new_S = numpy.zeros(S.shape)
    new_S[k,l] = S[k,l]
    new_R = numpy.dot(F_genes, numpy.dot(new_S, F_samples.T))
    plot_heatmap(new_R, genes, samples, 
                 folder_biclusters+'bicluster_%s_%s_%s' % (name,k,l))