'''
Load in the three datasets (GE, PM, GM), and compute the best reordering of
columns and rows. 
We return a list of the new order of genes and samples.

We do this based on the following criteria:

Genes
-> We compute the similarity (Euclidean distance) between genes, using GE.
-> We then compute a dendrogram of the gene similarities, and reorder the genes
   based on that.
   
Samples
-> Firstly we divide the genes into healthy vs tumour
-> Then for both groups compute a dendrogram as with the genes.
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from HMF.code.kernels.gaussian_kernel import GaussianKernel
from HMF.methylation.load_methylation import filter_driver_genes_std
from HMF.methylation.load_methylation import load_tumor_label_list
from HMF.methylation.load_methylation import reorder_samples_label
from HMF.methylation.bicluster_analysis.plot_datasets import plot_heatmap

import matplotlib.pyplot as plt
import numpy
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram


''' Load the datasets. '''
(R_ge, R_pm, R_gm, genes, samples) = filter_driver_genes_std()
labels_tumour = load_tumor_label_list()
n_genes, n_samples = len(genes), len(samples)


''' Reorder dataset for tumour labels. '''
R_ge, sorted_samples, sorted_labels_tumour = reorder_samples_label(R_ge, samples)
R_pm, _, _ = reorder_samples_label(R_pm, samples)
R_gm, _, _ = reorder_samples_label(R_gm, samples)


''' Method for computing dendrogram. Return order of indices. '''
def compute_dendrogram(R):
    #plt.figure()
    # Hierarchical clustering methods: 
    # single (Nearest Point), complete (Von Hees), average (UPGMA), weighted (WPGMA), centroid (UPGMC), median (WPGMC), ward (incremental)
    Y = linkage(y=R, method='centroid', metric='euclidean') 
    Z = dendrogram(Z=Y, orientation='top', no_plot=True)#False)
    reordered_indices = Z['leaves']
    return reordered_indices


''' Compute the dendrogram for genes, using the gene expression data, and reorder data. '''
reordered_indices_genes = compute_dendrogram(R_ge)

R_ge = R_ge[reordered_indices_genes,:]
R_pm = R_pm[reordered_indices_genes,:]
R_gm = R_gm[reordered_indices_genes,:]
genes_reordered = [genes[i] for i in reordered_indices_genes]


''' Compute the dendrogram for samples, using the gene expression data.
    Do two dendrograms: one for tumour samples, one for healthy samples.
'''
indices_healthy = [i for i in range(0,n_samples) if sorted_labels_tumour[i] == 0]
indices_tumour =  [i for i in range(0,n_samples) if sorted_labels_tumour[i] == 1]

# Split data into healthy and tumour
R_healthy, R_tumour = R_ge[:,indices_healthy], R_ge[:,indices_tumour]
samples_healthy = [sorted_samples[i] for i in indices_healthy]
samples_tumour =  [sorted_samples[i] for i in indices_tumour]

# Compute dendrograms and reorderings
reordered_indices_samples_healthy = compute_dendrogram(R_healthy.T)
reordered_indices_samples_tumour =  compute_dendrogram(R_tumour.T)

# Reorder datasets and append them again
R_ge_healthy, R_ge_tumour = R_ge[:,reordered_indices_samples_healthy], R_ge[:,reordered_indices_samples_tumour]
R_pm_healthy, R_pm_tumour = R_pm[:,reordered_indices_samples_healthy], R_pm[:,reordered_indices_samples_tumour]
R_gm_healthy, R_gm_tumour = R_gm[:,reordered_indices_samples_healthy], R_gm[:,reordered_indices_samples_tumour]

R_ge = numpy.append(R_ge_healthy, R_ge_tumour, axis=1)
R_pm = numpy.append(R_pm_healthy, R_pm_tumour, axis=1)
R_gm = numpy.append(R_gm_healthy, R_gm_tumour, axis=1)

samples_reordered_healthy = [samples_healthy[i] for i in reordered_indices_samples_healthy]
samples_reordered_tumour =  [samples_tumour[i]  for i in reordered_indices_samples_tumour]
samples_reordered = samples_reordered_healthy + samples_reordered_tumour

labels_tumour_reordered = [0 for i in range(0,len(indices_healthy))] + [1 for i in range(0,len(indices_tumour))]


''' Plot the reordered heatmap. '''
folder_plots = project_location+"HMF/methylation/bicluster_analysis/plots/"

plot_heatmap(R_ge, genes_reordered, samples_reordered, folder_plots+'heatmap_ge_reordered.png')
plot_heatmap(R_pm, genes_reordered, samples_reordered, folder_plots+'heatmap_pm_reordered.png')
plot_heatmap(R_gm, genes_reordered, samples_reordered, folder_plots+'heatmap_gm_reordered.png')


''' Also store a reordered version of the datasets. '''
folder_data = project_location+"HMF/methylation/bicluster_analysis/data_reordered/"
numpy.savetxt(folder_data+"R_ge_reordered", R_ge)
numpy.savetxt(folder_data+"R_pm_reordered", R_pm)
numpy.savetxt(folder_data+"R_gm_reordered", R_gm)
numpy.savetxt(folder_data+"labels_tumour_reordered", labels_tumour_reordered)
numpy.savetxt(folder_data+"genes_reordered",   genes_reordered,   fmt="%s")
numpy.savetxt(folder_data+"samples_reordered", samples_reordered, fmt="%s")
