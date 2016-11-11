'''
Plot a heatmap of the four drug sensitivity datasets.
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from HMF.methylation.load_methylation import filter_driver_genes_std
from HMF.methylation.load_methylation import load_driver_genes_std_tumour_labels_reordered

import matplotlib.pyplot as plt
import numpy


''' Method for creating and saving heatmap. '''
def plot_heatmap(R, rows, columns, outfile, size=(20, 15)):
    # Set up plot
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(R, cmap=plt.cm.bwr, interpolation='nearest', vmin=-2, vmax=2)
    
    # Axes labels
    ax.set_xlabel("Samples", fontsize=15)
    ax.xaxis.set_label_position('bottom') 
    ax.set_ylabel("Genes", fontsize=15)
    ax.yaxis.set_label_position('right') 
    
    # Turn off the frame
    ax.set_frame_on(False)
    
    # Put the major ticks at the middle of each cell
    ax.set_yticks(numpy.arange(R.shape[0]) + 0.5, minor=False)
    ax.set_xticks(numpy.arange(R.shape[1]) + 0.5, minor=False)
    
    # More table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    # Show drug and cell line names
    ax.set_xticklabels(columns, minor=False, fontsize = 4)
    ax.set_yticklabels(rows, minor=False, fontsize = 4)

    # Rotate x-labels 90 degrees    
    plt.xticks(rotation=90)
    
    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
        
    # Store plot
    fig.savefig(outfile, dpi=300)


if __name__ == '__main__':
    ''' Load the datasets. Reorder genes and samples:
        - Samples are grouped by healthy / tumour.
        - Genes..?
    '''
    (R_ge, R_pm, R_gm, labels_tumour, genes, samples) = load_driver_genes_std_tumour_labels_reordered()
    M_ge, M_pm, M_gm = numpy.ones(R_ge.shape), numpy.ones(R_pm.shape), numpy.ones(R_gm.shape)
    
    ''' Plot heatmaps and save them. '''
    folder_plots = project_location+"HMF/methylation/bicluster_analysis/plots/"
    
    plot_heatmap(R_ge, genes, samples, folder_plots+'heatmap_ge.png')
    plot_heatmap(R_pm, genes, samples, folder_plots+'heatmap_pm.png')
    plot_heatmap(R_gm, genes, samples, folder_plots+'heatmap_gm.png')
