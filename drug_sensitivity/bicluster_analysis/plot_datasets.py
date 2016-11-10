'''
Plot a heatmap of the four drug sensitivity datasets.
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.drug_sensitivity.load_dataset import load_data
from HMF.drug_sensitivity.load_dataset import load_data_without_empty
from HMF.drug_sensitivity.load_dataset import load_names

import matplotlib.pyplot as plt
import numpy


''' Load datasets '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data =                 location+"data_row_01/"
location_features_drugs =       location+"features_drugs/"
location_features_cell_lines =  location+"features_cell_lines/"
location_kernels =              location+"kernels_features/"

file_gdsc =    location_data+"gdsc_ic50_row_01.txt"
file_ctrp =    location_data+"ctrp_ec50_row_01.txt"
file_ccle_ic = location_data+"ccle_ic50_row_01.txt"
file_ccle_ec = location_data+"ccle_ec50_row_01.txt"

cell_lines, drugs = load_names()

''' Datasets containing all drugs and cell lines. '''
R_gdsc,     M_gdsc    = load_data(file_gdsc)
R_ccle_ec,  M_ccle_ec = load_data(file_ctrp)
R_ctrp,     M_ctrp    = load_data(file_ccle_ic)
R_ccle_ic,  M_ccle_ic = load_data(file_ccle_ec)

''' Datasets containing only drugs and cell lines with observed entries. '''
R_gdsc_filtered,    M_gdsc_filtered,    i_cl_gdsc,    i_drugs_gdsc =    load_data_without_empty(file_gdsc)
R_ctrp_filtered,    M_ctrp_filtered,    i_cl_ctrp,    i_drugs_ctrp =    load_data_without_empty(file_ctrp)
R_ccle_ic_filtered, M_ccle_ic_filtered, i_cl_ccle_ic, i_drugs_ccle_ic = load_data_without_empty(file_ccle_ic)
R_ccle_ec_filtered, M_ccle_ec_filtered, i_cl_ccle_ec, i_drugs_ccle_ec = load_data_without_empty(file_ccle_ec)

cell_lines_gdsc_filtered,    drugs_gdsc_filtered =    numpy.array(cell_lines)[i_cl_gdsc],    numpy.array(drugs)[i_drugs_gdsc]
cell_lines_ctrp_filtered,    drugs_ctrp_filtered =    numpy.array(cell_lines)[i_cl_ctrp],    numpy.array(drugs)[i_drugs_ctrp]
cell_lines_ccle_ic_filtered, drugs_ccle_ic_filtered = numpy.array(cell_lines)[i_cl_ccle_ic], numpy.array(drugs)[i_drugs_ccle_ic]
cell_lines_ccle_ec_filtered, drugs_ccle_ec_filtered = numpy.array(cell_lines)[i_cl_ccle_ec], numpy.array(drugs)[i_drugs_ccle_ec]

''' Method for creating and saving heatmap. '''
def plot_heatmap(R, rows, columns, outfile, size=(60, 10)):
    # Set up plot
    fig, ax = plt.subplots(figsize=(30, 8))
    heatmap = ax.pcolor(R, cmap=plt.cm.Reds, alpha=0.8)
    
    # Format
    fig = plt.gcf()
    fig.set_size_inches(size)
    
    # Turn off the frame
    ax.set_frame_on(False)
    
    # Put the major ticks at the middle of each cell
    ax.set_yticks(numpy.arange(R.shape[0]) + 0.5, minor=False)
    ax.set_xticks(numpy.arange(R.shape[1]) + 0.5, minor=False)
    
    # More table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    # Show drug and cell line names
    ax.set_xticklabels(columns, minor=False, fontsize = 8)
    ax.set_yticklabels(rows, minor=False, fontsize = 8)

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
    fig.savefig(outfile, dpi=50)
    
''' Plot heatmaps and save them. '''
folder_plots = project_location+"HMF/drug_sensitivity/bicluster_analysis/plots/"

plot_heatmap(R_gdsc.T, drugs,    cell_lines, folder_plots+'full/heatmap_gdsc_full.png',    size=(60, 10))
plot_heatmap(R_ctrp.T, drugs,    cell_lines, folder_plots+'full/heatmap_ctrp_full.png',    size=(60, 10))
plot_heatmap(R_ccle_ic.T, drugs, cell_lines, folder_plots+'full/heatmap_ccle_ic_full.png', size=(60, 10))
plot_heatmap(R_ccle_ec.T, drugs, cell_lines, folder_plots+'full/heatmap_ccle_ec_full.png', size=(60, 10))

plot_heatmap(R_gdsc_filtered.T,    drugs_gdsc_filtered,    cell_lines_gdsc_filtered,    
             folder_plots+'filtered/heatmap_gdsc_filtered.png',    size=(60, 10))
plot_heatmap(R_ctrp_filtered.T,    drugs_ctrp_filtered,    cell_lines_ctrp_filtered,    
             folder_plots+'filtered/heatmap_ctrp_filtered.png',    size=(60, 10))
plot_heatmap(R_ccle_ic_filtered.T, drugs_ccle_ic_filtered, cell_lines_ccle_ic_filtered, 
             folder_plots+'filtered/heatmap_ccle_ic_filtered.png', size=(60, 10))
plot_heatmap(R_ccle_ec_filtered.T, drugs_ccle_ec_filtered, cell_lines_ccle_ec_filtered, 
             folder_plots+'filtered/heatmap_ccle_ec_filtered.png', size=(60, 10))
