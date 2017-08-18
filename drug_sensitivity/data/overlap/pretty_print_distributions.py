"""
Make nice plots of the row-wise normalised data distributions.
"""

import numpy, matplotlib.pyplot as plt

''' Method for plotting the distributions '''
def plot_distribution(matrix, plot_location, binsize=0.1, dpi=600):
    values = matrix[~numpy.isnan(matrix)]
    fig = plt.figure(figsize=(2, 1.8))
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.11, top=0.99)   
    plt.hist(values,bins=numpy.arange(0., 1.+binsize, binsize))
    plt.xlim(0,1)
    plt.xticks(fontsize=8)
    plt.yticks([],fontsize=8)
    plt.show()
    plt.savefig(folder_plots+plot_location, dpi=dpi, bbox_inches='tight')

''' Load in the data '''
folder_in, folder_plots = "data_features/", "data_plots/"
folder_capped, folder_01_rows, folder_01_cols = "data_capped/", "data_row_01/", "data_column_01/"

ccle_ic = numpy.loadtxt(folder_01_rows+"ccle_ic50_row_01.txt", delimiter="\t")
ccle_ec = numpy.loadtxt(folder_01_rows+"ccle_ec50_row_01.txt", delimiter="\t")
ctrp_ec = numpy.loadtxt(folder_01_rows+"ctrp_ec50_row_01.txt", delimiter="\t")
gdsc_ic = numpy.loadtxt(folder_01_rows+"gdsc_ic50_row_01.txt", delimiter="\t")

plot_distribution(ccle_ic, "pretty_ccle_ic50_row01")
plot_distribution(ccle_ec, "pretty_ccle_ec50_row01")
plot_distribution(ctrp_ec, "pretty_ctrp_ec50_row01")
plot_distribution(gdsc_ic, "pretty_gdsc_ic50_row01")
