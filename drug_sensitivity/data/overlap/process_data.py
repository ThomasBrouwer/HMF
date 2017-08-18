"""
Preprocess the IC50 and EC50 values:
- Cap values too high/low
- Create [0,1] range per row or column datasets from the capped data
- Plot the distributions of values in these six datasets

Also tried doing log transform of data but:
- For CCLE IC50 and EC50 this doesn't improve distribution
- For CTRP EC50 and GDSC IC50 values close to 0 then go to -infinity
"""

import numpy, itertools, matplotlib.pyplot as plt

''' Method for plotting the distributions '''
def plot_distribution(matrix,plot_title,plot_location,bins=100,dpi=300):
    values = matrix[~numpy.isnan(matrix)]
    plt.figure()    
    plt.hist(values,bins=bins)
    plt.title(plot_title)
    plt.show()
    plt.savefig(folder_plots+plot_location,dpi=dpi)

''' Load in the data '''
folder_in, folder_plots = "data_features/", "data_plots/"
folder_capped, folder_01_rows, folder_01_cols = "data_capped/", "data_row_01/", "data_column_01/"

ccle_ic50 = numpy.loadtxt(folder_in+'ccle_ic50_features.txt',delimiter="\t")
ccle_ec50 = numpy.loadtxt(folder_in+'ccle_ec50_features.txt',delimiter="\t")
ctrp_ec50 = numpy.loadtxt(folder_in+'ctrp_ec50_features.txt',delimiter="\t")
gdsc_ic50 = numpy.loadtxt(folder_in+'gdsc_ic50_features.txt',delimiter="\t")

''' Plot the uncapped values '''
#plot_distribution(ccle_ic50,"CCLE IC50","ccle_ic50_uncapped")
#plot_distribution(ccle_ec50,"CCLE EC50","ccle_ec50_uncapped")
#plot_distribution(ctrp_ec50,"CTRP EC50","ctrp_ec50_uncapped")
#plot_distribution(gdsc_ic50,"GDSC IC50","gdsc_ic50_uncapped")

''' For GDSC IC50 we undo the log transform by taking the exponent, and plot it '''
gdsc_ic50_exp = numpy.exp(gdsc_ic50)
#plot_distribution(gdsc_ic50_exp,"GDSC IC50 exponential transform","gdsc_ic50_exp")
    
''' For GDSC IC50 and CTRP EC50 we need to cap the values '''
max_ctrp_ec50, max_gdsc_ic50 = 20, 20

ctrp_ec50_capped, gdsc_ic50_capped = numpy.copy(ctrp_ec50), numpy.copy(gdsc_ic50_exp)
ctrp_ec50_capped[ctrp_ec50_capped > max_ctrp_ec50] = max_ctrp_ec50
gdsc_ic50_capped[gdsc_ic50_capped > max_gdsc_ic50] = max_gdsc_ic50

#plot_distribution(ctrp_ec50_capped,"CTRP EC50 capped","ctrp_ec50_capped")
#plot_distribution(gdsc_ic50_capped,"GDSC IC50 capped","gdsc_ic50_capped")

''' Then create versions of the (capped) data where we map each row or column to [0,1] '''
def map_to_01(matrix):
    max_rows = numpy.nanmax(matrix,axis=1)[:,None]
    min_rows = numpy.nanmin(matrix,axis=1)[:,None]
    max_cols = numpy.nanmax(matrix,axis=0)
    min_cols = numpy.nanmin(matrix,axis=0)
    matrix_row01 = (matrix - min_rows) / (max_rows - min_rows)
    matrix_col01 = (matrix - min_cols) / (max_cols - min_cols)
    return matrix_row01, matrix_col01

#ccle_ic50_row_01, ccle_ic50_col_01 = map_to_01(ccle_ic50)
#ccle_ec50_row_01, ccle_ec50_col_01 = map_to_01(ccle_ec50)
#ctrp_ec50_row_01, ctrp_ec50_col_01 = map_to_01(ctrp_ec50_capped)
#gdsc_ic50_row_01, gdsc_ic50_col_01 = map_to_01(gdsc_ic50_capped)

#plot_distribution(ccle_ic50_row_01,"CCLE IC50 rows [0,1]","ccle_ic50_row01")
#plot_distribution(ccle_ec50_row_01,"CCLE EC50 rows [0,1]","ccle_ec50_row01")
#plot_distribution(ctrp_ec50_row_01,"CTRP EC50 rows [0,1]","ctrp_ec50_row01")
#plot_distribution(gdsc_ic50_row_01,"GDSC IC50 rows [0,1]","gdsc_ic50_row01")

#plot_distribution(ccle_ic50_col_01,"CCLE IC50 columns [0,1]","ccle_ic50_col01")
#plot_distribution(ccle_ec50_col_01,"CCLE EC50 columns [0,1]","ccle_ec50_col01")
#plot_distribution(ctrp_ec50_col_01,"CTRP EC50 columns [0,1]","ctrp_ec50_col01")
#plot_distribution(gdsc_ic50_col_01,"GDSC IC50 columns [0,1]","gdsc_ic50_col01")

''' Finally, store the capped and [0,1] range datasets '''
numpy.savetxt(folder_in+"gdsc_ic50_features_exp.txt",gdsc_ic50_exp,delimiter="\t")

numpy.savetxt(folder_capped+"ccle_ic50.txt",ccle_ic50,delimiter="\t")
numpy.savetxt(folder_capped+"ccle_ec50.txt",ccle_ec50,delimiter="\t")
numpy.savetxt(folder_capped+"ctrp_ec50_capped.txt",ctrp_ec50_capped,delimiter="\t")
numpy.savetxt(folder_capped+"gdsc_ic50_capped.txt",gdsc_ic50_capped,delimiter="\t")

numpy.savetxt(folder_01_rows+"ccle_ic50_row_01.txt",ccle_ic50_row_01,delimiter="\t")
numpy.savetxt(folder_01_rows+"ccle_ec50_row_01.txt",ccle_ec50_row_01,delimiter="\t")
numpy.savetxt(folder_01_rows+"ctrp_ec50_row_01.txt",ctrp_ec50_row_01,delimiter="\t")
numpy.savetxt(folder_01_rows+"gdsc_ic50_row_01.txt",gdsc_ic50_row_01,delimiter="\t")

numpy.savetxt(folder_01_cols+"ccle_ic50_col_01.txt",ccle_ic50_col_01,delimiter="\t")
numpy.savetxt(folder_01_cols+"ccle_ec50_col_01.txt",ccle_ec50_col_01,delimiter="\t")
numpy.savetxt(folder_01_cols+"ctrp_ec50_col_01.txt",ctrp_ec50_col_01,delimiter="\t")
numpy.savetxt(folder_01_cols+"gdsc_ic50_col_01.txt",gdsc_ic50_col_01,delimiter="\t")
