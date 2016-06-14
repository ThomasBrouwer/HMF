"""
Using the drugs and cell lines specified in /overlap/drugs.txt, cell_lines.txt,
construct the new datasets for CCLE, CTRP, GDSC using only those entries.
"""

import numpy

''' Load in the datasets, and list of drugs/cell lines '''
location_ccle, location_ctrp, location_gdsc = "./CCLE/processed_all/", "./CTRP/processed_all/", "./GDSC/processed_all/"

ccle_ic50 = numpy.loadtxt(location_ccle+"ic50.txt",dtype=str,delimiter="\t")
ccle_ec50 = numpy.loadtxt(location_ccle+"ec50.txt",dtype=str,delimiter="\t")
ctrp_ec50 = numpy.loadtxt(location_ctrp+"ec50.txt",dtype=str,delimiter="\t")
gdsc_ic50 = numpy.loadtxt(location_gdsc+"ic50.txt",dtype=str,delimiter="\t")
datasets = [ccle_ic50,ccle_ec50,ctrp_ec50,gdsc_ic50]

drugs_ccle = list(numpy.loadtxt(location_ccle+"drugs.txt",dtype=str,delimiter="\t")[:,0])
drugs_ctrp = list(numpy.loadtxt(location_ctrp+"drugs.txt",dtype=str,delimiter="\t")[:,0])
drugs_gdsc = list(numpy.loadtxt(location_gdsc+"drugs.txt",dtype=str,delimiter="\t")[:,0])

cell_lines_ccle = list(numpy.loadtxt(location_ccle+"cell_lines.txt",dtype=str,delimiter="\t")[:,0])
cell_lines_ctrp = list(numpy.loadtxt(location_ctrp+"cell_lines.txt",dtype=str,delimiter="\t")[:,0])
cell_lines_gdsc = list(numpy.loadtxt(location_gdsc+"cell_lines.txt",dtype=str,delimiter="\t")[:,0])

''' Load in the list of drugs and cell lines we are using '''
location_overlap = "./overlap/"
drugs = list(numpy.loadtxt(location_overlap+"features_drugs/drugs.txt",dtype=str))
cell_lines = list(numpy.loadtxt(location_overlap+"features_cell_lines/cell_lines.txt",dtype=str))

''' Initialise the new datasets '''
no_drugs, no_cell_lines = len(drugs), len(cell_lines)
new_datasets = [numpy.empty((no_cell_lines,no_drugs)) for dataset in datasets]
for dataset in new_datasets:
    dataset[:] = numpy.NaN
    
''' Then for each of the new datasets, extract the values '''
for i,cell_line in enumerate(cell_lines):
    for j,drug in enumerate(drugs):
        # Add CCLE values
        if cell_line in cell_lines_ccle and drug in drugs_ccle:
            i_ccle = cell_lines_ccle.index(cell_line)
            j_ccle = drugs_ccle.index(drug)
            new_datasets[0][i,j] = datasets[0][i_ccle,j_ccle]
            new_datasets[1][i,j] = datasets[1][i_ccle,j_ccle]
        else:
            new_datasets[0][i,j] = numpy.NaN
            
        # Add CTRP values
        if cell_line in cell_lines_ctrp and drug in drugs_ctrp:
            i_ctrp = cell_lines_ctrp.index(cell_line)
            j_ctrp = drugs_ctrp.index(drug)
            new_datasets[2][i,j] = datasets[2][i_ctrp,j_ctrp]
        else:
            new_datasets[2][i,j] = numpy.NaN
     
        # Add GDSC values
        if cell_line in cell_lines_gdsc and drug in drugs_gdsc:
            i_gdsc = cell_lines_gdsc.index(cell_line)
            j_gdsc = drugs_gdsc.index(drug)
            new_datasets[3][i,j] = datasets[3][i_gdsc,j_gdsc]
        else:
            new_datasets[3][i,j] = numpy.NaN
    
new_datasets = [numpy.array(dataset,dtype=float) for dataset in new_datasets]    
    
''' Store the new datasets '''
numpy.savetxt(location_overlap+"data_all/ccle_ic50.txt", new_datasets[0], delimiter="\t")
numpy.savetxt(location_overlap+"data_all/ccle_ec50.txt", new_datasets[1], delimiter="\t")
numpy.savetxt(location_overlap+"data_all/ctrp_ec50.txt", new_datasets[2], delimiter="\t")
numpy.savetxt(location_overlap+"data_all/gdsc_ic50.txt", new_datasets[3], delimiter="\t")
        
''' Print some statistics about sparsity of the datasets '''
no_drugs, no_cell_lines = len(drugs), len(cell_lines)
print "Number drugs: %s. Number cell lines: %s." % (no_drugs,no_cell_lines)

no_observed_ccle_ic50 = numpy.count_nonzero(~numpy.isnan(new_datasets[0]))
no_observed_ccle_ec50 = numpy.count_nonzero(~numpy.isnan(new_datasets[1]))
no_observed_ctrp_ec50 = numpy.count_nonzero(~numpy.isnan(new_datasets[2]))
no_observed_gdsc_ic50 = numpy.count_nonzero(~numpy.isnan(new_datasets[3]))
no_observed_overall = numpy.count_nonzero(logical_or(~numpy.isnan(new_datasets[0]), \
                                                     logical_or(~numpy.isnan(new_datasets[1]), \
                                                                logical_or(~numpy.isnan(new_datasets[2]),~numpy.isnan(new_datasets[3]))))) 

print "Number of observed entries CCLE IC50 / EC50, CTRP EC50, GDSC IC50: %s / %s, %s, %s." \
    % (no_observed_ccle_ic50, no_observed_ccle_ec50, no_observed_ctrp_ec50, no_observed_gdsc_ic50)
print "Fraction observed entries CCLE IC50 / EC50, CTRP EC50, GDSC IC50: %s / %s, %s, %s." \
    % (no_observed_ccle_ic50/float(no_drugs*no_cell_lines), no_observed_ccle_ec50/float(no_drugs*no_cell_lines), no_observed_ctrp_ec50/float(no_drugs*no_cell_lines), no_observed_gdsc_ic50/float(no_drugs*no_cell_lines))
print "Number observed overall: %s. Fraction observed entries overall: %s." % (no_observed_overall, no_observed_overall/float(no_drugs*no_cell_lines))