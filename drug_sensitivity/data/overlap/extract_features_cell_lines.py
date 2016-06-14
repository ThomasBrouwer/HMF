"""
Extract the features for the cell lines from gdsc_en_input_w5.csv.
First row gives the cell line names, first column gives the feature names.
Next 13321 rows are gene expression values, 426 after are copy number profiles, 
final 82 are cancer gene mutations.
"""

import numpy

''' Extract the features '''
filename = "features_cell_lines/gdsc_en_input_w5.csv"
lines = [line.split("\n")[0].split(",") for line in open(filename,'r').readlines()]

cell_lines_features_full = lines[0]
gene_expression = numpy.array([line[1:] for line in lines[1:1+13321]],dtype=float).T
cnv = numpy.array([line[1:] for line in lines[1+13321:1+13321+426]],dtype=float).T
mutation = numpy.array([line[1:] for line in lines[1+13321+426:]],dtype=float).T

''' Normalise the cell line names '''
def lowercase(values):
    return [v.lower() for v in values]
def remove(values,char):
    return [v.replace(char,"") for v in values]
def make_lowercase_remove_dashes_spaces_dots_commas(values):
    return remove(remove(remove(remove(lowercase(values),"-")," "),"."),",")
cell_lines_features = make_lowercase_remove_dashes_spaces_dots_commas(cell_lines_features_full)

''' Look up the cell lines we are interested in and extract the features - use the normalised names '''
cell_lines_data = list(numpy.loadtxt("features_cell_lines/cell_lines.txt",dtype=str))
drugs_data = list(numpy.loadtxt("features_drugs/drugs.txt",dtype=str))
cell_lines_selected, gene_expression_selected, cnv_selected, mutation_selected = [], [], [], []
for cl in cell_lines_data:
    if cl in cell_lines_features:
        cell_lines_selected.append(cl)
        index = cell_lines_features.index(cl)
        gene_expression_selected.append(gene_expression[index])
        cnv_selected.append(cnv[index])
        mutation_selected.append(mutation[index])
        
gene_expression_selected = numpy.array(gene_expression_selected)
cnv_selected = numpy.array(cnv_selected)
mutation_selected = numpy.array(mutation_selected)

''' Filter any features with the same value across all cell lines '''
indices_gene_expression_zero_std = [i for i,std in enumerate(gene_expression_selected.std(axis=0)) if std == 0.]
indices_cnv_zero_std = [i for i,std in enumerate(cnv_selected.std(axis=0)) if std == 0.]
indices_mutation_zero_std = [i for i,std in enumerate(mutation_selected.std(axis=0)) if std == 0.]

gene_expression_selected = numpy.delete(gene_expression_selected,indices_gene_expression_zero_std,axis=1)
cnv_selected = numpy.delete(cnv_selected,indices_cnv_zero_std,axis=1)
mutation_selected = numpy.delete(mutation_selected,indices_mutation_zero_std,axis=1)

''' Store the features for the cell lines we have features for '''
numpy.savetxt('features_cell_lines/cell_lines_features.txt',cell_lines_selected,delimiter="\t",fmt="%s")
numpy.savetxt('features_cell_lines/gene_expression.txt',gene_expression_selected,delimiter="\t")
numpy.savetxt('features_cell_lines/cnv.txt',cnv_selected,delimiter="\t")
numpy.savetxt('features_cell_lines/mutation.txt',mutation_selected,delimiter="\t")

''' Also store a standardised version (zero mean unit variance) of the gene expression and CNV '''
gene_expression_std_cols = gene_expression_selected.std(axis=0)
gene_expression_mean_cols = gene_expression_selected.mean(axis=0)
gene_expression_std = (gene_expression_selected - gene_expression_mean_cols) / gene_expression_std_cols

cnv_std_cols = cnv_selected.std(axis=0)
cnv_mean_cols = cnv_selected.mean(axis=0)
cnv_std = (cnv_selected - cnv_mean_cols) / cnv_std_cols

numpy.savetxt('features_cell_lines/gene_expression_std.txt',gene_expression_std,delimiter="\t")
numpy.savetxt('features_cell_lines/cnv_std.txt',cnv_std,delimiter="\t")

''' Also store the IC50 and EC50 datasets with only those cell lines '''
ccle_ic50 = numpy.loadtxt("data_all/ccle_ic50.txt",delimiter="\t")
ccle_ec50 = numpy.loadtxt("data_all/ccle_ec50.txt",delimiter="\t")
ctrp_ec50 = numpy.loadtxt("data_all/ctrp_ec50.txt",delimiter="\t")
gdsc_ic50 = numpy.loadtxt("data_all/gdsc_ic50.txt",delimiter="\t")

indices_cell_lines = [i for i,cell_line in enumerate(cell_lines_data) if cell_line in cell_lines_selected]
ccle_ic50_selected = ccle_ic50[indices_cell_lines,:]
ccle_ec50_selected = ccle_ec50[indices_cell_lines,:]
ctrp_ec50_selected = ctrp_ec50[indices_cell_lines,:]
gdsc_ic50_selected = gdsc_ic50[indices_cell_lines,:]

numpy.savetxt('data_features/ccle_ic50_features.txt',ccle_ic50_selected,delimiter="\t")
numpy.savetxt('data_features/ccle_ec50_features.txt',ccle_ec50_selected,delimiter="\t")
numpy.savetxt('data_features/ctrp_ec50_features.txt',ctrp_ec50_selected,delimiter="\t")
numpy.savetxt('data_features/gdsc_ic50_features.txt',gdsc_ic50_selected,delimiter="\t")

''' Print some statistics about sparsity of the datasets '''
no_drugs, no_cell_lines = len(drugs_data), len(cell_lines_selected)
no_entries = float(no_drugs*no_cell_lines)
print "Number of drugs: %s. Number cell lines: %s." % (no_drugs,no_cell_lines)

no_observed_ccle_ic50 = numpy.count_nonzero(~numpy.isnan(ccle_ic50_selected))
no_observed_ccle_ec50 = numpy.count_nonzero(~numpy.isnan(ccle_ec50_selected))
no_observed_ctrp_ec50 = numpy.count_nonzero(~numpy.isnan(ctrp_ec50_selected))
no_observed_gdsc_ic50 = numpy.count_nonzero(~numpy.isnan(gdsc_ic50_selected))
no_observed_overall = numpy.count_nonzero(logical_or(~numpy.isnan(ccle_ic50_selected), \
                                                     logical_or(~numpy.isnan(ccle_ec50_selected), \
                                                                logical_or(~numpy.isnan(ctrp_ec50_selected),~numpy.isnan(gdsc_ic50_selected))))) 

print "Number of observed entries CCLE IC50 / EC50, CTRP EC50, GDSC IC50: %s / %s, %s, %s." \
    % (no_observed_ccle_ic50, no_observed_ccle_ec50, no_observed_ctrp_ec50, no_observed_gdsc_ic50)
print "Fraction observed entries CCLE IC50 / EC50, CTRP EC50, GDSC IC50: %s / %s, %s, %s." \
    % (no_observed_ccle_ic50/no_entries, no_observed_ccle_ec50/no_entries, no_observed_ctrp_ec50/no_entries, no_observed_gdsc_ic50/no_entries)
print "Number observed overall: %s. Fraction observed entries overall: %s." % (no_observed_overall, no_observed_overall/no_entries)

''' Print the overlap of observed entries between the three datasets '''
# Overlaps two datasets
no_observed_ccle_ic50_ccle_ec50 = numpy.count_nonzero(logical_and(~numpy.isnan(ccle_ic50_selected),~numpy.isnan(ccle_ec50_selected)))
no_observed_ccle_ic50_ctrp_ec50 = numpy.count_nonzero(logical_and(~numpy.isnan(ccle_ic50_selected),~numpy.isnan(ctrp_ec50_selected)))
no_observed_ccle_ic50_gdsc_ic50 = numpy.count_nonzero(logical_and(~numpy.isnan(ccle_ic50_selected),~numpy.isnan(gdsc_ic50_selected)))
no_observed_ccle_ec50_ctrp_ec50 = numpy.count_nonzero(logical_and(~numpy.isnan(ccle_ec50_selected),~numpy.isnan(ctrp_ec50_selected)))
no_observed_ccle_ec50_gdsc_ic50 = numpy.count_nonzero(logical_and(~numpy.isnan(ccle_ec50_selected),~numpy.isnan(gdsc_ic50_selected)))
no_observed_ctrp_ec50_gdsc_ic50 = numpy.count_nonzero(logical_and(~numpy.isnan(ctrp_ec50_selected),~numpy.isnan(gdsc_ic50_selected)))

# Overlaps three datasets
no_observed_ccle_ic50_ccle_ec50_ctrp_ec50 = numpy.count_nonzero(logical_and(~numpy.isnan(ccle_ic50_selected),
                                                                logical_and(~numpy.isnan(ccle_ec50_selected),~numpy.isnan(ctrp_ec50_selected))))
no_observed_ccle_ic50_ccle_ec50_gdsc_ic50 = numpy.count_nonzero(logical_and(~numpy.isnan(ccle_ic50_selected),
                                                                logical_and(~numpy.isnan(ccle_ec50_selected),~numpy.isnan(gdsc_ic50_selected))))
no_observed_ccle_ic50_ctrp_ec50_gdsc_ic50 = numpy.count_nonzero(logical_and(~numpy.isnan(ccle_ic50_selected),
                                                                logical_and(~numpy.isnan(ctrp_ec50_selected),~numpy.isnan(gdsc_ic50_selected))))
no_observed_ccle_ec50_ctrp_ec50_gdsc_ic50 = numpy.count_nonzero(logical_and(~numpy.isnan(ccle_ec50_selected),
                                                                logical_and(~numpy.isnan(ctrp_ec50_selected),~numpy.isnan(gdsc_ic50_selected))))

# Overlap four datasets
no_observed_ccle_ic50_ccle_ec50_ctrp_ec50_gdsc_ic50 = numpy.count_nonzero(logical_and(~numpy.isnan(ccle_ec50_selected),logical_and(
                                                                                      ~numpy.isnan(ccle_ec50_selected),logical_and(
                                                                                      ~numpy.isnan(ctrp_ec50_selected),~numpy.isnan(gdsc_ic50_selected)))))

# Print overlaps
print "CCLE IC50 and CCLE EC50 = %s. CCLE IC50 and CTRP EC50 = %s. CCLE IC50 and GDSC IC50 = %s. CCLE EC50 and CTRP EC50 = %s. CCLE EC50 and GDSC IC50 = %s. CTRP EC50 and GDSC IC50 = %s." % \
    (no_observed_ccle_ic50_ccle_ec50/no_entries, no_observed_ccle_ic50_ctrp_ec50/no_entries, no_observed_ccle_ic50_gdsc_ic50/no_entries,
     no_observed_ccle_ec50_ctrp_ec50/no_entries, no_observed_ccle_ec50_gdsc_ic50/no_entries, no_observed_ctrp_ec50_gdsc_ic50/no_entries)
print "CCLE IC50 and CCLE EC50 and CRTP EC50 = %s. CCLE IC50 and CCLE EC50 and GDSC IC50 = %s. CCLE IC50 and CRTP EC50 and GDSC IC50 = %s. CCLE EC50 and CRTP EC50 and GDSC IC50 = %s." % \
    (no_observed_ccle_ic50_ccle_ec50_ctrp_ec50/no_entries, no_observed_ccle_ic50_ccle_ec50_gdsc_ic50/no_entries, 
     no_observed_ccle_ic50_ctrp_ec50_gdsc_ic50/no_entries, no_observed_ccle_ec50_ctrp_ec50_gdsc_ic50/no_entries)
print "CCLE IC50 and CCLE EC50 and CTRP EC50 and GDSC IC50 = %s." % (no_observed_ccle_ic50_ccle_ec50_ctrp_ec50_gdsc_ic50/no_entries)

'''
CCLE IC50 and CCLE EC50 = 0.107480239059. CCLE IC50 and CTRP EC50 = 0.10049161365. CCLE IC50 and GDSC IC50 = 0.0862251783304. CCLE EC50 and CTRP EC50 = 0.0619818777714. CCLE EC50 and GDSC IC50 = 0.0554270291112. CTRP EC50 and GDSC IC50 = 0.482263350684.
CCLE IC50 and CCLE EC50 and CRTP EC50 = 0.0576923076923. CCLE IC50 and CCLE EC50 and GDSC IC50 = 0.050655484866. CCLE IC50 and CRTP EC50 and GDSC IC50 = 0.0293522267206. CCLE EC50 and CRTP EC50 and GDSC IC50 = 0.0178812415655.
CCLE IC50 and CCLE EC50 and CTRP EC50 and GDSC IC50 = 0.0178812415655.
'''