"""
Process the raw CCLE drug sensitivity dataset in CCLE_NP24.2009_Drug_data_2015.02.24.csv.

Return:
- The matrix of IC50 values (in order of the drug and cell line lists). 
  Rows are cell lines, columns are drugs.
- The matrix of EC50 values (in order of the drug and cell line lists). 
  Rows are cell lines, columns are drugs.
- Cell lines (alphabetically, tab delimited; normalised name, name)
- Drug names (alphabetically, tab delimited; normalised name, name)
The matrix is tab-delimited, with NaN for missing entries. 
"""

import numpy, pandas

# Make names lowercase, remove dashes and spaces and dots
def lowercase(values):
    return [v.lower() for v in values]
def remove(values,char):
    return [v.replace(char,"") for v in values]
def normalise(values):
    return remove(remove(remove(remove(lowercase(values),"-")," "),"."),",")

# Process the raw CCLE data.
def load_ccle(location_file):
    data = pandas.read_csv(location_file,dtype=str)
    cell_lines = list(data["Primary Cell Line Name"].values)
    drugs = list(data["Compound"].values)
    ic50 = list(data["IC50 (uM)"].values)
    ec50 = list(data["EC50 (uM)"].values)
    
    unique_cell_lines, unique_drugs = sorted(set(cell_lines),key=lambda v: v.upper()), sorted(set(drugs),key=lambda v: v.upper())    
    no_cell_lines, no_drugs = len(unique_cell_lines), len(unique_drugs)
    
    ic50_matrix, ec50_matrix = numpy.empty((no_cell_lines,no_drugs)), numpy.empty((no_cell_lines,no_drugs))
    ic50_matrix[:], ic50_matrix[:] = numpy.NAN, numpy.NAN
    
    for cl,dr,ic,ec in zip(cell_lines,drugs,ic50,ec50):
        i, j = unique_cell_lines.index(cl), unique_drugs.index(dr)
        ic50_matrix[i,j], ec50_matrix[i,j] = ic, ec
        
    unique_cell_lines_normalised = normalise(unique_cell_lines)
    unique_drugs_normalised = normalise(unique_drugs)
    
    cell_lines_info = numpy.array(zip(unique_cell_lines_normalised,unique_cell_lines))
    drugs_info = numpy.array(zip(unique_drugs_normalised,unique_drugs))
    
    return ic50_matrix, ec50_matrix, cell_lines_info, drugs_info

# Load the data
file_drug_sensitivities = "./raw/CCLE_NP24.2009_Drug_data_2015.02.24.csv"
ic50, ec50, cell_lines, drugs = load_ccle(file_drug_sensitivities)

# Store the data
file_ic50, file_ec50, file_cell_lines, file_drugs = "./processed_all/ic50.txt", "./processed_all/ec50.txt", "./processed_all/cell_lines.txt", "./processed_all/drugs.txt"
numpy.savetxt(file_ic50, ic50, delimiter="\t")
numpy.savetxt(file_ec50, ec50, delimiter="\t")
numpy.savetxt(file_cell_lines, cell_lines, delimiter="\t", fmt="%s")
numpy.savetxt(file_drugs, drugs, delimiter="\t", fmt="%s")

# Print some statistics
no_drugs, no_cell_lines, no_observed_ic50, no_observed_ec50 = len(drugs), len(cell_lines), numpy.count_nonzero(~numpy.isnan(ic50)), numpy.count_nonzero(~numpy.isnan(ec50))
print "Number drugs: %s. Number cell lines: %s. Number of observed entries IC50 / EC50: %s / %s. Fraction observed IC50 / EC50: %s / %s." \
    % (no_drugs, no_cell_lines, no_observed_ic50, no_observed_ec50, no_observed_ic50 / float(no_drugs*no_cell_lines), no_observed_ec50 / float(no_drugs*no_cell_lines))