"""
Process the raw Sanger GDSC drug sensitivity dataset, and extract:
- The matrix of IC50 values (in order of the drug and cell line lists). 
  Rows are cell lines, columns are drugs.
- Cell lines (alphabetically, tab delimited; normalised name, name, cosmic id, cancer type, tissue type)
- Drug names (alphabetically, tab delimited; normalised name, name)
The matrix is tab-delimited, with NaN for missing entries. 
"""

import pandas, numpy

# Make names lowercase, remove dashes and spaces and dots
def lowercase(values):
    return [v.lower() for v in values]
def remove(values,char):
    return [v.replace(char,"") for v in values]
def normalise(values):
    return remove(remove(remove(remove(lowercase(values),"-")," "),"."),",")

# Load the data and return the matrix, list of row names, list of column names
# First row is column names, rows 2-6 are useless, then the data starts (rows are cell lines)
# Drugs start from column 76 (Erlotinib_IC_50) and end at 215 (AG-014699_IC_50) - 140 in total.
# One drug is duplicate (AZD6482_IC_50) so we filter out AZD6482_IC_50.1
def load_gdsc(file_location):
    skip_rows, ic50_text, no_drugs, filter_drugs = 5, "_IC_50", 140, ["AZD6482_IC_50.1"]
    data = pandas.read_csv(file_location,dtype=str)
    
    column_names = list(data.columns.values)
    cell_line_names = list(data['Cell Line'].values)[skip_rows:]
    cell_line_names_normalised = normalise(cell_line_names)
    cosmic_ids = list(data['Cosmic_ID'].values)[skip_rows:]
    cancer_types = list(data['Cancer Type'].values)[skip_rows:]
    tissue_types = list(data['Tissue'].values)[skip_rows:]
    cell_lines = zip(cell_line_names_normalised,cell_line_names,cosmic_ids,cancer_types,tissue_types)
    
    drug_names = [name for name in column_names if len(name.split(ic50_text)) > 1][:no_drugs]
    drug_names = [name for name in drug_names if name not in filter_drugs]
    sensitivities = data.as_matrix(drug_names)[skip_rows:]
    
    drug_names = [name.split(ic50_text)[0] for name in drug_names]
    drug_names_normalised = normalise(drug_names)
    drugs_info = zip(drug_names_normalised,drug_names)
    
    return sensitivities, cell_lines, drugs_info

# Sort the row and column names, reorder the matrix accordingly, and return all three again
def reorder_alphabetically(matrix,row_names,column_names):
    sorted_row_names = sorted(row_names,key=lambda x:x[0].upper())
    sorted_column_names = sorted(column_names,key=lambda x:x[0].upper())
    new_matrix = numpy.empty(matrix.shape)
    for i,row_name in enumerate(row_names):
        for j,col_name in enumerate(column_names):
            new_i,new_j = sorted_row_names.index(row_name), sorted_column_names.index(col_name)
            new_matrix[new_i,new_j] = matrix[i,j]
    return new_matrix, sorted_row_names, sorted_column_names
            
# Run the preprocessing
file_gdsc = "./raw/gdsc_manova_input_w5.csv"
sensitivities, cell_lines, drugs = load_gdsc(file_gdsc)
sensitivities_sorted, cell_lines_sorted, drugs_sorted = reorder_alphabetically(sensitivities,cell_lines,drugs)

cell_lines_sorted = numpy.array(cell_lines_sorted,dtype=str)

# Store the matrices
file_ic50, file_cell_lines, file_drugs = "./processed_all/ic50.txt", "./processed_all/cell_lines.txt", "./processed_all/drugs.txt"
numpy.savetxt(file_ic50, sensitivities_sorted, delimiter="\t")
numpy.savetxt(file_cell_lines, cell_lines_sorted, delimiter="\t", fmt="%s")
numpy.savetxt(file_drugs, drugs_sorted, delimiter="\t", fmt="%s")

# Print some statistics
no_drugs, no_cell_lines, no_observed = len(drugs_sorted), len(cell_lines_sorted), numpy.count_nonzero(~numpy.isnan(sensitivities_sorted))
print "Number drugs: %s. Number cell lines: %s. Number of observed entries: %s. Fraction observed: %s." \
    % (no_drugs, no_cell_lines, no_observed, no_observed / float(no_drugs*no_cell_lines))