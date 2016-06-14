"""
Process the raw CTRP drug sensitivity dataset.
- v20.meta.per_cell_line.txt contains cell line info (id, name, tissue, cancer type)
- v20.meta.per_compound.txt contains the drug info (id, name, SMILES)
- v20.meta.per_experiment.txt gives the cell line id used in each experiment id
- v20.data.curves_post_qc.txt gives the EC50 sensitivity scores for each experiment and drug

Return:
- The matrix of EC50 values (in order of the drug and cell line lists). 
  Rows are cell lines, columns are drugs.
- Cell lines (alphabetically, tab delimited; normalised name, name, id, cancer type, tissue type)
- Drug names (alphabetically, tab delimited; normalised name, name, id, SMILES)
The matrix is tab-delimited, with NaN for missing entries. 
"""

import numpy

# Make names lowercase, remove dashes and spaces and dots
def lowercase(v):
    return v.lower()
def remove(v,char):
    return v.replace(char,"")
def normalise(v):
    return remove(remove(remove(remove(lowercase(v),"-")," "),"."),",")
    
# Process the raw CTRP data.
# Extract the cell line id to info from location_cl_info
# Extract the drug id to info from location_drug_info
# Extract the experiment id to cell line id from location_experiment_info
# Extract the EC50 values from location_sensitivities and construct the matrix from this
def load_ctrp(location_cl_info,location_drug_info,location_experiment_info,location_sensitivities):
    # Extract the cell line id to name info from location_cl_info - master_ccl_id, ccl_name, ccl_availability, ccle_primary_site, ccle_primary_hist, ccle_hist_subtype_1
    info_cell_lines = [line.split("\t") for line in open(location_cl_info,'r').readlines()][1:]
    info_cell_lines = [(normalise(l[1]),l[1],l[0],l[4],l[3]) for l in info_cell_lines] #normalised name,name, id, cancer type, tissue
    sorted_info_cell_lines = sorted(info_cell_lines,key=lambda x:x[0].upper())
    cell_line_id_to_info = { cl_id:(norm_name,name,cl_id,cancer,tissue) for (norm_name,name,cl_id,cancer,tissue) in sorted_info_cell_lines }
    no_cell_lines = len(sorted_info_cell_lines)
    
    # Extract the drug id to name from location_drug_info - master_cpd_id, cpd_name, broad_cpd_id, top_test_conc_umol, cpd_status, inclusion_rationale, gene_symbol_of_protein_target, target_or_activity_of_compound, source_name, source_catalog_id, cpd_smiles
    info_drugs = [line.split("\r\n")[0].split("\t") for line in open(location_drug_info,'r').readlines()][1:] 
    info_drugs = [(normalise(l[1]),l[1],l[0],l[10]) for l in info_drugs] #normalised name, name, id, smiles
    sorted_info_drugs = sorted(info_drugs,key=lambda x:x[0].upper())
    drug_id_to_info = { drug_id:(norm_name,name,drug_id,smiles) for (norm_name,name,drug_id,smiles) in sorted_info_drugs }
    no_drugs = len(sorted_info_drugs)
    
    # Extract the experiment id to cell line id from location_experiment_info - experiment_id, run_id, experiment_date, culture_media, baseline_signal, cells_per_well, growth_mode, snp_fp_status, master_ccl_id
    info_experiments = [line.split("\r\n")[0].split("\t") for line in open(location_experiment_info,'r').readlines()][1:] 
    info_experiments = [(l[0],l[8]) for l in info_experiments] #experiment id, cell line id
    experiment_id_to_cell_line_id = { e_id:cl_id for e_id,cl_id in info_experiments }
    
    # Finally, extract the EC50 sensitivity values from location_sensitivities - experiment_id, conc_pts_fit, fit_num_param, p1_conf_int_high, p1_conf_int_low, p2_conf_int_high, p2_conf_int_low, p4_conf_int_high, p4_conf_int_low, p1_center, p2_slope, p3_total_decline, p4_baseline, apparent_ec50_umol, pred_pv_high_conc, area_under_curve, master_cpd_id
    sensitivities = [line.split("\r\n")[0].split("\t") for line in open(location_sensitivities,'r').readlines()][1:] 
    sensitivities = [(l[0],l[16],l[13]) for l in sensitivities] #experiment id, drug id, EC50
    
    matrix = numpy.empty((no_cell_lines,no_drugs))
    matrix[:] = numpy.NAN
    for experiment_id, drug_id, EC50 in sensitivities:
        cell_line_id = experiment_id_to_cell_line_id[experiment_id]
        i = sorted_info_cell_lines.index(cell_line_id_to_info[cell_line_id])
        j = sorted_info_drugs.index(drug_id_to_info[drug_id])
        matrix[i,j] = EC50
        
    return matrix, sorted_info_cell_lines, sorted_info_drugs
            
# Filter out any cell lines with only NaN entries
def filter_empty_cell_lines(sensitivities, info_cell_lines):
    cell_lines_to_remove = [i for i in range(0,len(info_cell_lines)) if all(numpy.isnan(sensitivities[i,:]))]
    filtered_sensitivities = [row for i,row in enumerate(sensitivities) if i not in cell_lines_to_remove]
    filtered_info_cell_lines = [info for i,info in enumerate(info_cell_lines) if i not in cell_lines_to_remove]
    return filtered_sensitivities, filtered_info_cell_lines            
            
    
# Run the code to extract the data
folder_data = "./raw/CTRPv2.0_2015_ctd2_ExpandedDataset/"
file_cl, file_drug = folder_data+"v20.meta.per_cell_line.txt", folder_data+"v20.meta.per_compound.txt"
file_experiment, file_sensitivity = folder_data+"v20.meta.per_experiment.txt", folder_data+"v20.data.curves_post_qc.txt"
sensitivities, cell_lines, drugs = load_ctrp(file_cl,file_drug,file_experiment,file_sensitivity)

# Filter out cell lines with no entries - only NaN
sensitivities_filtered, cell_lines_filtered = filter_empty_cell_lines(sensitivities, cell_lines)

# Make it into a numpy string array so we can store it easily
cell_lines_filtered = numpy.array(cell_lines_filtered,dtype=str)
drugs = numpy.array(drugs,dtype=str)

# Store the data
file_ec50, file_cell_lines, file_drugs = "./processed_all/ec50.txt", "./processed_all/cell_lines.txt", "./processed_all/drugs.txt"
numpy.savetxt(file_ec50, sensitivities_filtered, delimiter="\t")
numpy.savetxt(file_cell_lines, cell_lines_filtered, delimiter="\t", fmt="%s")
numpy.savetxt(file_drugs, drugs, delimiter="\t", fmt="%s")

# Print some statistics
no_drugs, no_cell_lines, no_observed = len(drugs), len(cell_lines_filtered), numpy.count_nonzero(~numpy.isnan(sensitivities))
print "Number drugs: %s. Number cell lines: %s. Number of observed entries: %s. Fraction observed: %s." \
    % (no_drugs, no_cell_lines, no_observed, no_observed / float(no_drugs*no_cell_lines))