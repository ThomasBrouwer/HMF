"""
Read in the lists of drugs and cell lines, clean up the names (lowercase, 
remove dashes spaces dots commas), and then find the intersection between names.
Plot as a Venn diagram.

Better method would be to map drug names to PubChem ids and then find intersection.
"""

import numpy

''' Read in the names '''
location_ccle, location_ctrp, location_gdsc = "./CCLE/processed_all/", "./CTRP/processed_all/", "./GDSC/processed_all/"

drugs_ccle_full = list(numpy.loadtxt(location_ccle+"drugs.txt",dtype=str,delimiter="\t")[:,1])
drugs_ctrp_full = list(numpy.loadtxt(location_ctrp+"drugs.txt",dtype=str,delimiter="\t")[:,1])
drugs_gdsc_full = list(numpy.loadtxt(location_gdsc+"drugs.txt",dtype=str,delimiter="\t")[:,1])
cell_lines_ccle_full = list(numpy.loadtxt(location_ccle+"cell_lines.txt",dtype=str,delimiter="\t")[:,1])
cell_lines_ctrp_full = list(numpy.loadtxt(location_ctrp+"cell_lines.txt",dtype=str,delimiter="\t")[:,1])
cell_lines_gdsc_full = list(numpy.loadtxt(location_gdsc+"cell_lines.txt",dtype=str,delimiter="\t")[:,1])

''' Make names lowercase, remove dashes and spaces and dots '''
def lowercase(values):
    return [v.lower() for v in values]
def remove(values,char):
    return [v.replace(char,"") for v in values]
def make_lowercase_remove_dashes_spaces_dots_commas(values):
    return remove(remove(remove(remove(lowercase(values),"-")," "),"."),",")

drugs_ccle = make_lowercase_remove_dashes_spaces_dots_commas(drugs_ccle_full)
drugs_ctrp = make_lowercase_remove_dashes_spaces_dots_commas(drugs_ctrp_full)
drugs_gdsc = make_lowercase_remove_dashes_spaces_dots_commas(drugs_gdsc_full)
cell_lines_ccle = make_lowercase_remove_dashes_spaces_dots_commas(cell_lines_ccle_full)
cell_lines_ctrp = make_lowercase_remove_dashes_spaces_dots_commas(cell_lines_ctrp_full)
cell_lines_gdsc = make_lowercase_remove_dashes_spaces_dots_commas(cell_lines_gdsc_full)

''' Check whether any drug or cell line names are no longer unique '''
assert len(drugs_ccle) == len(set(drugs_ccle))
assert len(drugs_ctrp) == len(set(drugs_ctrp))
assert len(drugs_gdsc) == len(set(drugs_gdsc))
assert len(cell_lines_ccle) == len(set(cell_lines_ccle))
assert len(cell_lines_ctrp) == len(set(cell_lines_ctrp))
assert len(cell_lines_gdsc) == len(set(cell_lines_gdsc))

''' Compute the list of all drugs and cell lines '''
all_drugs = sorted(set(drugs_ccle+drugs_ctrp+drugs_gdsc))
all_cell_lines = sorted(set(cell_lines_ccle+cell_lines_ctrp+cell_lines_gdsc))

''' Compute overlaps '''
def overlap(l1,l2):
    return [v1 for v1 in l1 if v1 in l2]
    
overlap_drugs_ccle_ctrp = overlap(drugs_ccle,drugs_ctrp)
overlap_drugs_ccle_gdsc = overlap(drugs_ccle,drugs_gdsc)
overlap_drugs_ctrp_gdsc = overlap(drugs_ctrp,drugs_gdsc)
overlap_drugs_all = overlap(overlap_drugs_ccle_ctrp,drugs_gdsc)

overlap_cell_lines_ccle_ctrp = overlap(cell_lines_ccle,cell_lines_ctrp)
overlap_cell_lines_ccle_gdsc = overlap(cell_lines_ccle,cell_lines_gdsc)
overlap_cell_lines_ctrp_gdsc = overlap(cell_lines_ctrp,cell_lines_gdsc)
overlap_cell_lines_all = overlap(overlap_cell_lines_ccle_ctrp,cell_lines_gdsc)

''' Store the drugs and cell lines in two or three of the datasets '''
drugs_in_two_or_more_datasets = [drug for drug in all_drugs if (drug in overlap_drugs_ccle_ctrp + overlap_drugs_ccle_gdsc + overlap_drugs_ctrp_gdsc)]
cell_lines_in_two_or_more_datasets = [cl for cl in all_cell_lines if (cl in overlap_cell_lines_ccle_ctrp + overlap_cell_lines_ccle_gdsc + overlap_cell_lines_ctrp_gdsc)]

''' Compute numbers for cell lines '''
drugs_ccle_only = [drug for drug in drugs_ccle if (drug not in overlap_drugs_ccle_ctrp and drug not in overlap_drugs_ccle_gdsc)]
drugs_ctrp_only = [drug for drug in drugs_ctrp if (drug not in overlap_drugs_ccle_ctrp and drug not in overlap_drugs_ctrp_gdsc)]
drugs_gdsc_only = [drug for drug in drugs_gdsc if (drug not in overlap_drugs_ccle_gdsc and drug not in overlap_drugs_ctrp_gdsc)]
cell_lines_ccle_only = [cl for cl in cell_lines_ccle if (cl not in overlap_cell_lines_ccle_ctrp and cl not in overlap_cell_lines_ccle_gdsc)]
cell_lines_ctrp_only = [cl for cl in cell_lines_ctrp if (cl not in overlap_cell_lines_ccle_ctrp and cl not in overlap_cell_lines_ctrp_gdsc)]
cell_lines_gdsc_only = [cl for cl in cell_lines_gdsc if (cl not in overlap_cell_lines_ccle_gdsc and cl not in overlap_cell_lines_ctrp_gdsc)]

n_drugs_ccle, n_drugs_ctrp, n_drugs_gdsc = len(drugs_ccle_only), len(drugs_ctrp_only), len(drugs_gdsc_only)
n_drugs_ccle_ctrp, n_drugs_ccle_gdsc, n_drugs_ctrp_gdsc = len(overlap_drugs_ccle_ctrp), len(overlap_drugs_ccle_gdsc), len(overlap_drugs_ctrp_gdsc)
n_drugs_ccle_ctrp_gdsc = len(overlap_drugs_all)

n_cell_lines_ccle, n_cell_lines_ctrp, n_cell_lines_gdsc = len(cell_lines_ccle_only), len(cell_lines_ctrp_only), len(cell_lines_gdsc_only)
n_cell_lines_ccle_ctrp, n_cell_lines_ccle_gdsc, n_cell_lines_ctrp_gdsc = len(overlap_cell_lines_ccle_ctrp), len(overlap_cell_lines_ccle_gdsc), len(overlap_cell_lines_ctrp_gdsc)
n_cell_lines_ccle_ctrp_gdsc = len(overlap_cell_lines_all)

''' Plot the Venn diagram '''
from matplotlib_venn import venn3
from matplotlib import pyplot as plt

plt.figure()
venn3(subsets=[set(drugs_ccle),set(drugs_ctrp),set(drugs_gdsc)], set_labels=("CCLE","CTRP","GDSC"))
plt.title("Intersections drugs for drug sensitivity datasets")
plt.savefig("./venn_drugs.pdf")

plt.figure()
venn3(subsets=[set(cell_lines_ccle),set(cell_lines_ctrp),set(cell_lines_gdsc)], set_labels=("CCLE","CTRP","GDSC"))
plt.title("Intersections cell lines for drug sensitivity datasets")
plt.savefig("./venn_cell_lines.pdf")

''' Store the drug and cell line names with 2+ datasets '''
numpy.savetxt("./overlap/features_drugs/drugs.txt", drugs_in_two_or_more_datasets, fmt="%s")
numpy.savetxt("./overlap/features_cell_lines/cell_lines.txt", cell_lines_in_two_or_more_datasets, fmt="%s")

''' Also store the full (un-normalised) names for the drugs and cell lines '''
drugs_in_two_or_more_datasets_full, cell_lines_in_two_or_more_datasets_full = [], []
for drug in drugs_in_two_or_more_datasets:
    drug_full = drugs_gdsc_full[drugs_gdsc.index(drug)] if drug in drugs_gdsc \
                else drugs_ctrp_full[drugs_ctrp.index(drug)] if drug in drugs_ctrp else drugs_ccle_full[drugs_ccle.index(drug)]
    drugs_in_two_or_more_datasets_full.append(drug_full)
for cl in cell_lines_in_two_or_more_datasets:
    cell_line_full = cell_lines_gdsc_full[cell_lines_gdsc.index(cl)] if cl in cell_lines_gdsc \
                     else cell_lines_ctrp_full[cell_lines_ctrp.index(cl)] if cl in cell_lines_ctrp else cell_lines_ccle_full[cell_lines_ccle.index(cl)]
    cell_lines_in_two_or_more_datasets_full.append(cell_line_full)
numpy.savetxt("./overlap/features_drugs/drugs_full.txt", drugs_in_two_or_more_datasets_full, fmt="%s")
numpy.savetxt("./overlap/features_cell_lines/cell_lines_full.txt", cell_lines_in_two_or_more_datasets_full, fmt="%s")