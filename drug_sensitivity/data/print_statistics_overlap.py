'''
Print information about individual datasets:
- Number of drugs, cell lines
- Fraction of entries observed
- Fraction of entries with overlap in GDSC IC50
- Fraction of entries with overlap in CTRP EC50
- Fraction of entries with overlap in CCLE IC50
- Fraction of entries with overlap in CCLE EC50
'''

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.drug_sensitivity.load_dataset import load_data_without_empty, load_data_filter

location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data =                 location+"data_row_01/"

def print_overlap(M_main,M1,M2,M3,names):
    n_cell_lines, n_drugs = M_main.shape
    n_observed = M_main.sum()
    fraction_observed = n_observed / float(n_cell_lines * n_drugs)
    
    M_overlap_1 = M_main * M1
    n_overlap_1 = M_overlap_1.sum()
    fraction_overlap_1 = n_overlap_1 / float(n_cell_lines * n_drugs)
    
    M_overlap_2 = M_main * M2
    n_overlap_2 = M_overlap_2.sum()
    fraction_overlap_2 = n_overlap_2 / float(n_cell_lines * n_drugs)
    
    M_overlap_3 = M_main * M3
    n_overlap_3 = M_overlap_3.sum()
    fraction_overlap_3 = n_overlap_3 / float(n_cell_lines * n_drugs)
    
    print "Dataset %s." % names[0]
    print "Number cell lines: %s. Number drugs: %s." % (n_cell_lines,n_drugs)
    print "Number observed: %s. Fraction observed: %s." % (n_observed,fraction_observed)
    print "%s. Number overlap: %s. Fraction overlap: %s." % (names[1],n_overlap_1,fraction_overlap_1)
    print "%s. Number overlap: %s. Fraction overlap: %s." % (names[2],n_overlap_2,fraction_overlap_2)
    print "%s. Number overlap: %s. Fraction overlap: %s." % (names[3],n_overlap_3,fraction_overlap_3)


''' GDSC IC50 as the main dataset '''
R_gdsc, M_gdsc, cell_lines, drugs = load_data_without_empty(location_data+"gdsc_ic50_row_01.txt")
R_ctrp,     M_ctrp                        = load_data_filter(location_data+"ctrp_ec50_row_01.txt",cell_lines,drugs)
R_ccle_ic,  M_ccle_ic                     = load_data_filter(location_data+"ccle_ic50_row_01.txt",cell_lines,drugs)
R_ccle_ec,  M_ccle_ec                     = load_data_filter(location_data+"ccle_ec50_row_01.txt",cell_lines,drugs)

print_overlap(M_gdsc,M_ctrp,M_ccle_ic,M_ccle_ec,['GDSC','CTRP','CCLE IC','CCLE EC'])

''' CTRP EC50 as the main dataset '''
R_ctrp,     M_ctrp,   cell_lines, drugs   = load_data_without_empty(location_data+"ctrp_ec50_row_01.txt")
R_ccle_ec,  M_ccle_ec                     = load_data_filter(location_data+"ccle_ec50_row_01.txt",cell_lines,drugs)
R_gdsc,     M_gdsc                        = load_data_filter(location_data+"gdsc_ic50_row_01.txt",cell_lines,drugs)
R_ccle_ic,  M_ccle_ic                     = load_data_filter(location_data+"ccle_ic50_row_01.txt",cell_lines,drugs)

print_overlap(M_ctrp,M_gdsc,M_ccle_ic,M_ccle_ec,['CTRP','GDSC','CCLE IC','CCLE EC'])

''' CCLE IC50 as the main dataset '''
R_ccle_ic,  M_ccle_ic, cell_lines, drugs  = load_data_without_empty(location_data+"ccle_ic50_row_01.txt")
R_ctrp,     M_ctrp                        = load_data_filter(location_data+"ctrp_ec50_row_01.txt",cell_lines,drugs)
R_gdsc,     M_gdsc                        = load_data_filter(location_data+"gdsc_ic50_row_01.txt",cell_lines,drugs)
R_ccle_ec,  M_ccle_ec                     = load_data_filter(location_data+"ccle_ec50_row_01.txt",cell_lines,drugs)

print_overlap(M_ccle_ic,M_gdsc,M_ctrp,M_ccle_ec,['CCLE IC','GDSC','CTRP','CCLE EC'])

''' CCLE IC50 as the main dataset '''
R_ccle_ec,  M_ccle_ec, cell_lines, drugs  = load_data_without_empty(location_data+"ccle_ec50_row_01.txt")
R_ctrp,     M_ctrp                        = load_data_filter(location_data+"ctrp_ec50_row_01.txt",cell_lines,drugs)
R_gdsc,     M_gdsc                        = load_data_filter(location_data+"gdsc_ic50_row_01.txt",cell_lines,drugs)
R_ccle_ic,  M_ccle_ic                     = load_data_filter(location_data+"ccle_ic50_row_01.txt",cell_lines,drugs)

print_overlap(M_ccle_ec,M_gdsc,M_ctrp,M_ccle_ic,['CCLE EC','GDSC','CTRP','CCLE IC'])


'''
Dataset GDSC.
Number cell lines: 399. Number drugs: 48.
Number observed: 14090. Fraction observed: 0.735693400167.
CTRP. Number overlap: 10006. Fraction overlap: 0.522451963241.
CCLE IC. Number overlap: 1789. Fraction overlap: 0.093410609858.
CCLE EC. Number overlap: 1150. Fraction overlap: 0.0600459482038.

Dataset CTRP.
Number cell lines: 379. Number drugs: 46.
Number observed: 14998. Fraction observed: 0.860273029712.
GDSC. Number overlap: 10006. Fraction overlap: 0.573935987152.
CCLE IC. Number overlap: 2085. Fraction overlap: 0.119593896983.
CCLE EC. Number overlap: 1285. Fraction overlap: 0.0737065504187.

Dataset CCLE IC.
Number cell lines: 253. Number drugs: 16.
Number observed: 3903. Fraction observed: 0.964179841897.
GDSC. Number overlap: 1789. Fraction overlap: 0.441946640316.
CTRP. Number overlap: 2085. Fraction overlap: 0.51506916996.
CCLE EC. Number overlap: 2229. Fraction overlap: 0.55064229249.

Dataset CCLE EC.
Number cell lines: 252. Number drugs: 16.
Number observed: 2374. Fraction observed: 0.58878968254.
GDSC. Number overlap: 1150. Fraction overlap: 0.285218253968.
CTRP. Number overlap: 1285. Fraction overlap: 0.318700396825.
CCLE IC. Number overlap: 2229. Fraction overlap: 0.552827380952.
'''