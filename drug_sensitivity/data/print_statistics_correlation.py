'''
Print information about the correlation between the overlapping entries of the
four datasets.

Correlation GDSC and CTRP: 0.467550926187.
Correlation GDSC and CCLE IC50: 0.593885325724.
Correlation GDSC and CCLE EC50: 0.389690956349.
Correlation CTRP and CCLE IC50: 0.440493814038.
Correlation CTRP and CCLE EC50: 0.449244917323.
Correlation CCLE IC50 and CCLE EC50: 0.645008511739.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

from HMF.drug_sensitivity.load_dataset import load_data

from scipy.stats import spearmanr
import itertools


''' Load in the datasets '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data = location+"data_row_01/"

R_gdsc,     M_gdsc    = load_data(location_data+"gdsc_ic50_row_01.txt")
R_ctrp,     M_ctrp    = load_data(location_data+"ctrp_ec50_row_01.txt")
R_ccle_ic,  M_ccle_ic = load_data(location_data+"ccle_ic50_row_01.txt")
R_ccle_ec,  M_ccle_ec = load_data(location_data+"ccle_ec50_row_01.txt")


''' Compute the correlations. '''
def compute_correlation(R1, M1, R2, M2):
    assert M1.shape == M2.shape
    I,J = M1.shape
    M_overlap = M1 * M2
    indices_overlap = [(i,j) for i,j in itertools.product(range(I),range(J)) if M_overlap[i,j]]
    
    v1 = [R1[i,j] for i,j  in indices_overlap]
    v2 = [R2[i,j] for i,j  in indices_overlap]   
    corr = spearmanr(v1,v2).correlation
    return corr

corr_gdsc_ctrp       = compute_correlation(R_gdsc,    M_gdsc,    R_ctrp,    M_ctrp   )
corr_gdsc_ccle_ic    = compute_correlation(R_gdsc,    M_gdsc,    R_ccle_ic, M_ccle_ic)
corr_gdsc_ccle_ec    = compute_correlation(R_gdsc,    M_gdsc,    R_ccle_ec, M_ccle_ec)
corr_ctrp_ccle_ic    = compute_correlation(R_ctrp,    M_ctrp,    R_ccle_ic, M_ccle_ic)
corr_ctrp_ccle_ec    = compute_correlation(R_ctrp,    M_ctrp,    R_ccle_ec, M_ccle_ec)
corr_ccle_ic_ccle_ec = compute_correlation(R_ccle_ic, M_ccle_ic, R_ccle_ec, M_ccle_ec)
    

''' Print the correlations. '''
print "Correlation GDSC and CTRP: %s." % corr_gdsc_ctrp
print "Correlation GDSC and CCLE IC50: %s." % corr_gdsc_ccle_ic
print "Correlation GDSC and CCLE EC50: %s." % corr_gdsc_ccle_ec
print "Correlation CTRP and CCLE IC50: %s." % corr_ctrp_ccle_ic
print "Correlation CTRP and CCLE EC50: %s." % corr_ctrp_ccle_ec
print "Correlation CCLE IC50 and CCLE EC50: %s." % corr_ccle_ic_ccle_ec