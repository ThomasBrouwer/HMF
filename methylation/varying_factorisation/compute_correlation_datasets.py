'''
Script for computing the correlations between the GE, GM, and PM datasets.
Compute both overall correlation per pair, and average per-sample correlation.

GE and GM. Global: -0.069254450455. Average: -0.0831571914872. 
GE and PM. Global: -0.116864662226. Average: -0.12496024005. 
GM and PM. Global: 0.13777601632.   Average: 0.0739930362991.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/" # "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.methylation.load_methylation import filter_driver_genes_std

import numpy
from scipy.stats import spearmanr



''' Load in data '''
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes_std()
R_ge, R_pm, R_gm = R_ge.T, R_pm.T, R_gm.T # rows are samples


''' Compute the correlations. '''
def global_correlation(R1, R2):
    v1, v2 = R1.flatten(), R2.flatten()
    return spearmanr(v1,v2).correlation

def average_sample_correlation(R1, R2):
    n_samples = R1.shape[0]
    correlations = [spearmanr(R1[n],R2[n]).correlation for n in range(n_samples)]
    return numpy.mean(correlations)
        
corr_global_ge_gm = global_correlation(R_ge, R_gm)
corr_global_ge_pm = global_correlation(R_ge, R_pm)
corr_global_gm_pm = global_correlation(R_gm, R_pm)
        
corr_avr_ge_gm = average_sample_correlation(R_ge, R_gm)
corr_avr_ge_pm = average_sample_correlation(R_ge, R_pm)
corr_avr_gm_pm = average_sample_correlation(R_gm, R_pm)

print "GE and GM. Global: %s. Average: %s. " % (corr_global_ge_gm, corr_avr_ge_gm)
print "GE and PM. Global: %s. Average: %s. " % (corr_global_ge_pm, corr_avr_ge_pm)
print "GM and PM. Global: %s. Average: %s. " % (corr_global_gm_pm, corr_avr_gm_pm)