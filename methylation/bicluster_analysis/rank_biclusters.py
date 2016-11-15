"""
Sort the biclusters by importance.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from plot_datasets import plot_heatmap

import numpy, itertools


''' Method for returning a sorted list [(k,l,Skl)] of biclusters with the 
    highest S value, for the given S matrix. '''
def sorted_biclusters_one_S(S):
    K,L = S.shape
    biclusters = [(k,l,S[k,l]) for k,l in itertools.product(range(0,K),range(0,L))]
    biclusters_sorted = sorted(biclusters, key=lambda x:x[2], reverse=True)
    return biclusters_sorted
    
''' Same but now total S value across all three. '''    
def sorted_biclusters_all_S(S1,S2,S3):
    K,L = S1.shape
    biclusters = [(k,l,S1[k,l]+S2[k,l]+S3[k,l]) for k,l in itertools.product(range(0,K),range(0,L))]
    biclusters_sorted = sorted(biclusters, key=lambda x:abs(x[2]), reverse=True)
    return biclusters_sorted

''' Method for returning a sorted list [(k,l,variance_kl)] of biclusters with 
    that explain the highest amount of variance (F * S * G.T). '''
def sorted_biclusters_one_variance(S,F,G):
    K,L = S.shape
    biclusters = []
    for k,l in itertools.product(range(0,K),range(0,L)):
        S_kl = numpy.zeros((K,L))
        S_kl[k,l] = S[k,l]
        R_pred_kl = numpy.dot(F,numpy.dot(S_kl,G.T))
        variance = (R_pred_kl**2).sum()
        biclusters.append((k,l,variance))
    biclusters_sorted = sorted(biclusters, key=lambda x:x[2], reverse=True)
    return biclusters_sorted
    
''' Same but now total variance across all three. ''' 
def sorted_biclusters_all_variance(S1,S2,S3,F,G):
    K,L = S1.shape
    biclusters = []
    for k,l in itertools.product(range(0,K),range(0,L)):
        S1_kl, S2_kl, S3_kl = numpy.zeros((K,L)), numpy.zeros((K,L)), numpy.zeros((K,L))
        S1_kl[k,l], S2_kl[k,l], S3_kl[k,l] = S1[k,l], S2[k,l], S3[k,l]
        R1_pred_kl = numpy.dot(F,numpy.dot(S1_kl,G.T))
        R2_pred_kl = numpy.dot(F,numpy.dot(S2_kl,G.T))
        R3_pred_kl = numpy.dot(F,numpy.dot(S3_kl,G.T))
        variance = (R1_pred_kl**2 + R2_pred_kl**2 + R3_pred_kl**2).sum()
        biclusters.append((k,l,variance))
    biclusters_sorted = sorted(biclusters, key=lambda x:x[2], reverse=True)
    return biclusters_sorted
    
    
if __name__ == "__main__":
    ''' Load in factor matrices. '''
    folder_matrices = project_location+'HMF/methylation/bicluster_analysis/matrices/'
    
    F_genes = numpy.loadtxt(folder_matrices+'F_genes')
    F_samples = numpy.loadtxt(folder_matrices+'F_samples')
    S_ge = numpy.loadtxt(folder_matrices+'S_ge')
    S_pm = numpy.loadtxt(folder_matrices+'S_pm')
    S_gm = numpy.loadtxt(folder_matrices+'S_gm')
    
    ''' Print ranked biclusters using total S. '''
    sorted_all_S = sorted_biclusters_all_S(S_ge,S_pm,S_gm)
    print "Sorted by S: ", sorted_all_S
    
    ''' Print ranked biclusters using total variance. '''
    sorted_all_variance = sorted_biclusters_all_variance(S_ge,S_pm,S_gm,F_genes,F_samples)
    print "Sorted by variance: ", sorted_all_variance
    
    ''' Also compute it per dataset. '''
    sorted_ge_variance = sorted_biclusters_one_variance(S_ge,F_genes,F_samples)
    sorted_pm_variance = sorted_biclusters_one_variance(S_pm,F_genes,F_samples)
    sorted_gm_variance = sorted_biclusters_one_variance(S_gm,F_genes,F_samples)