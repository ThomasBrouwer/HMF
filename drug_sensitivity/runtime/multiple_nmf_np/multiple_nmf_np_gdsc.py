"""
Measure the time it takes to train a single model.

Time taken: 16.8612511158 seconds. Average per iteration: 0.0168612511158.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/" # "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from HMF.code.models.nmf_np import nmf_np
from HMF.drug_sensitivity.load_dataset import load_data_without_empty
from HMF.drug_sensitivity.load_dataset import load_data_filter

import time
import numpy


''' Load datasets '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data =                 location+"data_row_01/"
location_features_drugs =       location+"features_drugs/"
location_features_cell_lines =  location+"features_cell_lines/"
location_kernels =              location+"kernels_features/"

R_gdsc,     M_gdsc,     cell_lines, drugs = load_data_without_empty(location_data+"gdsc_ic50_row_01.txt")
R_ctrp,     M_ctrp                        = load_data_filter(location_data+"ctrp_ec50_row_01.txt",rows=cell_lines,columns=None)
R_ccle_ec,  M_ccle_ec                     = load_data_filter(location_data+"ccle_ec50_row_01.txt",rows=cell_lines,columns=None)
R_ccle_ic,  M_ccle_ic                     = load_data_filter(location_data+"ccle_ic50_row_01.txt",rows=cell_lines,columns=None)

R_concat = numpy.concatenate((R_gdsc,R_ctrp,R_ccle_ec,R_ccle_ic),axis=1) #columns
M_concat = numpy.concatenate((M_gdsc,M_ctrp,M_ccle_ec,M_ccle_ic),axis=1) #columns


''' Remove entirely empty columns, due to the other three datasets that we concatenate '''
def remove_empty_columns(R,M):
    new_R, new_M = [], []
    for j,sum_column in enumerate(M.sum(axis=0)):
        if sum_column > 0:
            new_R.append(R[:,j])
            new_M.append(M[:,j])
    return numpy.array(new_R).T, numpy.array(new_M).T
        
R, M = remove_empty_columns(R_concat,M_concat)


''' Settings NMF '''
iterations = 1000
init_UV = 'random'
K = 10


''' Run the method and time it. '''
time_start = time.time()

NMF = nmf_np(R,M,K)
NMF.initialise(init_UV)
NMF.run(iterations)
    
time_end = time.time()
time_taken = time_end - time_start
time_average = time_taken / iterations
print "Time taken: %s seconds. Average per iteration: %s." % (time_taken, time_average)