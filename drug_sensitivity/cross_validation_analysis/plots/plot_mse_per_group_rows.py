"""
Plot the cross-validation performances (MSE) of the six methods,
- NMF-NP, NMTF-NP, BNMF, BNMTF, HMF D-MF, HMF D-MTF
where we group the rows by the number of observed entries we have in M.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from HMF.drug_sensitivity.load_dataset import load_data_without_empty

import matplotlib.pyplot as plt
import itertools
import numpy


''' Load performances. '''
folder_results = './../results/'

nmf_np = eval(open(folder_results+'nmf_np.txt','r').read())
nmtf_np = eval(open(folder_results+'nmtf_np.txt','r').read())
bnmf = eval(open(folder_results+'bnmf.txt','r').read())
bnmtf = eval(open(folder_results+'bnmtf.txt','r').read())
hmf_d_mf = eval(open(folder_results+'hmf_d_mf.txt','r').read())
hmf_d_mtf = eval(open(folder_results+'hmf_d_mtf.txt','r').read())

method_names = ['NMF-NP', 'NMTF-NP', 'BNMF', 'BNMTF', 'HMF D-MF', 'HMF D-MTF']
colours = ['#0000ff','#ff0000','#00ff00','#00ffff','#ff00ff','#ffff00']
all_i_j_real_pred = [nmf_np, nmtf_np, bnmf, bnmtf, hmf_d_mf, hmf_d_mtf]


''' Load datasets '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data = location+"data_row_01/"
R, M, cell_lines, drugs = load_data_without_empty(location_data+"gdsc_ic50_row_01.txt")

I, J = R.shape
indices = [(i,j) for i,j in itertools.product(range(I),range(J)) if M[i,j]]


''' For each row, compute the number of observed entries. 
    Then find the row indices for each bucket of observed entries. '''
entries_per_row = M.sum(axis=1)
buckets = [29,30,47,48] # Bucket thresholds: 10-29, 30-30, 31-47, 48
row_indices_per_bucket = [set([]) for bucket_max in buckets]
indices_per_bucket = [[] for bucket_max in buckets]
for i,j in indices:
    for b,bucket_max in enumerate(buckets):
        if entries_per_row[i] <= bucket_max:
            row_indices_per_bucket[b].add(i), indices_per_bucket[b].append((i,j))
            break
print "Rows per bucket: %s." % ([len(indices_list) for indices_list in row_indices_per_bucket])
print "Entries per bucket: %s." % ([len(indices_list) for indices_list in indices_per_bucket])
            

''' For each bucket, load in the relevant (real,pred) pairs for each method, and compute the MSE. '''
real_pred_per_bucket_per_method = [
    [
        [
            (real, pred) for (i,j,real,pred) in i_j_real_pred if i in row_indices_per_bucket[b]
        ]
        for b,indices_bucket_b in enumerate(row_indices_per_bucket)
    ]
    for i_j_real_pred in all_i_j_real_pred
]
mse_per_bucket_per_method = [
    [
        sum([(real-pred)**2 for (real,pred) in real_pred]) / float(len(real_pred))
        for real_pred in real_pred_per_bucket
    ]
    for real_pred_per_bucket in real_pred_per_bucket_per_method
]


''' Plot settings. '''
y_min, y_max = 0.05, 0.14
bar_width = 0.14
xtick_offset = 0.3555 # to make xticks center of bars

folder_plots = "./"
plot_file = folder_plots+"performance_grouped_rows.png"


''' Plot the performances in a bar chart. '''
fig = plt.figure(figsize=(4,2))
fig.subplots_adjust(left=0.11, right=0.99, bottom=0.165, top=0.98)
plt.xlabel("Number of observed entries per row", fontsize=9, labelpad=1)
plt.ylabel("MSE", fontsize=9, labelpad=1)

x = range(len(buckets))
for m, mse_per_bucket in enumerate(mse_per_bucket_per_method):
    method_name, colour = method_names[m], colours[m]
    y = mse_per_bucket
    x_i = [v+m*bar_width for v in x]
    plt.bar(x_i, y, bar_width, color=colour, edgecolor='black', label=method_name)
    
xlabels = [
    "%s-%s" % (min(entries_per_row), buckets[0]),
    "%s-%s" % (buckets[0] + 1, buckets[1]),
    "%s-%s" % (buckets[1] + 1, buckets[2]),
    "%s-%s" % (buckets[2] + 1, buckets[3]),
]
plt.xticks(numpy.arange(len(buckets)) + xtick_offset, xlabels, fontsize=6)
plt.yticks(numpy.arange(0,y_max+1,0.01),fontsize=6)
plt.ylim(y_min, y_max)
plt.xlim(-0.2, len(buckets)-0.1)
plt.legend(ncol=2, fontsize=8)

plt.savefig(plot_file, dpi=600)