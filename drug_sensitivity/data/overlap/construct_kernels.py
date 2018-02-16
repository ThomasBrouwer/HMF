"""
Construct the similarity kernels and save them.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from HMF.code.kernels.jaccard_kernel import JaccardKernel
from HMF.code.kernels.gaussian_kernel import GaussianKernel

import numpy, matplotlib.pyplot as plt

''' Load in the features '''
location_drugs, location_cell_lines = "./features_drugs/", "./features_cell_lines/"
location_kernels = "./kernels_features/"

delim = "\t"
name_drug_names,name_cl_names =         'drugs.txt',            'cell_lines_features.txt'
name_drug_1d2d, name_drug_1d2d_std =    'drug_1d2d.txt',        'drug_1d2d_std.txt'
name_drug_fps,  name_drug_targets =     'drug_fingerprints.txt','drug_targets.txt'
name_cl_ge,     name_cl_ge_std =        'gene_expression.txt',  'gene_expression_std.txt'
name_cl_cnv,    name_cl_cnv_std =       'cnv.txt',              'cnv_std.txt'
name_cl_mutation =                      'mutation.txt'

drug_names =        numpy.loadtxt(location_drugs+name_drug_names,   dtype=str)
cell_line_names =   numpy.loadtxt(location_cell_lines+name_cl_names,dtype=str)

''' Construct similarity kernels for the drugs '''
no_features_1d2d = 1160

kernel_drug_1d2d = GaussianKernel()
kernel_drug_1d2d.load_features_construct_store_kernel(
    location_features=  location_drugs+     name_drug_1d2d,
    location_output=    location_kernels+   name_drug_1d2d,
    sigma_2=            no_features_1d2d/4.
)
kernel_drug_1d2d_std = GaussianKernel()
kernel_drug_1d2d_std.load_features_construct_store_kernel(
    location_features=  location_drugs+     name_drug_1d2d_std,
    location_output=    location_kernels+   name_drug_1d2d_std,
    sigma_2=            no_features_1d2d/4.
)

kernel_drug_fps = JaccardKernel()
kernel_drug_fps.load_features_construct_store_kernel(
    location_features=  location_drugs+     name_drug_fps,
    location_output=    location_kernels+   name_drug_fps
)

kernel_drug_targets = JaccardKernel()
kernel_drug_targets.load_features_construct_store_kernel(
    location_features=  location_drugs+     name_drug_targets,
    location_output=    location_kernels+   name_drug_targets
)

''' Construct similarity kernels for the cell lines '''
no_features_ge,no_features_cnv = 13321, 426

kernel_cl_ge = GaussianKernel()
kernel_cl_ge.load_features_construct_store_kernel(
    location_features=  location_cell_lines+name_cl_ge,
    location_output=    location_kernels+name_cl_ge,
    sigma_2=            no_features_ge/4.
)
kernel_cl_ge_std = GaussianKernel()
kernel_cl_ge_std.load_features_construct_store_kernel(
    location_features=  location_cell_lines+name_cl_ge_std,
    location_output=    location_kernels+   name_cl_ge_std,
    sigma_2=            no_features_ge/4.
)

kernel_cl_cnv = GaussianKernel()
kernel_cl_cnv.load_features_construct_store_kernel(
    location_features=  location_cell_lines+name_cl_cnv,
    location_output=    location_kernels+   name_cl_cnv,
    sigma_2=            no_features_cnv/4.
)
kernel_cl_cnv_std = GaussianKernel()
kernel_cl_cnv_std.load_features_construct_store_kernel(
    location_features=  location_cell_lines+name_cl_cnv_std,
    location_output=    location_kernels+   name_cl_cnv_std,
    sigma_2=            no_features_cnv/4.
)

kernel_cl_mutation = JaccardKernel()
kernel_cl_mutation.load_features_construct_store_kernel(
    location_features=  location_cell_lines+name_cl_mutation,
    location_output=    location_kernels+   name_cl_mutation
)

''' Plot the distributions of kernel values '''
folder_plots = location_kernels+"plots_distributions/"
kernel_plots = [
    (kernel_drug_1d2d,      "kernel_drug_1d2d",     "Drug kernel 1D2D descriptors"),
    (kernel_drug_1d2d_std,  "kernel_drug_1d2d_std", "Drug kernel 1D2D descriptors (standardised)"),
    (kernel_drug_fps,       "kernel_drug_fps",      "Drug kernel fingerprints"),
    (kernel_drug_targets,   "kernel_drug_targets",  "Drug kernel targets"),
    (kernel_cl_ge,          "kernel_cl_ge",         "Cell line kernel gene expression"),
    (kernel_cl_ge_std,      "kernel_cl_ge_std",     "Cell line kernel gene expression (standardised)"),
    (kernel_cl_cnv,         "kernel_cl_cnv",        "Cell line kernel copy number variations"),
    (kernel_cl_cnv_std,     "kernel_cl_cnv_std",    "Cell line kernel copy number variations (standardised)"),
    (kernel_cl_mutation,    "kernel_cl_mutation",   "Cell line kernel mutations")
]
bins, dpi = 100, 300
for kernel,plot_name,plot_title in kernel_plots:
    matrix = kernel.kernel
    values = matrix[~numpy.isnan(matrix)]
    plt.figure()    
    plt.hist(values,bins=bins)
    plt.title(plot_title)
    plt.show()
    plt.savefig(folder_plots+plot_name,dpi=dpi)