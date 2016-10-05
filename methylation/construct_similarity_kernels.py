"""
Use the gene expression and two methylation datasets to construct similarity
kernels between the samples.

We use only the cancer driver genes, and standardise the values per gene.
We use a Gaussian kernel, using sigma^2 = no. genes / 4.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from load_methylation import filter_driver_genes_std
from HMF.code.kernels.gaussian_kernel import GaussianKernel

''' Load in the datasets. '''
R_ge, R_pm, R_gm, genes, samples = filter_driver_genes_std()
R_ge, R_pm, R_gm = R_ge.T, R_pm.T, R_gm.T

''' Construct the kernels. '''
no_points, no_genes = len(samples), len(genes)
sigma_2 = no_genes

kernel_ge = GaussianKernel()
kernel_ge.features, kernel_ge.no_points = R_ge, no_points
kernel_ge.construct_kernel(sigma_2=sigma_2)

kernel_gm = GaussianKernel()
kernel_gm.features, kernel_gm.no_points = R_gm, no_points
kernel_gm.construct_kernel(sigma_2=sigma_2)

kernel_pm = GaussianKernel()
kernel_pm.features, kernel_pm.no_points = R_pm, no_points
kernel_pm.construct_kernel(sigma_2=sigma_2)

''' Store the kernels. '''
output_folder = project_location+"HMF/methylation/data/"

kernel_ge.store_kernel(location_output=output_folder+"kernel_ge_std")
kernel_gm.store_kernel(location_output=output_folder+"kernel_gm_std")
kernel_pm.store_kernel(location_output=output_folder+"kernel_pm_std")