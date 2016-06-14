"""
Plot the overlaps of the observed entries in the four matrices.
We do this using the following website:
http://bioinfogp.cnb.csic.es/tools/venny/
"""

import numpy, itertools

ccle_ic50 = numpy.loadtxt('data_features/ccle_ic50_features.txt',delimiter="\t")
ccle_ec50 = numpy.loadtxt('data_features/ccle_ec50_features.txt',delimiter="\t")
ctrp_ic50 = numpy.loadtxt('data_features/ctrp_ec50_features.txt',delimiter="\t")
gdsc_ic50 = numpy.loadtxt('data_features/gdsc_ic50_features.txt',delimiter="\t")
I,J = gdsc_ic50.shape

indices_ccle_ic50 = [(i,j) for (i,j) in itertools.product(range(0,I),range(0,J)) if not numpy.isnan(ccle_ic50[i,j])]
indices_ccle_ec50 = [(i,j) for (i,j) in itertools.product(range(0,I),range(0,J)) if not numpy.isnan(ccle_ec50[i,j])]
indices_ctrp_ic50 = [(i,j) for (i,j) in itertools.product(range(0,I),range(0,J)) if not numpy.isnan(ctrp_ic50[i,j])]
indices_gdsc_ic50 = [(i,j) for (i,j) in itertools.product(range(0,I),range(0,J)) if not numpy.isnan(gdsc_ic50[i,j])]

'''
299+880+35+54+45+734+317+446+292+9343+596 = 13041 entries have 2+ datasets, so 
fraction 13041/float(399*52) = 0.6285425101214575.
'''
