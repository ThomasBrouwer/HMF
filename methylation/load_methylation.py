''' 
Methods for loading methylation and gene expression datasets, and for filtering
out the top n genes with the most variation.
'''

import numpy

location_methylation = '/home/tab43/Documents/Projects/libraries/HMF/methylation/data/'
name_gene_expression = 'matched_expression'
name_promoter_methylation = 'matched_methylation_genePromoter'
name_gene_body_methylation = 'matched_methylation_geneBody'
name_driver_genes = 'intogen-BRCA-drivers-data.geneid'

def load_dataset(filename,delim='\t'):
    ''' Return a tuple (values,genes,samples) - numpy array, row names, column names. '''
    data = numpy.array([line.split('\n')[0].split(delim) for line in open(filename,'r').readlines()],dtype=str)
    sample_names = data[0,1:]
    gene_names = data[1:,0]
    values = numpy.array(data[1:,1:],dtype=float)
    return (values,gene_names,sample_names)
    
def load_gene_expression(filename=location_methylation+name_gene_expression,delim='\t'):
    return load_dataset(filename,delim)
    
def load_promoter_methylation(filename=location_methylation+name_promoter_methylation,delim='\t'):
    return load_dataset(filename,delim)
    
def load_gene_body_methylation(filename=location_methylation+name_gene_body_methylation,delim='\t'):
    return load_dataset(filename,delim)
    
def compute_gene_ranks(R):
    ''' Return a list of ranks for the genes. 
        First argsort gives index of 1st, 2nd, ... lowest variance. Second
        argsort then gives rank of 1st gene, rank of 2nd gene, etc. '''
    no_genes,_ = R.shape
    return no_genes - numpy.array(R.var(axis=1).argsort()).argsort()
    
def return_top_n_genes(Rs,n):
    ''' Given a list Rs = [R1,R2,..] of gene-patient datasets, and the number
        of genes n we want to use, return the same datasets but using only
        the top n with the highest overall variance.
        We compute the variance rank for each dataset, sum them up, and take 
        the lowest n. '''
    no_genes, no_samples = Rs[0].shape
    ranks = numpy.zeros(no_genes)
    for R in Rs:
        ranks_R = compute_gene_ranks(R)
        ranks = numpy.add(ranks,ranks_R)
        
    indices_top_n = ranks.argsort()[0:n]
    new_Rs = [R[indices_top_n,:] for R in Rs]
    return new_Rs
    
def load_all():
    ''' Return tuple (R_ge, R_pm, R_gm, genes, samples). '''
    R_ge, genes_ge, samples_ge = load_gene_expression()
    R_pm, genes_pm, samples_pm = load_promoter_methylation()
    R_gm, genes_gm, samples_gm = load_gene_body_methylation()
    assert numpy.array_equal(genes_ge,genes_pm) and numpy.array_equal(genes_ge,genes_gm)
    assert numpy.array_equal(samples_ge,samples_pm) and numpy.array_equal(samples_ge,samples_gm)
    return (R_ge, R_pm, R_gm, genes_ge, samples_ge)
    
def load_all_top_n_genes(n):
    ''' Return tuple (R_ge, R_pm, R_gm, genes, samples) with n most variant genes. '''
    (R_ge, R_pm, R_gm, genes, samples) = load_all()
    R_ge_n, R_pm_n, R_gm_n = tuple(return_top_n_genes([R_ge,R_pm,R_gm],n))
    return (R_ge_n, R_pm_n, R_gm_n, genes, samples)
    
def load_ge_pm_top_n_genes(n):
    ''' Return tuple (R_ge, R_pm, genes, samples) with n most variant genes. '''
    (R_ge, R_pm, _, genes, samples) = load_all()
    R_ge_n, R_pm_n = tuple(return_top_n_genes([R_ge,R_pm],n))
    return (R_ge_n, R_pm_n, genes, samples)
    
def filter_driver_genes():
    ''' Load the three datasets with only the driver genes, returning 
        (R_ge, R_pm, R_gm, genes, samples). '''
    (R_ge, R_pm, R_gm, all_genes, samples) = load_all()
    driver_gene_names = [line.split("\n")[0] for line in open(location_methylation+name_driver_genes,'r').readlines()]
    all_genes = list(all_genes)
    
    genes_in_overlap = [gene for gene in driver_gene_names if gene in all_genes]
    genes_not_in_overlap = [gene for gene in driver_gene_names if not gene in all_genes]
    print "Selecting %s driver genes. %s driver genes are not in the methylation data." % \
        (len(genes_in_overlap),len(genes_not_in_overlap))
    
    driver_gene_indices = [all_genes.index(gene) for gene in genes_in_overlap]
    (R_ge, R_pm, R_gm) = (R_ge[driver_gene_indices,:], R_pm[driver_gene_indices,:], R_gm[driver_gene_indices,:])
    return (R_ge, R_pm, R_gm, genes_in_overlap, samples)
    
def filter_driver_genes_std():
    ''' Load the three datasets with only the driver genes and standardise values
        per gene, returning (R_ge_std, R_pm_std, R_gm_std, genes, samples). '''
    (R_ge, R_pm, R_gm, genes, samples) = filter_driver_genes()
    R_ge_std = standardise_data(R_ge)
    R_pm_std = standardise_data(R_pm)
    R_gm_std = standardise_data(R_gm)
    return (R_ge_std, R_pm_std, R_gm_std, genes, samples)
    
def standardise_data(R):
    ''' Transform R to have row-wise mean 0 and std 1. '''
    R = numpy.array(R)
    I,J = R.shape
    mean = R.mean(axis=1)
    std = R.std(axis=1)
    R_std = numpy.copy(R)
    for j in range(0,J):
        R_std[:,j] = (R_std[:,j] - mean) / std
    return R_std
    
(R_ge, R_pm, R_gm, genes, samples) = load_all()