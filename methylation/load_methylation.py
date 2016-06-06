''' 
Methods for loading methylation and gene expression datasets, and for filtering
out the top n genes with the most variation.
'''

import numpy

location_methylation = '/home/tab43/Documents/Projects/libraries/HMF/methylation/data/'
name_gene_expression = 'matched_expression'
name_promoter_methylation = 'matched_methylation_genePromoter'
name_gene_body_methylation = 'matched_methylation_geneBody'

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