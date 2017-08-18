''' 
Methods for loading methylation and gene expression datasets, and for filtering
out the top n genes with the most variation.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/" # "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from HMF.code.kernels.gaussian_kernel import GaussianKernel

from collections import Counter
import numpy


location_methylation = '/Users/thomasbrouwer/Documents/Projects/libraries/HMF/methylation/data/' # '/home/tab43/Documents/Projects/libraries/HMF/methylation/data/'
name_gene_expression = 'matched_expression'
name_promoter_methylation = 'matched_methylation_genePromoter'
name_gene_body_methylation = 'matched_methylation_geneBody'
name_driver_genes = 'intogen-BRCA-drivers-data.geneid'
name_labels = 'matched_sample_label'

location_gene_annotation = location_methylation+'gene_classes/'
name_gene_ids = 'gene_ids'
name_gene_go = 'gene_ontologies'


""" Methods for loading gene expression and methylation datasets. """
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
    
    
""" Method for loading similarity kernels. """
def load_kernels():
    ''' Load similarity kernels samples. '''
    kernel_folder = project_location+"HMF/methylation/data/"
    K_ge, K_pm, K_gm = GaussianKernel(), GaussianKernel(), GaussianKernel()
    K_ge.load_kernel(location_input=kernel_folder+"kernel_ge_std_samples")
    K_pm.load_kernel(location_input=kernel_folder+"kernel_pm_std_samples")
    K_gm.load_kernel(location_input=kernel_folder+"kernel_gm_std_samples")
    return (K_ge.kernel, K_pm.kernel, K_gm.kernel)
    
def load_kernels_genes():
    ''' Load similarity kernels genes. '''
    kernel_folder = project_location+"HMF/methylation/data/"
    K_ge, K_pm, K_gm = GaussianKernel(), GaussianKernel(), GaussianKernel()
    K_ge.load_kernel(location_input=kernel_folder+"kernel_ge_std_genes")
    K_pm.load_kernel(location_input=kernel_folder+"kernel_pm_std_genes")
    K_gm.load_kernel(location_input=kernel_folder+"kernel_gm_std_genes")
    return (K_ge.kernel, K_pm.kernel, K_gm.kernel)
        
        
""" Methods for loading healthy / tumour labels. """
def load_labels(filename=location_methylation+name_labels,delim='\t'):
    ''' Return a list of 'tumor' or 'normal' labels. '''
    data = numpy.array([line.split('\n')[0].split(delim) for line in open(filename,'r').readlines()],dtype=str)
    return data[:,1]
    
def load_label_matrix():
    ''' Return a <no_samples> by 2 matrix, with the first column containing 1
        if the sample is healthy (0 in second column), and vice versa for a 
        tumour tissue. 
    '''
    labels = load_labels()
    label_matrix = numpy.zeros((len(labels),2))
    for i,label in enumerate(labels):
        if label == 'tumor':
            label_matrix[i,1] = 1
        else:
            label_matrix[i,0] = 1
    return label_matrix

def load_tumor_label_list():
    ''' Return a list of labels, with 1 for tumor and 0 for healthy. '''
    return load_label_matrix()[:,1]

def load_healthy_label_list():
    ''' Return a list of labels, with 1 for healthy and 0 for tumor. '''
    return load_label_matrix()[:,0]
    
    
""" Methods for reordering rows and columns of datasets. """
def reorder_rows(R, old_rows, new_rows):
    ''' Reorder the rows of R, with old ordering old_rows and new ordering new_rows. '''
    mapping_name_to_values = {
        old_name: R[i]
        for i,old_name in enumerate(old_rows)
    }
    new_R = numpy.array([
        mapping_name_to_values[new_name]
        for new_name in new_rows
    ])
    return new_R
    
def reorder_rows_columns(R, old_rows, new_rows, old_columns, new_columns):
    ''' Reorder both the row and the column entries. '''
    R_rows = reorder_rows(R, old_rows, new_rows)
    R_rows_and_columns = reorder_rows(R_rows.T, old_columns, new_columns).T
    return R_rows_and_columns
    
def reorder_samples_label(R, samples):
    ''' Reorder the columns (samples) so that the healthy samples come first. 
        Return (R, sorted_samples, sorted_labels_tumours).    
    '''
    labels_tumour = load_tumor_label_list()
    sorted_samples_labels = sorted([(name,labels_tumour[index]) for index,name in enumerate(samples)], key=lambda x:(x[1], x[0]))
    sorted_samples = [v[0] for v in sorted_samples_labels]
    sorted_labels_tumour = [v[1] for v in sorted_samples_labels]
    R_sorted = reorder_rows(R.T, samples, sorted_samples).T
    return (R_sorted, sorted_samples, sorted_labels_tumour)
    
def load_driver_genes_std_tumour_labels_reordered():
    ''' Load the three datasets with only the driver genes and standardise values
        per gene. Also load the tumour labels, and reorder samples to have tumours
        next to each other, genes to have TODO.
        Return (R_ge_std, R_pm_std, R_gm_std, tumour_labels, genes, samples).
    '''
    R_ge, R_pm, R_gm, genes, samples = filter_driver_genes_std()
    
    # Figure out reordering samples and genes
    (_, sorted_samples, sorted_labels_tumour) = reorder_samples_label(R_ge, samples)
    #TODO: genes reordering
    
    # Do the reordering and return the data    
    R_ge_reordered = reorder_rows_columns(R_ge, genes, genes, samples, sorted_samples)
    R_pm_reordered = reorder_rows_columns(R_pm, genes, genes, samples, sorted_samples)
    R_gm_reordered = reorder_rows_columns(R_gm, genes, genes, samples, sorted_samples)
    
    return (R_ge_reordered, R_pm_reordered, R_gm_reordered, sorted_labels_tumour, genes, sorted_samples)
    

''' Method for loading gene annotations (toplevel GO terms). '''
def load_GO_term(genes=[], go_terms=[]):
    ''' Return a list of the toplevel GO id for each gene, in the order of genes 
        specified by :genes (or, if it is [], in the original order). If :go_terms 
        is not [], replace any GO id not in that list by 'other'. 
    '''
    genes_in_order = [line.split('\n')[0] for line in open(location_gene_annotation+name_gene_ids,'r').readlines()]    
    go_in_order = [line.split('\n')[0].split('\t')[0] for line in open(location_gene_annotation+name_gene_go,'r').readlines()]    

    # Replace any GO ids not in :go_terms by 'other'    
    if go_terms:
        for i,go_term in enumerate(go_in_order):
            if go_term not in go_terms or go_term == '(no value)':
                go_in_order[i] = 'other'
        
    # Reorder genes to order specified by :genes
    if genes:
        go_reordered = [go_in_order[genes_in_order.index(gene)] for gene in genes]
        return go_reordered
    else:
        return go_in_order

def load_top_n_GO_terms(n, genes=[]):
    ''' Same as above but using the top n GO terms for the 160 driver genes. '''
    # First count the occurrences for each GO term id and sort them
    go_terms = load_GO_term(genes=genes)
    c = Counter(go_terms)
    
    # Then extract the top n GO term ids    
    top_n_go_terms_and_counts = c.most_common(n)
    top_n_go_terms = [v[0] for v in top_n_go_terms_and_counts]
    
    # Finally, return the list of GO terms associated with the genes (replacing uncommon ones with 'other')
    return load_GO_term(genes=genes, go_terms=top_n_go_terms)

def load_top_n_GO_terms_as_rank(n, genes=[]):
    ''' Same as above but return the rank (1, 2, ..., n+1) of each GO term sorted by frequency. '''
    # First count the occurrences for each GO term id and sort them
    go_terms = load_GO_term(genes=genes)
    c = Counter(go_terms)
    
    # Extract the top n GO term ids    
    top_n_go_terms_and_counts = c.most_common(n)
    top_n_go_terms = [v[0] for v in top_n_go_terms_and_counts]
    
    # Replace GO terms by its ranking. 'other' has ranking n+1
    rank_go_terms = []    
    for go_term in go_terms:
        rank_go_terms.append(n+1 if go_term not in top_n_go_terms else top_n_go_terms.index(go_term)+1)
    return rank_go_terms
    

#(R_ge, R_pm, R_gm, genes, samples) = filter_driver_genes()
#(R_ge_std, R_pm_std, R_gm_std, genes_std, samples_std) = filter_driver_genes_std()
#K_ge, K_pm, K_gm = load_kernels()

#(R_ge_reordered, R_pm_reordered, R_gm_reordered, labels_tumour, genes, sorted_samples) = load_driver_genes_std_tumour_labels_reordered()

#genes_reordered = ['207', '7157', '23451', '3845', '2625', '4214', '999', '9611', '6416', '861', '4763', '5925', '8289', '6926', '8085', '865', '10664', '1108', '5295', '10735', '841', '3169', '4602', '677', '23266', '1027', '79718', '675', '288', '2064', '4627', '9439', '6103', '3054', '79784', '4853', '6709', '55749', '5984', '790', '57522', '91', '2874', '5590', '55294', '673', '4893', '3417', '29072', '1956', '8242', '196528', '23152', '55193', '9817', '6794', '84295', '8880', '1964', '23499', '79026', '9968', '10142', '79987', '259266', '545', '4297', '2475', '55870', '64324', '51755', '4629', '7204', '9869', '8997', '1974', '23269', '2122', '2909', '4928', '10274', '6597', '54880', '10076', '2322', '10564', '2186', '27086', '9862', '7812', '2033', '2335', '54796', '80205', '5291', '4040', '6654', '54510', '1654', '27125', '10043', '2068', '3312', '57122', '55914', '659', '1213', '4851', '23512', '3105', '4849', '6655', '3131', '1665', '1981', '48', '3936', '8394', '4929', '57591', '8454', '3842', '2521', '63967', '6789', '5931', '6421', '1997', '4149', '192669', '406', '9314', '9908', '10498', '6934', '9871', '831', '23122', '23305', '200958', '4771', '6453', '10181', '9716', '4481', '9967', '9950', '60', '6009', '466', '471', '5046', '7048', '10963', '5079', '7027', '5742', '10397', '8503', '1456']
#go_terms = load_GO_term(go_terms=['GO:0000122', 'GO:0000165', 'GO:0006351'])
#go_terms_reordered = load_GO_term(genes=genes_reordered, go_terms=['GO:0000122', 'GO:0000165', 'GO:0006351'])
#top_go_terms = load_top_n_GO_terms(n=3)
#top_go_terms_reordered = load_top_n_GO_terms(n=3, genes=genes_reordered)
#top_go_term_ranks = load_top_n_GO_terms_as_rank(n=3)
#top_go_term_ranks_reordered = load_top_n_GO_terms_as_rank(n=3, genes=genes_reordered)