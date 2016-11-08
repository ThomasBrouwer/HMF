# Bayesian hybrid matrix factorisation for data integration
## Python implementation of the Hybrid Matrix Factorisation model.

### Folder structure

- **/code/** - Code for the HMF model, as well as other matrix factorisation models, helper methods, and more.
  - **/Gibbs/**
    - **draws_Gibbs.py** - Code for drawing new values for matrix factorisation models using Gibbs sampling (given the parameter values).
    - **init_Gibbs.py** - Code implementing the different initialisation approaches for the Gibbs sampling parameters (random, expectation, K-means, least squares).
    - **updates_Gibbs.py** - Code for computing the Gibbs sampling posterior parameter updates.
  - **/models/** - Classes that implement the actual matrix factorisation models - NMF, NMTF, BNMF, BNMTF, HMF. These use the /Gibbs/ files.
  - **/distributions/** - Python classes that handle computing the mean and variance, and drawing samples, from the probability distributions used in the models.
  - **/kmeans/** - Implementation of K-means clustering when the matrix is only partially observed.
  - **/kernels/** - Classes for computing, storing, and loading, similarity kernels (Jaccard and Gaussian).
  - **/statistics/** - Python code for computing the prediction errors of a list or matrix of values (MSE, R^2, Rp).
  - **/model_selection/** - Classes that help with heuristic model selection of the BNMF and BNMTF models (using line search, grid search, or greedy search - see NIPS workshop paper).
  - **/cross_validation/** - Python scripts for performing in-matrix cross-validation experiments for non-probabilistic and Bayesian matrix factorisation models, and HMF. Some scripts also allow you to do model selection (either nested cross-validation, or using heuristics from NIPS workshop paper).
  - **/generate_mask/** - Methods for helping generate observation matrices (1 indicating observed entries, 0 indicating unobserved).
  - **/generate_toy/** - Method for generating toy/synthetic data for the HMF model.

- **/drug_sensitivity/** - Data and code for drug sensitivity data integration experiments.
  - **/data/** - Data for drug sensitivity application (GDSC, CTRP, CCLE IC50, CCLE EC50). See description.txt for more details.
  - **/cross_validation_in_matrix/** - Cross-validation scripts and results for the different methods, for in-matrix predictions.
  - **/varying_init/** - Exploration of different initialisation methods for HMF D-MF and HMF D-MTF.
  - **/varying_K/** - Exploration of effectiveness of ARD, by trying the HMF D-MF and HMF D-MTF models with different values of K, on the four drug sensitivity datasets.
  - **/varying_sparsity/** - Experiments measuring the performance of the matrix factorisation methods (NMF, NMTF, BNMF, BNMTF, HMF D-MF, HMF D-MTF) as the sparsity of the GDSC and CTRP drug sensitivity datasets increases.
  - **/varying_negativity/** - Exploration of the trade-offs between the different negativity constraints (nonnegative, semi-nonnegative, real-valued), on the drug sensitivity datasets.
  - **/varying_negativity_sparsity/** - Exploration of the trade-offs between the different negativity constraints (nonnegative, semi-nonnegative, real-valued), on the CTRP drug sensitivity dataset, when the sparsity of the datasets increases.
  - **/bicluster_analysis/** - Analysis of biclusters of running HMF D-MTF on the four drug sensitivity datasets.

- **/methylation/** - Data and code for gene expression and methylation data integration experiments.
  - **/data/**
    - **/data_plots/** - Visualisation of the data sources (raw, standardised, and similarity kernels).
    - **intogen-BRCA-drivers-data.geneid** - list of Entrez Gene ID's for the genes (in order of other datasets).
    - **matched_sample_label** - list of sample names, and whether they are tumour or healthy (in order of other datasets).
    - **matched_expression** - gene expression data (first row is sample names, first row is gene ids).
    - **matched_methylation_geneBody** - gene body methylation data (first row is sample names, first row is gene ids).
    - **matched_methylation_genePromoter** - promoter-region methylation data (first row is sample names, first row is gene ids).
  - **/convergence/** - Script for running HMF D-MTF on the gene expression and promoter-region methylation data, to check convergence.
  - **/cross_validation/** - Cross-validation scripts and results for the different methods, for out-of-matrix predictions.
  - **/varying_negativity/** - Exploration of the trade-offs between the different negativity constraints (nonnegative, semi-nonnegative, real-valued), on the gene expression and methylation datasets.
  - **/bicluster_analysis/** - Analysis of biclusters of running HMF D-MTF on the gene expression and methylation datasets.

- **/tests/** - Unit tests (py.test) for a large portion of the implementations in /code/.

- **/toy_experiments/** - Very brief tests on the toy dataset, mainly for checking convergence of the model.

