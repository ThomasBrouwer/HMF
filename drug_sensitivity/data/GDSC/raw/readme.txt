Cell Line 
- Cell line name 

Cosmic_ID 
- Unique cell line identifier 

Tissue
- Tissue type of cell line 

Cancer-type 
- Cancer sub-type of cell line based on tissue and histology 

Genetic information 
- Genetic mutation data for cancer genes. Includes MSI status (1 = unstable and 0 = stable) and gene-fusions. A binary code 'x::y' description is used for each gene where 'x' identifies a coding variant and 'y' indicates copy number information from SNP6.0 data. For gene fusions, cell lines are identified as fusion not-detected (0) or the identified fusion is given. The following abbreviations are used: not analysed (na), not detected or wild-type (wt), no copy number information (nci). 

IC50 values for each drug 
- Half maximal inhibitory (IC50) drug concentrations (natural log microMolar) 

Drug response curve features for each drug 
- ALPHA - sharp parameter from curve-fitting 
- BETA - slope parameter from curve-fitting 
- B - variance parameter from curve-fitting 
- IC_25 - 25% inhibitory drug concentration (natural log microMolar) 
- IC_50 - Half maximal inhibitory (50%) drug concentrations (natural log microMolar) 
- IC_75 - 75% inhibitory drug concentration (natural log microMolar) 
- IC_90 - 90% inhibitory drug concentration (natural log microMolar) 
- AUC - area under curve 
- D - residuals from curve fitting of data 
- IC_RESULTS_ID - unique result identifier 

IC50 value including confidence intervals for each drug 
- IC_50_LOW - IC50 low confidence interval (natural log microMolar) 
- IC_50 - Half maximal inhibitory (50%) drug concentrations (natural log microMolar) 
- IC_50_HIGH - IC50 high confidence interval (natural log microMolar)
