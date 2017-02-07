Folder containing the drug sensitivity datasets used for the data integration model for predicting drug sensitivity values.

We consider the following datasets:
- Sanger GDSC (*Genomics of Drug Sensitivity in Cancer*)

  IC50 values. Number drugs: 139. Number cell lines: 707. Number of observed entries: 79262. Fraction observed: 0.806549103009
  
  http://www.cancerrxgene.org/downloads/
  
- CCLE (*Cancer Cell Line Encyclopedia*)
  IC50 and EC50 values. Number drugs: 24. Number cell lines: 504. Number of observed entries IC50 / EC50: 11670 / 7626. Fraction observed IC50 / EC50: 0.964781746032 / 0.630456349206.
  
  http://www.broadinstitute.org/ccle
  
  (if website is down, use https://cghub.ucsc.edu/datasets/ccle.html)
  
- CTRP (Cancer Therapeutics Response Portal)

  EC50 values. Number drugs: 545. Number cell lines: 887. Number of observed entries: 387130. Fraction observed: 0.800823309165. 
  
  http://www.broadinstitute.org/ctrp/?page=#ctd2BodyHome
  
  Download from https://ctd2.nci.nih.gov/dataPortal/	


Difference IC50 and EC50 (from http://www.fda.gov/ohrms/dockets/ac/00/slides/3621s1d/sld036.htm):
"The IC50 represents the concentration of a drug that is required for 50% inhibition of viral replication in vitro (can be corrected for protein binding etc.).
The EC50 represents the plasma concentration/AUC required for obtaining 50% of the maximum effect in vivo."
(In vitro = in controlled environment outside living organism, in vivo = experimentation using a whole, living organism)

The datasets are stored in **/GDSC/**, **/CTRP/**, and **/CCLE/**. Each folder contains a **/raw/** folder containing the raw datasets, as downloaded, and a Python/numpy-friendly version in **/processed_all/**, created by the Python script **process_all_gdsc/ctrp/ccle.py**.

We compute the overlap of the four datasets, as described in the paper and supplementary materials, using **construct_datasets_overlap.py**, placing the files in **/overlap/**. The script **print_statistics_overlap.py** gives some statistics about the overlaps between the datasets, and **print_statistics_correlation.py** the correlation between overlapping values between the datasets. We also constructed two Venn diagrams for the overlapping drugs and cell lines, using **find_overlap.py**.

After computing the overlap between the datasets, we conducted further preprocessing in **/overlap/**, again as described in the paper and supplementary materials. **extract_features_cell_lines.py** extracts the gene expression, CNV and mutation features for the cell lines from features_cell_lines/gdsc_en_input_w5.csv. Also store version of IC50, EC50 datasets using only cell lines with features. **extract_features_drugs.py** extract the drug target features from GDSC for 48 of our drugs, and make all entries NaN for the other 4. Also remove features (targets, 1D2D, fingerprints) with same value for all drugs, and store list of remaining target names. **process_data.py** preprocesses the IC50 and EC50 values: cuts off values too high/low and plot the distribution of values. Creates row- and column-standardised datasets, and [0,1] per row/column as well. **plot_overlap_observed.py** makes a Venn diagram of the overlaps between observed entries in the dataset of 52 drugs and 399 cell lines. **construct_kernels.py** constructs the similarity kernels from the feature datasets, and plots their distributions.

Preprocessing order: process_all_gdsc.py, process_all_ctrp.py, process_all_ccle.py, find_overlap.py, construct_datasets_overlap.py, extract_features_cell_lines.py, extract_features_drugs.py, process_data.py, construct_kernels.py.

Below more details are given about each folder in **/overlap/**:
- **/data_all/** - Drug sensitivity datasets containing overlap of the four datasets.
- **/data_features/** - Drug sensitivity datasets containing overlap of four datasets, but only the cell lines with features in GDSC.
  - **gdsc_ic50_features.txt**, **ctrp_ec50_features.txt**, **ccle_ic50_features.txt**, **ccle_ec50_features.txt**
  - **gdsc_ic50_features_exp.txt** - exponential transform of IC50 values to undo log transform and make it the same as other datasets.
- **/data_capped/** - The CTRP EC50 and GDSC IC50 datasets have some extremely high values (due to the way drug sensitivity datasets are made), so we cap them at 20. We use the GDSC values with the exponential transform.
  - **gdsc_ic50_capped.txt**, **ctrp_ec50_capped.txt**, **ccle_ic50.txt**, **ccle_ec50.txt**
- **/data_row_01/** - We map the entries in each row (cell line) to the range [0,1] for the datasets in /data_capped/.
  - **gdsc_ic50_row_01.txt**, **ctrp_ec50_row_01.txt**, **ccle_ic50_row_01.txt**, **ccle_ec50_row_01.txt**
- **/data_column_01/** - We map the entries in each column (drug) to the range [0,1] for the datasets in /data_capped/.
  - **gdsc_ic50_col_01.txt**, **ctrp_ec50_col_01.txt**, **ccle_ic50_col_01.txt**, **ccle_ec50_col_01.txt**
- **/features_drugs/** - Feature datasets for the drugs.
  - **drugs.txt** - List of drugs in overlap (in order of datasets) - only normalised name (lowercase, removed dash, space, comma, full stop). drugs_full.txt gives the full drug name (unnormalised).
  - **drugs_full.txt** - Full drug names of the drugs in drugs.txt.
  - **drug_pubchem_ids.txt** - PubChem compound ids for the drugs, using https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange.cgi (use the Synonyms option). We sometimes got multiple ids, in which case we took the first one.
  - **drug_smiles.smi** - SMILES structure codes for the drugs, using https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange.cgi.
  - **drug_fingerprints.csv** - Binary Pubchem fingerprints extracted using PaDeL from the SMILES codes, indicating presence of 881 functional groups. Default settings.
  - **drug_fingerprints.txt** - Same as above but just the numpy array as values, and only features with differing values. 495 descriptors.
  - **drug_1d2d.csv** - 1D and 2D descriptors extracted using PaDeL, default settings - varying very much in scale (some are counts, some are in the millions). Might not be a good feature dataset.
  - **drug_1d2d.txt** - Same as above but just the numpy array as values, and only features with differing values. 1160 descriptors.
  - **drug_1d2d.txt** - Standardised each row to have zero mean and unit variance, for better kernels.
  - **drug_targets_gdsc.csv** - Binary matrix of drug targets for all of the GDSC drugs (48/52 for our dataset - we mark the remaining 4 as unknown). First row is target names, first column is drug Pubchem ids. Drugs in overlap but not in GDSC: l685458, nutlin3 (nutlin3a?), raf265, topotecan (full names: L-685458, nutlin-3, RAF265, topotecan).
  - **drug_targets.txt** - Binary matrix of drug targets for our 52 drugs (NaN for 4 of them), and 53 targets.
  - **target_names.txt** - List of 53 target names.
- **/features_cell_lines/** - Feature datasets for the cell lines.
  - **cell_lines.txt** - List of cell lines in overlap (in order of datasets) - only normalised name. cell_lines_full.txt gives the full cell line name (unnormalised).
  - **gdsc_en_input_w5.csv** - Raw feature file from the GDSC dataset. First row gives the cell line names, first column gives the feature names - next 13321 rows are gene expression values, 426 after are copy number profiles, final 82 are cancer gene mutations.
  - **cell_lines_features.txt** - List of cell lines with feature values (gdsc_en_input_w5.csv).
  - **gene_expression.txt** - Gene expression features. Positive, real-valued. 13321 features. In order of cell_lines_features.txt.
  - **gene_expression_std.txt** - As above but with standardised features (zero mean unit variance).
  - **cnv.txt** - Copy number variation features. Positive count data. 426 features. In order of cell_lines_features.txt.
  - **cnv_std.txt** - As above but with standardised features (zero mean unit variance).
  - **mutation.txt** - Mutation features. Binary data. 82 features. In order of cell_lines_features.txt.
- **/similarity_kernels/** - The similarity kernels constructed from the feature datasets. For the Gaussian kernels, we use sigma^2 = D, where D is the number of features.
- **/plots_distributions/** - Plots of the values in each of the kernels.


###############################################################################################################################################################################

/GDSC/
	/raw/
	Original Sanger "Genomics of Drug Sensitivity in Cancer" datasets (nothing filtered). Contains 140 drugs and 707 cell lines.

		gdsc_manova_input_w5.csv
		The complete raw dataset from Sanger.

	/processed_all/
	Extracted the data from the raw data files and put it into a big matrix. There is one duplicate drug (AZD6482_IC_50) so we filter it.
	Number drugs: 139. Number cell lines: 707. Number of observed entries: 79262. Fraction observed: 0.806549103009.

		ic50.txt
		The IC50 values only for all drugs and cell lines. Rows are cell lines, columns are drugs.

		drugs.txt
		List of drugs (in order of ic50.txt) - normalised name, name.

		cell_lines.txt
		List of cell lines (in order of ic50.txt) - normalised name, name, COSMIC id, cancer type, tissue.

	/features/
	Features for the drugs and cell lines.

		gdsc_en_input_w5.csv
		Contains all the cell line features, including cell lines we do not have drug sensitivity data of. First row gives the cell line names, first column gives the feature names - first 13321 rows are gene expression values, next 426 are copy number profiles, final 82 are cancer gene mutations. Original file from website is called gdsc_en_input_w5.csv. 

###############################################################################################################################################################################

/CTRP/
	/raw/
	Original Cancer Therapeutic Response Portal datasets. Contains 481 compounds (70 DFA approved, 100 clinical candidates, 311 small-molecule probes) and 860 cancer cell lines.

		CTRPv2.0._COLUMNS.xlsx
		Descriptions of the columns in CTRPv2.0_2015_ctd2_ExpandedDataset.

		CTRPv2.0._INFORMER_SET.xlsx
		Descriptions of the drugs in CTRPv2.0_2015_ctd2_ExpandedDataset (including SMILES codes).

		CTRPv2.0._README.docx
		Descriptions of files in CTRPv2.0_2015_ctd2_ExpandedDataset.zip

		CTRPv2.0_2015_ctd2_ExpandedDataset.zip
		Zipped raw data.

		CTRPv2.0_2015_ctd2_ExpandedDataset
		Unzipped. Interesting files are:
		- v20.meta.per_experiment.txt
			Data about experiments (each experiment corresponds to one cell line)
		- v20.meta.per_compound.txt
			Data about drugs (id's, SMILES).
		- v20.meta.per_cell_line.txt
			Data about cell lines (id's, tissue and cancer types).
		- v20.data.curves_post_qc.txt
			Area-under-concentration-response curve (AUC) sensitivity scores, curve-fit parameters, and confidence intervals for each cancer cell line and each compound.
			experiment_id gives the experiment id, which can give the cell line name in v20.meta.per_experiment.txt.
			apparent_ec50_umol gives the EC50 value 
			master_cpd_id gives the drug id
		So need to extract the apparent_ec50_umol values, link up the experiment_id to the cell line id, and then make one big matrix.

	/processed_all/
	Extracted the data from the raw data files and put it into a big matrix. There are some VERY large values that we may want to filter out.
	Number drugs: 545. Number cell lines: 887. Number of observed entries: 387130. Fraction observed: 0.800823309165.

		ec50.txt
		The EC50 values only for all drugs and cell lines. Rows are cell lines, columns are drugs.

		drugs.txt
		List of drugs (in order of ic50.txt) - normalised name, name, id, SMILES code.

		cell_lines.txt
		List of cell lines (in order of ic50.txt) - normalised name, name, id, cancer type, tissue.

###############################################################################################################################################################################

/CCLE/
	/raw/
	Raw Cancer Cell Line Encyclopedia datasets. Contains 24 cancer drugs and 504 cell lines.

		CCLE_NP24.2009_Drug_data_2015.02.24.csv
		Drug sensitivity values as list of experiments - drug dose on a specific cell line, IC50, EC50.
		Should use this dataset. Extract both IC50 and EC50 and then use one. IC50 has lots of 8 values (where EC50 is normally NA).

		CCLE_NP24.2009_profiling_2012.02.20.csv
		Data about drugs - name, primary targets, manifacturer.

		CCLE_GNF_data_090613.xls
		Values of the 24 drugs on 504 cell lines. Some values are missing, NoFit, >20, or <0.000x. 
		This seems to be drug sensitivity but does not have the same values as stored in CCLE_NP24.2009_Drug_data_2015.02.24.csv.

	/processed_all/
	Extracted the data from the raw data files and put it into a big matrix.
	Number drugs: 24. Number cell lines: 504. Number of observed entries IC50 / EC50: 11670 / 7626. Fraction observed IC50 / EC50: 0.964781746032 / 0.630456349206.

		ec50.txt
		The EC50 values only for all drugs and cell lines. Rows are cell lines, columns are drugs.

		ic50.txt
		The IC50 values only for all drugs and cell lines. Rows are cell lines, columns are drugs.

		drugs.txt
		List of drugs (in order of ic50.txt) - normalised name, name, id, SMILES code.

		cell_lines.txt
		List of cell lines (in order of ic50.txt) - normalised name, name, id, cancer type, tissue.

###############################################################################################################################################################################


