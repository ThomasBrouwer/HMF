This folder contains different gene classifications, so we can visualise the biclusters better and see whether we pick up any biological signals using HMF D-MTF.

Resources:
- MyGene.info (http://mygene.info/) provides an API (including Python library) to obtain info about genes. Instructions: http://docs.mygene.info/en/latest/doc/packages.html


Files:
- convert_gene_ids.py
Python script for converting the gene ids into: names, symbols, ...
- gene_ids
Contains the gene ids (Entrez gene ID, or UID), in the order of load_methylation.filter_driver_genes_std().
- gene_names
Contains the gene names, in order of gene_ids.
- gene_symbols
Contains the gene symbols, in order of gene_ids.
- gene_summaries
Short summary per gene. For some genes there is no summary.
- gene_ontologies
Gene ontology per gene. We use the biological process GO per gene, and then take the most toplevel GO out of the GO terms associated to the gene (this is the first one, with the lowest GO id). Only 3 of the 160 genes have no BP GO term.
This file has the format: <GO_id>\t<GO_description>\n.

Count of unique GO terms:
Counter({u'GO:0000122': 27, u'GO:0000165': 10, u'GO:0006351': 9, u'GO:0000398': 5, u'GO:0006355': 5, u'GO:0006366': 3, u'GO:0000245': 2, u'GO:0000082': 2, u'GO:0001707': 2, u'GO:0000226': 2, u'GO:0006886': 2, u'GO:0007156': 2, u'GO:0001843': 2, u'GO:0000209': 2, u'GO:0000722': 2, u'GO:0001701': 2, u'GO:0000059': 2, u'GO:0001764': 2, u'GO:0000187': 2, u'GO:0001525': 2, u'GO:0007264': 2, u'GO:0000077': 2, u'GO:0000086': 1, u'GO:0006470': 1, u'GO:0001503': 1, u'GO:0007165': 1, u'GO:0007166': 1, u'GO:0002903': 1, u'GO:0006306': 1, u'GO:0006661': 1, u'GO:0003416': 1, u'GO:0006936': 1, u'GO:0006605': 1, u'GO:0001516': 1, u'GO:0003009': 1, u'GO:0002674': 1, u'GO:0010632': 1, u'GO:0000717': 1, u'GO:0006337': 1, u'GO:0000301': 1, u'GO:0007062': 1, u'GO:0000381': 1, u'GO:0000902': 1, u'GO:0007253': 1, u'GO:0007050': 1, u'GO:0001776': 1, u'GO:0001678': 1, u'GO:0001676': 1, u'GO:0007354': 1, u'GO:0006898': 1, u'GO:0000973': 1, u'GO:0006357': 1, u'GO:0001709': 1, u'GO:0001662': 1, u'GO:0000288': 1, u'GO:0000289': 1, u'GO:0043484': 1, u'GO:0000281': 1, u'GO:0006508': 1, u'GO:0008285': 1, u'GO:0000212': 1, u'GO:0001558': 1, u'GO:0001895': 1, u'GO:0001555': 1, u'GO:0006338': 1, u'GO:0000050': 1, u'GO:0001817': 1, u'GO:0000184': 1, u'GO:0006650': 1, u'GO:0006099': 1, u'GO:0006097': 1, u'GO:0001649': 1, u'GO:0002474': 1, u'GO:0003360': 1, u'GO:0001881': 1, u'GO:0000060': 1, u'GO:0006402': 1, u'GO:0006468': 1, u'GO:0006915': 1, u'GO:0006464': 1, u'GO:0010951': 1, u'GO:0002286': 1, u'GO:0000075': 1, u'GO:0000079': 1, u'GO:0003420': 1, u'GO:0001933': 1, u'GO:0001568': 1, u'GO:0006397': 1})
