'''
Python script for converting the gene ids into: names, symbols, summary, GO.

Interesting fields for gene annotation:
- go (gene ontology)
    This returns a dictionary with three fields: 
    'BP' = Biological process; 'CC' = Cellular component; 'MF' = Molecular function.
    We probably only want BP.

Uninteresting fields:
- type_of_gene (all genes are 'protein-coding', so not interesting)
'''

import mygene

mg = mygene.MyGeneInfo()


''' File locations. '''
file_gene_ids = 'gene_ids'
file_gene_names = 'gene_names'
file_gene_symbols = 'gene_symbols'
file_gene_summaries = 'gene_summaries'
file_gene_ontologies = 'gene_ontologies'


''' Load the gene ids. '''
gene_ids = [line.split('\n')[0] for line in open(file_gene_ids,'r').readlines()]


''' For each, extract the name and symbol and store in a file. '''
all_fout = [
    open(file_gene_names, 'w'),
    open(file_gene_symbols, 'w'),
    open(file_gene_summaries, 'w'),
    open(file_gene_ontologies, 'w'),
]
all_field_names = ['name', 'symbol', 'summary', 'go', ]

info_to_get = ','.join(all_field_names) #'all'
gene_infos = mg.getgenes(gene_ids, info_to_get)

all_go_ids = []

for gene_info in gene_infos:
    for fout, field_name in zip(all_fout, all_field_names):
        try:
            value = gene_info[field_name]
            
            # For GO, extract the first BP term (lowest GO id, and hence most general)     
            if field_name == 'go':
                bp_go = value['BP']
                toplevel_go = bp_go[0]
                fout.write(toplevel_go['id']+'\t'+toplevel_go['term']+'\n')
                all_go_ids.append(toplevel_go['id'])
            else:
                fout.write(value+'\n')
        except KeyError:
            fout.write('(no value)\n')
        
for fout in all_fout:
    fout.close()