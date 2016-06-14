"""
Extract the target features for the drugs from drug_targets_gdsc.csv.
First row gives the target names, first column gives the drug Pubchem ids.

Drugs in overlap but not in GDSC: l685458, nutlin3 (nutlin3a?), raf265, topotecan (full names: L-685458, nutlin-3, RAF265, topotecan).

Other drugs with unfound PubChem id (but still in GDSC): 11353973, 9826308, 46930998, 7251185.
We replaced these with the PubChem ids from GDSC: 10390396, 15983966, 24964624, 5420805.
"""

import numpy

''' Look up the Pubchem ids of the drugs we are interested in '''
drugs_data_pubchem_ids = list(numpy.loadtxt("features_drugs/drug_pubchem_ids.txt",dtype=str))

''' Extract the target features - for each of these, look up the drug targets.
    If there aren't any, set it to NaN. Also remove targets with all 0's or 1's. '''
filename = "features_drugs/drug_targets_gdsc.csv"
lines = [line.split("\n")[0].split("\t") for line in open(filename,'r').readlines()]
target_names = lines[0][1:]
drugs_pubchem_id_to_features = {}
for row in lines:
    drugs_pubchem_id_to_features[row[0]] = row[1:]
    
drug_targets = []
for pid in drugs_data_pubchem_ids:
    if pid in drugs_pubchem_id_to_features.keys():
        drug_targets.append(drugs_pubchem_id_to_features[pid])
    else:
        nan_row = numpy.empty(len(drug_targets[0]))
        nan_row[:] = numpy.NaN
        drug_targets.append(nan_row)
drug_targets = numpy.array(drug_targets,dtype=float)

# Remove targets with all 0's or 1's - but exclude drugs with only NaN, row indices 25, 31, 43, 48
drug_targets_excl_nan = numpy.delete(drug_targets,[25,31,43,48],axis=0)
indices_targets_zero_std = [i for i,std in enumerate(drug_targets_excl_nan.std(axis=0)) if std == 0.]
drug_targets = numpy.delete(drug_targets,indices_targets_zero_std,axis=1)
numpy.savetxt('features_drugs/drug_targets.txt',drug_targets,delimiter="\t")

# Store the remaining target names
target_names = numpy.delete(target_names,indices_targets_zero_std)
numpy.savetxt('features_drugs/target_names.txt',target_names,fmt="%s")

''' Next do the drug fingerprints - remove features with the same values everywhere '''
filename = "features_drugs/drug_fingerprints.csv"
drug_fingerprints = numpy.array([line.split("\n")[0].split(",")[1:] for line in open(filename,'r').readlines()[1:]],dtype=float)
indices_fingerprints_zero_std = [i for i,std in enumerate(drug_fingerprints.std(axis=0)) if std == 0.]
drug_fingerprints = numpy.delete(drug_fingerprints,indices_fingerprints_zero_std,axis=1)
numpy.savetxt('features_drugs/drug_fingerprints.txt',drug_fingerprints,delimiter="\t",fmt="%s")

''' For the 1D2D descriptors also create a version that has each feature standardised,
    so columns have mean 0 and std 1. We remove features with 0 std (same across entire column). '''
filename = "features_drugs/drug_1d2d.csv"
drug_1d2d = numpy.array([line.split("\n")[0].split(",")[1:] for line in open(filename,'r').readlines()[1:]],dtype=float)
indices_1d2d_zero_std = [i for i,std in enumerate(drug_1d2d.std(axis=0)) if std == 0.]

drug_1d2d = numpy.delete(drug_1d2d,indices_1d2d_zero_std,axis=1)
drug_1d2d_std_cols = numpy.array(drug_1d2d,dtype=float).std(axis=0)
drug_1d2d_mean_cols = numpy.array(drug_1d2d,dtype=float).mean(axis=0)
drug_1d2d_std = (numpy.array(drug_1d2d,dtype=float) - drug_1d2d_mean_cols) / drug_1d2d_std_cols

numpy.savetxt('features_drugs/drug_1d2d.txt',drug_1d2d,delimiter="\t",fmt="%s")
numpy.savetxt('features_drugs/drug_1d2d_std.txt',drug_1d2d_std,delimiter="\t",fmt="%s")