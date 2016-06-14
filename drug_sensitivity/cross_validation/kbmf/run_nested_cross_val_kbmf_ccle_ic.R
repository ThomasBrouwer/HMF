# Run the nested cross-validation for KBMF

source("nested_cross_val_kbmf.R")
K <- 10
R_values <- c(2,3,4)

# Load in the drug sensitivity values
folder_data <- '../../../drug_sensitivity/data/overlap/'
folder_drug_sensitivity <- paste(folder_data,'data_row_01/',sep='')
name_ccle_ic <- 'ccle_ic50_row_01.txt'
Y <- as.matrix(read.table(paste(folder_drug_sensitivity,name_ccle_ic,sep='')))
Y[is.nan(Y)] = NA

# Load in the kernels - X = cancer cell lines, Z = drugs
folder_kernels <- folder_drug_sensitivity <- paste(folder_data,'kernels_features/',sep='')

kernel_copy_variation <- as.matrix(read.table(paste(folder_kernels,'cnv_std.txt',sep='')))
kernel_gene_expression <- as.matrix(read.table(paste(folder_kernels,'gene_expression_std.txt',sep='')))
kernel_mutation <- as.matrix(read.table(paste(folder_kernels,'mutation.txt',sep='')))

kernel_1d2d <- as.matrix(read.table(paste(folder_kernels,'drug_1d2d_std.txt',sep='')))
kernel_fingerprints <- as.matrix(read.table(paste(folder_kernels,'drug_fingerprints.txt',sep='')))
kernel_targets <- as.matrix(read.table(paste(folder_kernels,'drug_targets.txt',sep='')))

# Remove the drugs and cell lines that have no entries for this dataset
cell_lines_to_remove <- c(1, 4, 8, 15, 16, 17, 21, 22, 26, 27, 28, 30, 31, 35, 40, 41, 42, 44, 46, 47, 48, 49, 54, 55, 57, 58, 59, 60, 64, 67, 68, 70, 71, 74, 75, 76, 77, 79, 80, 85, 86, 89, 91, 92, 96, 97, 99, 102, 111, 114, 116, 118, 119, 123, 125, 128, 130, 131, 136, 137, 144, 149, 154, 159, 160, 161, 162, 166, 167, 178, 179, 180, 182, 183, 184, 185, 187, 191, 192, 197, 198, 204, 211, 213, 216, 218, 221, 227, 234, 237, 240, 243, 245, 246, 247, 249, 250, 253, 255, 256, 261, 265, 266, 269, 273, 275, 276, 278, 279, 282, 286, 288, 291, 293, 297, 300, 301, 305, 307, 308, 309, 311, 313, 319, 322, 324, 333, 336, 337, 346, 352, 354, 355, 358, 360, 362, 363, 369, 370, 375, 378, 382, 383, 393, 396, 397)
drugs_to_remove <- c(2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 33, 34, 35, 37, 38, 43, 46, 47, 48, 50, 51, 52)
Y <- Y[-cell_lines_to_remove,-drugs_to_remove]

kernel_copy_variation <- kernel_copy_variation[-cell_lines_to_remove,-cell_lines_to_remove]
kernel_gene_expression <- kernel_gene_expression[-cell_lines_to_remove,-cell_lines_to_remove]
kernel_mutation <- kernel_mutation[-cell_lines_to_remove,-cell_lines_to_remove]

kernel_1d2d <- kernel_1d2d[-drugs_to_remove,-drugs_to_remove]
kernel_fingerprints <- kernel_fingerprints[-drugs_to_remove,-drugs_to_remove]
kernel_targets <- kernel_targets[-drugs_to_remove,-drugs_to_remove]

Px <- 3
Nx <- 253
Pz <- 2
Nz <- 16

Kx <- array(0, c(Nx, Nx, Px))
Kx[,, 1] <- kernel_copy_variation
Kx[,, 2] <- kernel_gene_expression
Kx[,, 3] <- kernel_mutation

Kz <- array(0, c(Nz, Nz, Pz))
Kz[,, 1] <- kernel_1d2d
Kz[,, 2] <- kernel_fingerprints
# No targets as we don't have that info for all drugs
#Kz[,, 3] <- kernel_targets

# Run the cross-validation
kbmf_nested_cross_validation(Kx, Kz, Y, R_values, K)

# R_values <- c(2,3,4)]
# All performances nested cross-validation: MSE=0.0732, R^2=0.5609, Rp=0.7631.
# All performances nested cross-validation: MSE=0.0743, R^2=0.5848, Rp=0.7770.
# All performances nested cross-validation: MSE=0.0720, R^2=0.5679, Rp=0.7817.
# All performances nested cross-validation: MSE=0.0854, R^2=0.5197, Rp=0.7339.
# All performances nested cross-validation: MSE=0.0647, R^2=0.6339, Rp=0.8239.
# All performances nested cross-validation: MSE=0.0798, R^2=0.5477, Rp=0.7651.
# All performances nested cross-validation: MSE=0.0756, R^2=0.5685, Rp=0.7769.
# All performances nested cross-validation: MSE=0.0743, R^2=0.5862, Rp=0.7862.
# All performances nested cross-validation: MSE=0.0846, R^2=0.4628, Rp=0.7184.
# All performances nested cross-validation: MSE=0.0801, R^2=0.5695, Rp=0.7670.
# Performances nested cross-validation: MSE=0.0764, R^2=0.5602, Rp=0.7693.
