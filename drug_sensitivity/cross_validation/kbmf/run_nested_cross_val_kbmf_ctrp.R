# Run the nested cross-validation for KBMF

source("nested_cross_val_kbmf.R")
K <- 10
R_values <- c(2,3,4)

# Load in the drug sensitivity values
folder_data <- '../../../drug_sensitivity/data/overlap/'
folder_drug_sensitivity <- paste(folder_data,'data_row_01/',sep='')
name_ctrp <- 'ctrp_ec50_row_01.txt'
Y <- as.matrix(read.table(paste(folder_drug_sensitivity,name_ctrp,sep='')))
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
cell_lines_to_remove <- c(39, 61, 66, 73, 90, 93, 106, 110, 112, 120, 129, 193, 196, 202, 207, 222, 327, 353, 367, 368)
drugs_to_remove <- c(1, 3, 4, 39, 40, 41)
Y <- Y[-cell_lines_to_remove,-drugs_to_remove]

kernel_copy_variation <- kernel_copy_variation[-cell_lines_to_remove,-cell_lines_to_remove]
kernel_gene_expression <- kernel_gene_expression[-cell_lines_to_remove,-cell_lines_to_remove]
kernel_mutation <- kernel_mutation[-cell_lines_to_remove,-cell_lines_to_remove]

kernel_1d2d <- kernel_1d2d[-drugs_to_remove,-drugs_to_remove]
kernel_fingerprints <- kernel_fingerprints[-drugs_to_remove,-drugs_to_remove]
kernel_targets <- kernel_targets[-drugs_to_remove,-drugs_to_remove]

Px <- 3
Nx <- 399 - 20
Pz <- 2
Nz <- 52 - 6

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

# Using sigma_y = 0.5, 10 fold cross-validation, and kernels with sigma^2 = no. features
# R_values <- c(2,3,4)
# "All performances nested cross-validation: MSE=0.0993, R^2=0.3996, Rp=0.6370."
# "All performances nested cross-validation: MSE=0.0934, R^2=0.3939, Rp=0.6280."
# "All performances nested cross-validation: MSE=0.0902, R^2=0.4300, Rp=0.6564."
# "All performances nested cross-validation: MSE=0.0982, R^2=0.3908, Rp=0.6312."
# "All performances nested cross-validation: MSE=0.0853, R^2=0.4640, Rp=0.6835."
# "All performances nested cross-validation: MSE=0.0932, R^2=0.3981, Rp=0.6309."
# "All performances nested cross-validation: MSE=0.0942, R^2=0.4274, Rp=0.6592."
# "All performances nested cross-validation: MSE=0.0929, R^2=0.4260, Rp=0.6552."
# "All performances nested cross-validation: MSE=0.0944, R^2=0.4037, Rp=0.6365."
# "All performances nested cross-validation: MSE=0.0875, R^2=0.4196, Rp=0.6493."
# "Performances nested cross-validation: MSE=0.0929, R^2=0.4153, Rp=0.6467."

# Using sigma_y = 0.5, 10 fold cross-validation, and kernels with sigma^2 = no. features / 4.
# R_values <- c(2,3,4)
# "All performances nested cross-validation: MSE=0.0909, R^2=0.4255, Rp=0.6573."
# "All performances nested cross-validation: MSE=0.0922, R^2=0.4070, Rp=0.6407."
# "All performances nested cross-validation: MSE=0.0998, R^2=0.3720, Rp=0.6136."
# "All performances nested cross-validation: MSE=0.0952, R^2=0.3913, Rp=0.6280."
# "All performances nested cross-validation: MSE=0.0874, R^2=0.4561, Rp=0.6780."
# "All performances nested cross-validation: MSE=0.1007, R^2=0.3832, Rp=0.6237."
# "All performances nested cross-validation: MSE=0.0883, R^2=0.4493, Rp=0.6727."
# "All performances nested cross-validation: MSE=0.0900, R^2=0.4377, Rp=0.6663."
# "All performances nested cross-validation: MSE=0.0869, R^2=0.4389, Rp=0.6636."
# "All performances nested cross-validation: MSE=0.0880, R^2=0.4553, Rp=0.6772."
# "Performances nested cross-validation: MSE=0.0919, R^2=0.4216, Rp=0.6521."
