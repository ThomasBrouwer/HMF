source("kbmf_regression_train.R")
source("kbmf_regression_test.R")

set.seed(0)

# Load in the drug sensitivity values
folder_data <- '../../../drug_sensitivity/data/overlap/'
folder_drug_sensitivity <- paste(folder_data,'data_row_01/',sep='')
name_gdsc <- 'gdsc_ic50_row_01.txt'
Y <- as.matrix(read.table(paste(folder_drug_sensitivity,name_gdsc,sep='')))
Y[is.nan(Y)] = NA

print("Loaded data")

# Load in the kernels - X = cancer cell lines, Z = drugs
folder_kernels <- folder_drug_sensitivity <- paste(folder_data,'kernels_features/',sep='')

kernel_copy_variation <- as.matrix(read.table(paste(folder_kernels,'cnv_std.txt',sep='')))
kernel_gene_expression <- as.matrix(read.table(paste(folder_kernels,'gene_expression_std.txt',sep='')))
kernel_mutation <- as.matrix(read.table(paste(folder_kernels,'mutation.txt',sep='')))

kernel_1d2d <- as.matrix(read.table(paste(folder_kernels,'drug_1d2d_std.txt',sep='')))
kernel_fingerprints <- as.matrix(read.table(paste(folder_kernels,'drug_fingerprints.txt',sep='')))
kernel_targets <- as.matrix(read.table(paste(folder_kernels,'drug_targets.txt',sep='')))

# Remove the drugs and cell lines that have no entries for this dataset
# For GDSC: drug indices 26, 32, 44, 49 (not zero-indexed!)
Y <- Y[,-c(26,32,44,49)]
kernel_1d2d <- kernel_1d2d[-c(26,32,44,49),-c(26,32,44,49)]
kernel_fingerprints <- kernel_fingerprints[-c(26,32,44,49),-c(26,32,44,49)]
kernel_targets <- kernel_targets[-c(26,32,44,49),-c(26,32,44,49)]

Px <- 3
Nx <- 399
Pz <- 3
Nz <- 52 - 4

Kx <- array(0, c(Nx, Nx, Px))
Kx[,, 1] <- kernel_copy_variation
Kx[,, 2] <- kernel_gene_expression
Kx[,, 3] <- kernel_mutation

Kz <- array(0, c(Nz, Nz, Pz))
Kz[,, 1] <- kernel_1d2d
Kz[,, 2] <- kernel_fingerprints
Kz[,, 3] <- kernel_targets

print("Loaded kernels")

# Train the model, and test the performance on the training data
R <- 5
state <- kbmf_regression_train(Kx, Kz, Y, R)
prediction <- kbmf_regression_test(Kx, Kz, state)

print("Trained model")
#print(prediction$Y$mu)

print(sprintf("MSE = %.4f", mean((prediction$Y$mu - Y)^2, na.rm=TRUE )))
# R=5, 200 iterations: "MSE = 2.0170"
# R=5, 1000 iterations: "MSE = 2.0131"
# R=10, 100 iterations: "MSE = 1.5869"
# R=10, 200 iterations: "MSE = 1.5736"
# R=10, 1000 iterations: "MSE = 1.5644"

print("kernel weights on X")
print(state$ex$mu)

print("kernel weights on Z")
print(state$ez$mu)
