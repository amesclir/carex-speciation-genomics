#---------------------------------------------------#
# Step 1: Install and Load Required Packages
#---------------------------------------------------#
if (!requireNamespace("vegan", quietly = TRUE)) {
  install.packages("vegan", dependencies = TRUE)
}
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", dependencies = TRUE)
}
if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table", dependencies = TRUE)
}

library(vegan)
library(ggplot2)
library(data.table)

#---------------------------------------------------#
# Step 2: Load and Clean Data
#---------------------------------------------------#
cat("Loading and preparing data...\n")

# Load the bioclimatic data.
bioclim_data <- read.csv("final_df_for_analysis.csv")

# Define file paths
raw_data_path <- "admixture_final_ld_pruned_filtered_for_RDA.raw"
bim_data_path <- "admixture_final_ld_pruned_filtered.bim"

# Check for the existence of the bim file
if (!file.exists(bim_data_path)) {
  stop("Error: PLINK .bim file not found. Please ensure 'admixture_final_ld_pruned_filtered.bim' is in the current directory.")
}

# Load the .raw file (genotype data)
raw_data <- fread(raw_data_path)

# Load the .bim file to get SNP metadata
bim_data <- fread(bim_data_path)
colnames(bim_data) <- c("CHR", "SNP", "cM", "POS", "A1", "A2")

# --- Filter data to focus on a single chromosome (e.g., chromosome 20) ---
target_chromosome <- 20
bim_data_chr20 <- bim_data[bim_data$CHR == target_chromosome, ]
cat("Total SNPs found on Chromosome", target_chromosome, ":", nrow(bim_data_chr20), "\n")
if (nrow(bim_data_chr20) == 0) {
  stop(paste0("Error: No SNPs found on the target chromosome (", target_chromosome, ")."))
}

# Create a unique SNP ID (CHR_POS) for filtering and column naming
snp_ids <- paste0(bim_data_chr20$CHR, "_", bim_data_chr20$POS)

# Filter the genotype data to keep only SNPs on chromosome 20
genetic_data_final <- as.data.frame(raw_data[, 7:ncol(raw_data)])
# Assign the corrected SNP IDs from the *full* .bim file as temporary column names
temp_snp_ids <- paste0(bim_data$CHR, "_", bim_data$POS)
colnames(genetic_data_final) <- temp_snp_ids

# Keep only the columns corresponding to SNPs on the target chromosome
# This requires `snp_ids` to be present in the temporary column names
genetic_data_final <- genetic_data_final[, intersect(snp_ids, colnames(genetic_data_final))]
rownames(genetic_data_final) <- raw_data$IID


# --- Diagnostic Step ---
cat("Total SNPs loaded for Chromosome 20:", ncol(genetic_data_final), "\n")
cat("SNP IDs after cleaning (first 5):", head(colnames(genetic_data_final), 5), "\n")


# We will check for and remove any SNPs (columns) with NA/NaN/Inf values.
invalid_snps <- which(apply(genetic_data_final, 2, function(x) any(!is.finite(x))))
if(length(invalid_snps) > 0) {
    warning("Removed ", length(invalid_snps), " SNPs with invalid data.")
    genetic_data_final <- genetic_data_final[, -invalid_snps]
}

# Subsample SNPs to a manageable number (e.g., 5000) from the *filtered chromosome*
set.seed(42) # For reproducibility
if (ncol(genetic_data_final) > 5000) {
    selected_snps <- sample(1:ncol(genetic_data_final), 5000)
    genetic_data_final <- genetic_data_final[, selected_snps]
}
cat("Final genetic data dimensions:", dim(genetic_data_final), "\n")


#---------------------------------------------------#
# Step 3: Align and Filter Data by Individuals
#---------------------------------------------------#
# This step ensures both datasets are perfectly aligned.
# Find individuals with complete bioclimatic data.
valid_bioclim_individuals <- complete.cases(bioclim_data)

# Find individuals with complete genetic data (this should be all of them now).
valid_genetic_individuals <- complete.cases(genetic_data_final)

# Find common individuals with valid data in both sets.
common_valid_individuals <- intersect(bioclim_data$IND[valid_bioclim_individuals], rownames(genetic_data_final)[valid_genetic_individuals])

# Filter both data frames to keep only these individuals.
bioclim_data_filtered <- bioclim_data[bioclim_data$IND %in% common_valid_individuals, ]
genetic_data_final <- genetic_data_final[common_valid_individuals, ]

# Order both data frames to ensure perfect alignment.
bioclim_data_filtered <- bioclim_data_filtered[order(bioclim_data_filtered$IND), ]
genetic_data_final <- genetic_data_final[order(rownames(genetic_data_final)), ]

# Final check of alignment
stopifnot(all(rownames(genetic_data_final) == bioclim_data_filtered$IND))
cat("Final number of individuals for RDA:", nrow(genetic_data_final), "\n")


#---------------------------------------------------#
# Step 4: Run the RDA Analysis
#---------------------------------------------------#
# Separate the predictor variables (bioclimatic) from the full data frame.
bioclim_predictors <- bioclim_data_filtered[, c(
  "wc2.1_30s_bio_1", "wc2.1_30s_bio_4", "wc2.1_30s_bio_12",
  "wc2.1_30s_bio_14", "wc2.1_30s_bio_16" # CORRECTED: Removed extra double quote here
)]

# Scale and center the predictor variables.
bioclim_scaled <- scale(bioclim_predictors)

# Run the RDA.
rda_result <- rda(genetic_data_final ~ ., data = as.data.frame(bioclim_scaled))

#---------------------------------------------------#
# Step 5: Export RDA results for Python plotting
#---------------------------------------------------#
cat("Exporting RDA results for Python plotting...\n")
# Get the SNP scores (species scores)
snp_scores_df <- as.data.frame(scores(rda_result, display = "species", choices = c(1, 2), scaling = 2))
snp_scores_df$SNP_ID <- rownames(snp_scores_df)

# Save the SNP scores to a CSV file
write.csv(snp_scores_df, "rda_snp_scores_chr20.csv", row.names = FALSE)

# Save the list of selected SNPs for mapping
selected_snps_df <- data.frame(SNP_ID = colnames(genetic_data_final))
write.csv(selected_snps_df, "selected_snps_for_plotting_chr20.csv", row.names = FALSE)

cat("RDA results exported to 'rda_snp_scores_chr20.csv' and 'selected_snps_for_plotting_chr20.csv'\n")

