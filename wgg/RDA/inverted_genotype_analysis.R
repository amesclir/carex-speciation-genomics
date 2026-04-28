#---------------------------------------------------#
# Step 1: Install and Load Required Packages
#---------------------------------------------------#
if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table", dependencies = TRUE)
}
library(data.table)

# --- USER-DEFINED FILE PATHS ---
raw_data_path <- "admixture_final_ld_pruned_filtered_for_RDA_subset.raw"
bim_data_path <- "admixture_final_ld_pruned_filtered_for_RDA_subset.bim"

# --- INVERSION COORDINATES ---
inversions <- list(
  "14" = list(start = 3040103, end = 5331878),
  "20" = list(start = 4207035, end = 4474484)
)

#---------------------------------------------------#
# Step 2: Load and Identify SNPs in Inversions
#---------------------------------------------------#
cat("Loading SNP metadata to identify inversion SNPs...\n")
df_bim <- fread(bim_data_path, header = FALSE, col.names = c("CHR", "SNP", "cM", "POS", "A1", "A2"))

# Create a unique SNP ID by combining Chromosome and Position
df_bim[, unique_snp_id := paste(CHR, POS, sep = "_")]

all_inversion_unique_snps <- c()
for (chr in names(inversions)) {
  coords <- inversions[[chr]]
  cat(paste0("Identifying SNPs in inversion on Chromosome ", chr, "...\n"))
  
  # Filter SNPs within the inversion region
  inversion_bim_subset <- df_bim[CHR == chr & POS >= coords$start & POS <= coords$end]
  
  if (nrow(inversion_bim_subset) == 0) {
    cat(paste0("Warning: No SNPs found in the inversion on Chromosome ", chr, ".\n"))
  } else {
    cat(paste0("Found ", nrow(inversion_bim_subset), " SNPs on Chromosome ", chr, ".\n"))
    # Add the unique SNP ID to the list
    all_inversion_unique_snps <- c(all_inversion_unique_snps, inversion_bim_subset$unique_snp_id)
  }
}

# Add essential columns for identification
columns_to_keep <- c("IID", "FID", "PAT", "MAT", "SEX", "PHENOTYPE", all_inversion_unique_snps)

#---------------------------------------------------#
# Step 3: Load, Rename, and Filter Genotype Data
#---------------------------------------------------#
cat("Loading raw genotype data...\n")
df_raw <- fread(raw_data_path, header = TRUE)

# Get the unique SNP names directly from the loaded .bim file, as the .raw and .bim files
# should have their SNPs in the same order.
unique_snp_ids <- df_bim$unique_snp_id

# Create a new header for the data.table using the unique SNP IDs
new_header <- c(colnames(df_raw)[1:6], unique_snp_ids)
setnames(df_raw, new_header)

# Select and filter the columns
df_raw_subset <- df_raw[, ..columns_to_keep]

# Save the filtered data to a new CSV file
output_file <- "filtered_inversion_genotypes.csv"
fwrite(df_raw_subset, output_file)

cat(paste0("Filtered genotype data saved to '", output_file, "'.\n"))

