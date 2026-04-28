
# Install and load necessary packages
#install.packages("data.table")
#install.packages("vegan")
library(data.table)

# --- USER-DEFINED FILE PATHS ---
genetic_data_path <- "admixture_final_ld_pruned_filtered_for_RDA.raw"
bioclim_data_path <- "final_df_for_analysis.csv"

# --- SECTION: Load and process data ---
cat("Loading bioclimatic data...\n")
bioclim_data <- fread(bioclim_data_path)
# Ensure the bioclim data is clean and get the list of individuals
bioclim_data <- na.omit(bioclim_data)
common_individuals <- bioclim_data$IND

cat("Loading genetic data with fread()...\n")

# Get all column names from the header
header <- names(fread(genetic_data_path, nrows=0))
snp_columns <- header[7:length(header)] # Skip the first 6 columns

# Randomly select 5000 SNPs (or all if fewer exist)
set.seed(42) # For reproducibility
if (length(snp_columns) > 5000) {
    selected_snps <- sample(snp_columns, 5000)
} else {
    selected_snps <- snp_columns
}

# Add the key columns to the selection
# It's safer to get the first 6 columns from the header directly to avoid a typo like 'PHEN'
relevant_cols <- c(header[1:6], selected_snps)

# Remove any duplicate column names from the list
relevant_cols <- unique(relevant_cols)

# Load only the relevant columns and rows
genetic_data <- fread(genetic_data_path, select = relevant_cols)
setnames(genetic_data, "IID", "IND")
filtered_data <- genetic_data[IND %in% common_individuals]

cat(paste("Filtered data dimensions:", dim(filtered_data)[1], "rows,", dim(filtered_data)[2], "columns\n"))

# --- SECTION: Save the filtered genetic data ---
output_file <- "filtered_genetic_data.tsv"
cat(paste("Saving filtered data to", output_file, "...\n"))
fwrite(filtered_data, output_file, sep="\t")

cat("Filtering complete. The RDA can now be run on the smaller file in R or Python.\n")
