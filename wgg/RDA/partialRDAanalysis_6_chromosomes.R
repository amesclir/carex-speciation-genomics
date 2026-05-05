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
# Step 2: Load Global Data (Only Once!)
#---------------------------------------------------#
cat("Loading global datasets...\n")

# Load the bioclimatic data
bioclim_data <- read.csv("final_df_for_analysis.csv")

# Load the structure data for the partial RDA condition
pop_data_struct <- read.table("population_list_pop1_2.txt", sep=" ", header = TRUE)

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

# Create temporary global SNP IDs
temp_snp_ids <- paste0(bim_data$CHR, "_", bim_data$POS)

# Define the chromosomes to analyze
target_chromosomes <- c(3, 14, 18, 20, 28, 32)

#---------------------------------------------------#
# Step 3: Loop Through Each Chromosome
#---------------------------------------------------#
for (target_chromosome in target_chromosomes) {
  
  cat("\n=======================================================\n")
  cat("PROCESSING CHROMOSOME", target_chromosome, "\n")
  cat("=======================================================\n")
  
  # Filter bim data to focus on the current chromosome
  bim_data_chr <- bim_data[bim_data$CHR == target_chromosome, ]
  cat("Total SNPs found on Chromosome", target_chromosome, ":", nrow(bim_data_chr), "\n")
  if (nrow(bim_data_chr) == 0) {
    warning(paste0("No SNPs found on chromosome ", target_chromosome, ". Skipping..."))
    next
  }
  
  # Create a unique SNP ID (CHR_POS) for filtering
  snp_ids <- paste0(bim_data_chr$CHR, "_", bim_data_chr$POS)
  
  # Extract only the columns needed for this chromosome to save memory
  # +6 accounts for the first 6 metadata columns in a .raw file
  col_indices <- which(temp_snp_ids %in% snp_ids) + 6
  genetic_data_final <- as.data.frame(raw_data[, ..col_indices]) 
  colnames(genetic_data_final) <- temp_snp_ids[col_indices - 6]
  rownames(genetic_data_final) <- raw_data$IID
  
  # We will check for and remove any SNPs (columns) with NA/NaN/Inf values.
  invalid_snps <- which(apply(genetic_data_final, 2, function(x) any(!is.finite(x))))
  if(length(invalid_snps) > 0) {
      warning("Removed ", length(invalid_snps), " SNPs with invalid data.")
      genetic_data_final <- genetic_data_final[, -invalid_snps]
  }
  
  # Subsample SNPs to a manageable number (e.g., 5000)
  set.seed(42) # For reproducibility
  if (ncol(genetic_data_final) > 5000) {
      selected_snps <- sample(1:ncol(genetic_data_final), 5000)
      genetic_data_final <- genetic_data_final[, selected_snps]
  }
  cat("Final genetic data dimensions for Chr", target_chromosome, ":", dim(genetic_data_final), "\n")
  
  
  # --- Align and Filter Data by Individuals ---
  valid_bioclim_individuals <- complete.cases(bioclim_data)
  valid_genetic_individuals <- complete.cases(genetic_data_final)
  
  # Find common individuals across Bioclim, Genetics, AND the Population text file
  common_valid_individuals <- intersect(bioclim_data$IND[valid_bioclim_individuals], rownames(genetic_data_final)[valid_genetic_individuals])
  common_valid_individuals <- intersect(common_valid_individuals, pop_data_struct$IID)
  
  # Filter all three data frames to keep only these individuals
  bioclim_data_filtered <- bioclim_data[bioclim_data$IND %in% common_valid_individuals, ]
  genetic_data_final <- genetic_data_final[common_valid_individuals, ]
  pop_filtered_struct <- pop_data_struct[pop_data_struct$IID %in% common_valid_individuals, , drop = FALSE]
  
  # Order data frames to ensure perfect alignment
  bioclim_data_filtered <- bioclim_data_filtered[order(bioclim_data_filtered$IND), ]
  genetic_data_final <- genetic_data_final[order(rownames(genetic_data_final)), ]
  pop_filtered_struct <- pop_filtered_struct[order(pop_filtered_struct$IID), , drop = FALSE]
  
  # Final check of alignment
  stopifnot(all(rownames(genetic_data_final) == bioclim_data_filtered$IND))
  stopifnot(all(rownames(genetic_data_final) == pop_filtered_struct$IID))
  cat("Final number of individuals for partial RDA:", nrow(genetic_data_final), "\n")
  
  
  # --- Run the PARTIAL RDA Analysis ---
  # Separate the predictor variables (bioclimatic) from the full data frame
  bioclim_predictors <- bioclim_data_filtered[, c(
    "wc2.1_30s_bio_1", "wc2.1_30s_bio_4", "wc2.1_30s_bio_12",
    "wc2.1_30s_bio_14", "wc2.1_30s_bio_16" 
  )]
  
  # Scale and center the predictor variables
  bioclim_scaled <- as.data.frame(scale(bioclim_predictors))
  
  # Combine with Population Structure
  env_df <- cbind(bioclim_scaled, Species_Structure = as.factor(pop_filtered_struct$POP))
  
  cat("Running Partial RDA for Chromosome", target_chromosome, "...\n")
  rda_result <- rda(genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 + 
                                         wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + 
                                         wc2.1_30s_bio_16 + 
                                         Condition(Species_Structure), 
                    data = env_df)
  
  # --- PRINT STATISTICS ---
  adj_r2 <- RsquareAdj(rda_result)
  cat("\nAdjusted R-squared for Chromosome", target_chromosome, "(controlling for structure):\n")
  print(adj_r2)
  
  cat("\nANOVA Permutation test for Chromosome", target_chromosome, ":\n")
  anova_res <- anova.cca(rda_result, permutations = 999)
  print(anova_res)
  
  
  # --- Export pRDA results for Python plotting ---
  cat("\nExporting partial RDA results for Python plotting...\n")
  snp_scores_df <- as.data.frame(scores(rda_result, display = "species", choices = c(1, 2), scaling = 2))
  snp_scores_df$SNP_ID <- rownames(snp_scores_df)
  
  # Dynamic filenames based on the current loop chromosome
  scores_filename <- paste0("rda_snp_scores_chr", target_chromosome, ".csv")
  snps_filename <- paste0("selected_snps_for_plotting_chr", target_chromosome, ".csv")
  
  write.csv(snp_scores_df, scores_filename, row.names = FALSE)
  
  selected_snps_df <- data.frame(SNP_ID = colnames(genetic_data_final))
  write.csv(selected_snps_df, snps_filename, row.names = FALSE)
  
  cat("RDA results exported to", scores_filename, "and", snps_filename, "\n")
}

cat("\nAll partial RDA chromosomes processed successfully!\n")
