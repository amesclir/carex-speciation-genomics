#---------------------------------------------------#
# Step 1: Install and Load Required Packages
#---------------------------------------------------#
# 'vegan' for the RDA, 'ggplot2' for advanced plotting,
# and 'data.table' for efficient data reading.
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

# Load the bioclimatic data.
bioclim_data <- read.csv("final_df_for_analysis.csv")

# Load the genetic data using fread() for efficiency.
raw_data <- fread("admixture_final_ld_pruned_filtered_for_RDA.raw")

# Load the population data.
tryCatch({
  # Changed from read.csv with sep="\t" to read.table with sep=" "
  pop_data <- read.table("population_list.txt", sep=" ", header = TRUE)
  if (is.null(pop_data) || nrow(pop_data) == 0) {
    stop("Population data file is empty or invalid.")
  }
}, error = function(e) {
  stop("Error loading population data: ", e$message)
})

# Format the genetic data: remove non-genotype columns and set row names.
genetic_data_final <- as.data.frame(raw_data[, 7:ncol(raw_data)])
rownames(genetic_data_final) <- raw_data$IID

# --- Subsample and clean the genetic data ---
# We will check for and remove any SNPs (columns) with NA/NaN/Inf values.
invalid_snps <- which(apply(genetic_data_final, 2, function(x) any(!is.finite(x))))
if(length(invalid_snps) > 0) {
    warning("Removed ", length(invalid_snps), " SNPs with invalid data.")
    genetic_data_final <- genetic_data_final[, -invalid_snps]
}

# Subsample SNPs to a manageable number (e.g., 5000) AFTER cleaning
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

# Find individuals with complete genetic data.
valid_genetic_individuals <- complete.cases(genetic_data_final)

# Find common individuals with valid data in all three sets.
common_valid_individuals <- intersect(bioclim_data$IND[valid_bioclim_individuals], rownames(genetic_data_final)[valid_genetic_individuals])
common_valid_individuals <- intersect(common_valid_individuals, pop_data$IID)

# --- NEW CHECK: Stop if no common individuals are found ---
if (length(common_valid_individuals) == 0) {
  stop("Error: No common individuals found across all three datasets (bioclimatic, genetic, and population data). Please check your input files.")
}

# Filter all data frames to keep only these individuals.
bioclim_data_filtered <- bioclim_data[bioclim_data$IND %in% common_valid_individuals, ]
genetic_data_final <- genetic_data_final[common_valid_individuals, ]
# The key change is adding drop = FALSE here to prevent simplification to a vector
pop_data_filtered <- pop_data[pop_data$IID %in% common_valid_individuals, , drop = FALSE]

# Order all data frames to ensure perfect alignment.
bioclim_data_filtered <- bioclim_data_filtered[order(bioclim_data_filtered$IND), ]
genetic_data_final <- genetic_data_final[order(rownames(genetic_data_final)), ]
# The key change is adding drop = FALSE here to prevent simplification to a vector
pop_data_filtered <- pop_data_filtered[order(pop_data_filtered$IID), , drop = FALSE]

# Final check of alignment
stopifnot(all(rownames(genetic_data_final) == bioclim_data_filtered$IND))
stopifnot(all(rownames(genetic_data_final) == pop_data_filtered$IID))
cat("Final number of individuals for RDA:", nrow(genetic_data_final), "\n")


#---------------------------------------------------#
# Step 4: Run the RDA Analysis
#---------------------------------------------------#
# Separate the predictor variables (bioclimatic) from the full data frame.
bioclim_predictors <- bioclim_data_filtered[, c(
  "wc2.1_30s_bio_1", "wc2.1_30s_bio_4", "wc2.1_30s_bio_12",
  "wc2.1_30s_bio_14", "wc2.1_30s_bio_16"
)]

# Scale and center the predictor variables.
bioclim_scaled <- scale(bioclim_predictors)

# Run the RDA.
rda_result <- rda(genetic_data_final ~ ., data = as.data.frame(bioclim_scaled))

#---------------------------------------------------#
# Step 5: Interpret and Visualize the RDA Results with ggplot2
#---------------------------------------------------#
summary(rda_result)
anova.cca(rda_result, permutations = 999)
anova.cca(rda_result, by = "margin", permutations = 999)

# Get the scores for plotting
site_scores <- scores(rda_result, display = "sites", scaling = 2)
env_scores <- scores(rda_result, display = "bp", scaling = 2)

# Convert scores to data frames for use with ggplot2
site_scores_df <- as.data.frame(site_scores)
site_scores_df$IID <- rownames(site_scores_df)
env_scores_df <- as.data.frame(env_scores)
env_scores_df$variables <- rownames(env_scores_df)

# Merge site scores with population data
plot_data <- merge(site_scores_df, pop_data_filtered, by = "IID")

# Create the RDA plot
rda_plot <- ggplot() +
  # Add points colored by population
  geom_point(data = plot_data, aes(x = RDA1, y = RDA2, color = POP), size = 3) +
  # Add arrows for environmental variables
  geom_segment(data = env_scores_df, aes(x = 0, y = 0, xend = RDA1, yend = RDA2), 
               arrow = arrow(length = unit(0.2, "cm")), color = "red") +
  # Add labels for environmental variables
  geom_text(data = env_scores_df, aes(x = RDA1, y = RDA2, label = variables), 
            hjust = 0, nudge_x = 0.05, color = "red") +
  # Set plot title and labels
  labs(title = "RDA Plot: Genetic Variation Explained by Bioclimatic Variables",
       subtitle = "Points colored by population",
       x = "RDA1", y = "RDA2", color = "Population") +
  # Add a theme for a clean look
  theme_minimal() +
  # Center the title and customize legend
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

# Save the RDA plot to a PDF file
pdf("RDA_plot_pop.pdf", width = 8, height = 8)
print(rda_plot)
dev.off()

print("RDA plot saved as RDA_plot_pop.pdf")

#---------------------------------------------------#
# Step 6: Perform and Visualize PCA on Bioclimatic Data
#---------------------------------------------------#
print("Performing PCA on bioclimatic variables...")

# Fix: Set the row names of the bioclimatic data to the individual IDs
# to ensure the merge with population data works correctly.
rownames(bioclim_predictors) <- bioclim_data_filtered$IND

# Perform PCA on the scaled bioclimatic data.
pca_bioclim <- prcomp(bioclim_predictors, center = TRUE, scale. = TRUE)

# Get the individual scores for PC1 and PC2.
pca_scores <- as.data.frame(pca_bioclim$x)
pca_scores$IID <- rownames(pca_scores)

# Merge PCA scores with population data.
pca_plot_data <- merge(pca_scores, pop_data_filtered, by = "IID")

# Create the PCA plot.
pca_plot <- ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = POP)) +
  geom_point(size = 3) +
  labs(title = "PCA Plot: Bioclimatic Variation",
       subtitle = "Individuals colored by population",
       x = "PC1", y = "PC2", color = "Population") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

# Save the PCA plot to a PDF file.
pdf("PCA_plot_bioclim.pdf", width = 8, height = 8)
print(pca_plot)
dev.off()

print("PCA plot saved as PCA_plot_bioclim.pdf")

