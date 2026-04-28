#---------------------------------------------------#
# Step 1: Install and Load Required Packages
#---------------------------------------------------#
if (!requireNamespace("vegan", quietly = TRUE)) install.packages("vegan")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")

library(vegan)
library(ggplot2)
library(data.table)

#---------------------------------------------------#
# Step 2: Load and Clean Data
#---------------------------------------------------#
bioclim_data <- read.csv("final_df_for_analysis.csv")
raw_data <- fread("admixture_final_ld_pruned_filtered_for_RDA.raw")
pop_data <- read.table("population_list.txt", sep=" ", header = TRUE)

genetic_data_full <- as.data.frame(raw_data[, 7:ncol(raw_data)])
rownames(genetic_data_full) <- raw_data$IID

# Remove invalid SNPs
invalid_snps <- which(apply(genetic_data_full, 2, function(x) any(!is.finite(x))))
if(length(invalid_snps) > 0) genetic_data_full <- genetic_data_full[, -invalid_snps]

# Subsample SNPs
set.seed(42)
if (ncol(genetic_data_full) > 5000) {
    genetic_data_final <- genetic_data_full[, sample(1:ncol(genetic_data_full), 5000)]
} else {
    genetic_data_final <- genetic_data_full
}

#---------------------------------------------------#
# Step 3: Align and Filter Data
#---------------------------------------------------#
common_ids <- intersect(intersect(bioclim_data$IND, rownames(genetic_data_final)), pop_data$IID)

bioclim_filtered <- bioclim_data[bioclim_data$IND %in% common_ids, ]
genetic_filtered <- genetic_data_final[common_ids, ]
pop_filtered <- pop_data[pop_data$IID %in% common_ids, , drop = FALSE]

# Sort to align
bioclim_filtered <- bioclim_filtered[order(bioclim_filtered$IND), ]
genetic_filtered <- genetic_filtered[order(rownames(genetic_filtered)), ]
pop_filtered <- pop_filtered[order(pop_filtered$IID), , drop = FALSE]

#---------------------------------------------------#
# NEW Step: Define Population Structure Covariates
#---------------------------------------------------#
# To control for species structure, we run a PCA on the genetic data 
# and use the first few PCs as conditioning variables.
neutral_pca <- prcomp(genetic_filtered, center = TRUE, scale. = FALSE)
# Using PC1 and PC2 usually captures the species-level divergence
struct_covariates <- neutral_pca$x[, 1:2] 

#---------------------------------------------------#
# Step 4: Run the Partial RDA Analysis
#---------------------------------------------------#
bioclim_predictors <- bioclim_filtered[, c(
  "wc2.1_30s_bio_1", "wc2.1_30s_bio_4", "wc2.1_30s_bio_12",
  "wc2.1_30s_bio_14", "wc2.1_30s_bio_16"
)]

bioclim_scaled <- scale(bioclim_predictors)

# partial RDA syntax: Y ~ predictors + Condition(covariates)
rda_result <- rda(genetic_filtered ~ bioclim_scaled + Condition(struct_covariates))

#---------------------------------------------------#
# Step 5: Interpret and Visualize
#---------------------------------------------------#
summary_res <- summary(rda_result)
# Test significance of the bioclimatic variables only
anova_res <- anova.cca(rda_result, permutations = 999)
print(anova_res)

# Get scores
site_scores <- as.data.frame(scores(rda_result, display = "sites", scaling = 2))
env_scores <- as.data.frame(scores(rda_result, display = "bp", scaling = 2))
site_scores$IID <- rownames(site_scores)
env_scores$variables <- c("Bio1", "Bio4", "Bio12", "Bio14", "Bio16")

plot_data <- merge(site_scores, pop_filtered, by = "IID")

rda_plot <- ggplot() +
  geom_point(data = plot_data, aes(x = RDA1, y = RDA2, color = POP), size = 3) +
  geom_segment(data = env_scores, aes(x = 0, y = 0, xend = RDA1, yend = RDA2), 
               arrow = arrow(length = unit(0.2, "cm")), color = "red") +
  geom_text(data = env_scores, aes(x = RDA1, y = RDA2, label = variables), 
            hjust = 0, nudge_x = 0.05, color = "red") +
  scale_color_manual(
    values = c("C_borbonica1"="cyan","C_borbonica2"="cyan3", "hybrid1"="cyan4", 
               "C_boryana1"="lightgoldenrod1", "C_boryana2"="gold","C_boryana3"="orange",
               "C_boryana4"="orange3","C_boryana5"="orange4", "hybrid2"="brown4"),
    labels = c("C_borbonica1"="borbonica PF", "C_borbonica2"="borbonica LM", "hybrid1"="borbonica int.", 
               "C_boryana1"="boryana PF", "C_boryana2"="boryana TF", "C_boryana3"="boryana LM", 
               "C_boryana4"="boryana PN", "C_boryana5"="boryana RE",  "hybrid2"="boryana int.")
  ) +
  labs(title = "Partial RDA (Controlled for Population Structure)",
       x = "RDA1", y = "RDA2", color = "Locations") +
  theme_minimal()

adj_r2 <- RsquareAdj(rda_result)
print(adj_r2)

pdf("pRDA_plot_controlled.pdf", width = 8, height = 8)
print(rda_plot)
dev.off()
