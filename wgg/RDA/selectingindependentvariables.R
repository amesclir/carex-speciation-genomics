#---------------------------------------------------#
# Step 1: Load the Data
#---------------------------------------------------#

# Load the combined data file that contains your bioclimatic variables.
bioclim_data <- read.csv("individual_bioclim_data.csv")

#---------------------------------------------------#
# Step 2: Select Only the Bioclimatic Variables
#---------------------------------------------------#

# The bioclimatic variables are typically named bio_1 through bio_19.
# Select only these columns for the correlation analysis.
# We will use a regular expression to find all columns that start with "wc2.1_30s_bio_".
bio_vars <- grep("^wc2\\.1_30s_bio_", names(bioclim_data), value = TRUE)
bioclim_subset <- bioclim_data[, bio_vars]

#---------------------------------------------------#
# Step 3: Calculate and Visualize the Correlation Matrix
#---------------------------------------------------#

# Calculate the correlation matrix
cor_matrix <- cor(bioclim_subset)

# Visualize the correlation matrix using a heatmap.
# This makes it easy to spot highly correlated variables.
# install.packages("corrplot") # Uncomment and run if you don't have this package
library(corrplot)
corrplot(cor_matrix, method = "circle", type = "upper", order = "hclust",
         tl.col = "black", tl.srt = 45)

#---------------------------------------------------#
# Step 4: Identify and Filter Highly Correlated Variables
#---------------------------------------------------#

# Set a correlation threshold (e.g., 0.7)
threshold <- 0.7

# Find highly correlated pairs
highly_cor_pairs <- which(abs(cor_matrix) > threshold & lower.tri(cor_matrix), arr.ind = TRUE)
print("Highly correlated pairs (r > 0.7):")
print(highly_cor_pairs)

# The final selected list of six uncorrelated variables
selected_variables <- c(
  "wc2.1_30s_bio_1",  # Annual Mean Temperature
  "wc2.1_30s_bio_4",  # Temperature Seasonality
 # "wc2.1_30s_bio_5",  # Max Temperature of Warmest Month
  "wc2.1_30s_bio_12", # Annual Precipitation
 # "wc2.1_30s_bio_15", # Precipitation Seasonality
 # "wc2.1_30s_bio_18"  # Precipitation of Warmest Quarter
 "wc2.1_30s_bio_14", # Prep Drought Extreme
 "wc2.1_30s_bio_16" # Prep Wet Extreme
)

print("Final selected independent variables:")
print(selected_variables)

# Create a new dataframe with only the selected variables
# This assumes the 'locations' object from a previous script run is in memory.
# Read your tab-separated file called 'locations.txt'
# The 'header=TRUE' argument assumes the first row contains column names.
# The 'sep="\t"' argument specifies that tabs are used as separators.
locations <- read.csv("locations.txt", header = TRUE, sep = "\t")

final_df_for_analysis <- bioclim_data[, c(names(locations), selected_variables)]
head(final_df_for_analysis)

# Save the final combined data to a CSV file for future use.
# This is the new line of code you requested.
write.csv(final_df_for_analysis, "final_df_for_analysis.csv", row.names = FALSE)
