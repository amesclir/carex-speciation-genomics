import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# The file path is relative since we are in the same directory
fst_file_path = "admixture_final_ld_pruned_filtered_pop1_vs_pop2.fst"

# Load the Fst data from the file
df_fst = pd.read_csv(fst_file_path, sep='\s+')

# Drop rows with NaN Fst values, which cannot be plotted
df_fst.dropna(subset=['FST'], inplace=True)

# --- NEW: Set any negative Fst values to 0 for plotting ---
df_fst['FST'] = df_fst['FST'].clip(lower=0)

# Create a 'CHR' column that is an integer for logical checks
df_fst['CHR_int'] = df_fst['CHR'].astype(int)

# Create a new column for odd/even coloring
df_fst['color'] = np.where(df_fst['CHR_int'] % 2 == 1, 'gray', 'black')

# Create a 'BPcum' column for the continuous x-axis
df_fst['base_pos'] = 0
last_pos = 0
# Calculate the cumulative position for each chromosome
for chr_num in df_fst['CHR_int'].unique():
    chr_indices = df_fst['CHR_int'] == chr_num
    df_fst.loc[chr_indices, 'base_pos'] = last_pos
    last_pos += df_fst.loc[chr_indices, 'POS'].max()
df_fst['BPcum'] = df_fst['base_pos'] + df_fst['POS']

# Plotting the data
fig, ax = plt.subplots(figsize=(15, 6))

# Plot all points with odd/even colors
ax.scatter(x=df_fst['BPcum'], y=df_fst['FST'], s=2, c=df_fst['color'])

# Highlight the top 1% of Fst values to show outliers
threshold = df_fst['FST'].quantile(0.99)
df_outliers = df_fst[df_fst['FST'] > threshold]
ax.scatter(x=df_outliers['BPcum'], y=df_outliers['FST'], s=10, c='red')

# Set x-axis ticks to the middle of each chromosome
chrom_df = df_fst.groupby('CHR_int')['BPcum'].agg(['min', 'max'])
chrom_df['mid'] = (chrom_df['min'] + chrom_df['max']) / 2
ax.set_xticks(chrom_df['mid'])
ax.set_xticklabels(chrom_df.index)

# Add labels and a title to the plot
ax.set_xlabel("Chromosome", fontsize=12)
ax.set_ylabel("Fst", fontsize=12)
ax.set_title("Fst Values for POP1 vs POP2", fontsize=16)

# Save the plot to a file in the current directory
plt.savefig("fst_manhattan_plot_pop1_vs_pop2.png")
