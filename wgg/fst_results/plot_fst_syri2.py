import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# The file path is relative since we are in the same directory
fst_file_path = "admixture_final_ld_pruned_filtered_pop1_vs_pop2.fst"

# Load the Fst data from the file
df_fst = pd.read_csv(fst_file_path, sep='\s+')

# Drop rows with NaN Fst values, which cannot be plotted
df_fst.dropna(subset=['FST'], inplace=True)

# Set any negative Fst values to 0 for plotting
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


# --- SECTION: Load and process SyRI data ---
syri_file_path = "/home/aescudero/syri/syri.out"
df_syri = pd.read_csv(syri_file_path, sep='\s+', header=None,
                      names=['CHR', 'START', 'END', 'REF_ALLELE', 'QUERY_ALLELE',
                             'QUERY_CHR', 'QUERY_START', 'QUERY_END', 'ID', 'DUP', 'TYPE', 'INFO'])

# --- CHANGES MADE HERE: Filtering the rearrangements ---

# Filter for large-scale rearrangements (excluding NOTAL)
rearrangement_types = ['INVTR', 'DEL', 'INTR', 'INTRAL', 'INS', 'DUP','INV']
df_rearrangements = df_syri[df_syri['TYPE'].isin(rearrangement_types)].copy()

# Ensure START and END columns are numeric
df_rearrangements['START'] = pd.to_numeric(df_rearrangements['START'])
df_rearrangements['END'] = pd.to_numeric(df_rearrangements['END'])

# Calculate the size of each rearrangement and filter for significant ones
df_rearrangements['SIZE'] = df_rearrangements['END'] - df_rearrangements['START']

# New approach: Define a threshold based on the data
# Let's use the 90th percentile of rearrangement sizes as a threshold
SIZE_THRESHOLD = df_rearrangements['SIZE'].abs().quantile(0.90)

df_rearrangements = df_rearrangements[df_rearrangements['SIZE'].abs() > SIZE_THRESHOLD].copy()

# Add a cumulative position to the syri data for plotting
df_rearrangements['CHR_int'] = df_rearrangements['CHR'].str.replace('scaffold_', '').astype(int)
df_rearrangements = pd.merge(df_rearrangements, df_fst[['CHR_int', 'base_pos']].drop_duplicates(), on='CHR_int', how='left')

df_rearrangements['START_cum'] = df_rearrangements['base_pos'] + df_rearrangements['START']
df_rearrangements['END_cum'] = df_rearrangements['base_pos'] + df_rearrangements['END']


# --- Plotting the data ---
fig, ax = plt.subplots(figsize=(15, 6))

# Highlight the rearrangement areas first (as shaded rectangles)
for index, row in df_rearrangements.iterrows():
    ax.axvspan(row['START_cum'], row['END_cum'], color='pink', alpha=0.3, zorder=0)
    # This creates a dummy entry for the legend
    ax.axvspan(0, 0, color='pink', alpha=0.3, label=f'Rearrangements (>{int(SIZE_THRESHOLD)}bp)')
    break # Only need one legend entry

# Plot all Fst points on top of the shaded regions
# CHANGED: Reduced dot size to s=0.5
ax.scatter(x=df_fst['BPcum'], y=df_fst['FST'], s=0.5, c=df_fst['color'], zorder=1, label='Fst Loci')

# Highlight the top 1% of Fst values to show outliers
threshold = df_fst['FST'].quantile(0.99)
df_outliers = df_fst[df_fst['FST'] > threshold]
# CHANGED: Added label for legend
ax.scatter(x=df_outliers['BPcum'], y=df_outliers['FST'], s=10, c='red', zorder=2, label=f'Top 1% Fst Loci (>{threshold:.2f})')

# Set x-axis ticks to the middle of each chromosome
chrom_df = df_fst.groupby('CHR_int')['BPcum'].agg(['min', 'max'])
chrom_df['mid'] = (chrom_df['min'] + chrom_df['max']) / 2
ax.set_xticks(chrom_df['mid'])
ax.set_xticklabels(chrom_df.index)

# Set y-axis limits to focus on high Fst values
ax.set_ylim(0, 1.0) # Set to a fixed range for better comparison

# Add labels and a title to the plot
ax.set_xlabel("Chromosome", fontsize=12)
ax.set_ylabel("Fst", fontsize=12)
ax.set_title("Fst Values for POP1 vs POP2", fontsize=16)

# Add the legend to the plot
ax.legend()

# Save the plot to a new file in the current directory
plt.savefig(f"fst_manhattan_plot_resized_filtered_pop1_vs_pop2.png")
