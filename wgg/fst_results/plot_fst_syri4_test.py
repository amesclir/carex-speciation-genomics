import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

# --- USER-DEFINED FILE PATHS ---
fst_file_path = "admixture_final_ld_pruned_filtered_pop1_vs_pop2.fst"
syri_file_path = "/home/aescudero/syri/syri.out"
output_plot_path = "fst_manhattan_plot_high_Fst_10k_rearrangementsINVDUPTRANS_with_enrichment.png"

# --- SECTION: Load and process Fst data ---
print("Loading and processing Fst data...")
# Read the full FST file
df_fst_full = pd.read_csv(fst_file_path, sep='\s+')
df_fst_full.dropna(subset=['FST'], inplace=True)
df_fst_full['FST'] = df_fst_full['FST'].clip(lower=0)

# Explicitly convert 'CHR' column to string to avoid the AttributeError
df_fst_full['CHR'] = df_fst_full['CHR'].astype(str)
df_fst_full['CHR_int'] = df_fst_full['CHR'].str.replace('scaffold_', '').astype(int)

# Create a 'BPcum' column for the continuous x-axis on the full dataset
df_fst_full['base_pos'] = 0
last_pos = 0
for chr_num in df_fst_full['CHR_int'].unique():
    chr_indices = df_fst_full['CHR_int'] == chr_num
    # Check if there are any loci for the chromosome before processing
    if not chr_indices.any():
        continue
    df_fst_full.loc[chr_indices, 'base_pos'] = last_pos
    last_pos += df_fst_full.loc[chr_indices, 'POS'].max()
df_fst_full['BPcum'] = df_fst_full['base_pos'] + df_fst_full['POS']

# Now, filter the dataset for plotting and analysis
df_outliers = df_fst_full[df_fst_full['FST'] > df_fst_full['FST'].quantile(0.99)].copy()
df_fst = df_fst_full[df_fst_full['FST'] > 0.9].copy()
df_fst['color'] = np.where(df_fst['CHR_int'] % 2 == 1, 'gray', 'black')

# --- SECTION: Load and process SyRI data ---
print("Loading and processing SyRI data...")
# Explicitly set dtypes to avoid DtypeWarning and ensure string operations work
df_raw = pd.read_csv(syri_file_path, sep='\s+', header=None, dtype={0: str, 5: str})
df_raw = df_raw.rename(columns={10: 'TYPE'})

# Filter for specific rearrangement types
rearrangement_types = ['INV', 'DUP', 'TRANS']
df_rearrangements = df_raw[df_raw['TYPE'].isin(rearrangement_types)].copy()

# Combine data from two possible formats
df_rearrangements['CHR'] = np.where(df_rearrangements[0] != '-', df_rearrangements[0], df_rearrangements[5])
df_rearrangements['START'] = np.where(df_rearrangements[1] != '-', df_rearrangements[1], df_rearrangements[6])
df_rearrangements['END'] = np.where(df_rearrangements[2] != '-', df_rearrangements[2], df_rearrangements[7])

# Filter out rows where CHR, START, or END are still invalid
df_rearrangements = df_rearrangements[df_rearrangements['CHR'].str.contains('scaffold_')].copy()

# Convert positions to numeric, coercing errors to NaN
df_rearrangements['START'] = pd.to_numeric(df_rearrangements['START'], errors='coerce')
df_rearrangements['END'] = pd.to_numeric(df_rearrangements['END'], errors='coerce')
df_rearrangements.dropna(subset=['START', 'END'], inplace=True)

# Calculate rearrangement size and filter
df_rearrangements['SIZE'] = df_rearrangements['END'] - df_rearrangements['START']
SIZE_THRESHOLD = 10000
df_rearrangements = df_rearrangements[df_rearrangements['SIZE'].abs() > SIZE_THRESHOLD].copy()

# Add a cumulative position to the syri data for plotting
df_rearrangements['CHR_int'] = df_rearrangements['CHR'].str.replace('scaffold_', '').astype(int)
# Merge with the base_pos from the filtered Fst data
df_rearrangements = pd.merge(df_rearrangements, df_fst_full[['CHR_int', 'base_pos']].drop_duplicates(), on='CHR_int', how='left')
df_rearrangements.dropna(subset=['base_pos'], inplace=True)
df_rearrangements['START_cum'] = df_rearrangements['base_pos'] + df_rearrangements['START']
df_rearrangements['END_cum'] = df_rearrangements['base_pos'] + df_rearrangements['END']

# --- SECTION: Statistical Enrichment Analysis ---
print("\n--- Testing All Rearrangements for Fst Enrichment ---")

def is_in_rearrangement(row, rearrangement_df):
    for _, r_row in rearrangement_df.iterrows():
        if row['CHR_int'] == r_row['CHR_int'] and r_row['START'] <= row['POS'] <= r_row['END']:
            return True
    return False

# Identify loci inside and outside rearrangements
df_fst_full['in_rearrangement'] = df_fst_full.apply(lambda row: is_in_rearrangement(row, df_rearrangements), axis=1)
df_in_rearrangements = df_fst_full[df_fst_full['in_rearrangement']]['FST']
df_outside_rearrangements = df_fst_full[~df_fst_full['in_rearrangement']]['FST']

if not df_in_rearrangements.empty and not df_outside_rearrangements.empty:
    # Perform Mann-Whitney U test
    u_statistic, p_value = stats.mannwhitneyu(df_in_rearrangements, df_outside_rearrangements, alternative='greater')
    
    print(f"Number of loci within rearrangements: {len(df_in_rearrangements)}")
    print(f"Number of loci outside rearrangements: {len(df_outside_rearrangements)}")
    print(f"Mean Fst within rearrangements: {df_in_rearrangements.mean():.4f}")
    print(f"Mean Fst outside rearrangements: {df_outside_rearrangements.mean():.4f}")
    print(f"Median Fst within rearrangements: {df_in_rearrangements.median():.4f}")
    print(f"Median Fst outside rearrangements: {df_outside_rearrangements.median():.4f}")
    print(f"\nMann-Whitney U test (testing if Fst is higher within rearrangements):")
    print(f"  U-statistic: {u_statistic:.2f}")
    print(f"  P-value: {p_value:.5f}")

    if p_value < 0.05:
        print("\n--> CONCLUSION: Fst values within rearrangements are SIGNIFICANTLY higher than outside.")
    else:
        print("\n--> CONCLUSION: No significant difference in Fst values between loci inside and outside rearrangements.")
else:
    print("Not enough data in one of the groups to perform statistical analysis.")

# --- FIX: Define the threshold before it's used ---
# Calculate the threshold for high Fst values to be used in the next section.
threshold = df_fst_full['FST'].quantile(0.99)
print(f"\nCalculated Fst threshold (99th percentile): {threshold:.4f}")

print("\n--- Testing Top 10 Largest Rearrangements Individually ---")
top_10_rearrangements = df_rearrangements.sort_values('SIZE', ascending=False).head(10).copy()

for index, row in top_10_rearrangements.iterrows():
    rearrangement_loci = df_fst_full[
        (df_fst_full['CHR_int'] == row['CHR_int']) &
        (df_fst_full['POS'] >= row['START']) &
        (df_fst_full['POS'] <= row['END'])
    ]
    high_fst_in_rearrangement = rearrangement_loci[rearrangement_loci['FST'] > threshold]
    
    rearrangement_length = row['END'] - row['START']
    total_fst_loci_in_rearrangement = len(rearrangement_loci)
    observed_high_fst_count = len(high_fst_in_rearrangement)

    print(f"\nRearrangement {index + 1}: {row['TYPE']} on CHR {row['CHR_int']} from {row['START']:,} to {row['END']:,} (Size: {rearrangement_length:,} bp)")
    print(f"  - Loci within this rearrangement: {total_fst_loci_in_rearrangement}")
    print(f"  - Top 1% Fst loci within this rearrangement: {observed_high_fst_count}")

    if total_fst_loci_in_rearrangement > 0 and len(df_outliers) > 0:
        proportion_rearrangement = observed_high_fst_count / total_fst_loci_in_rearrangement
        proportion_genome = len(df_outliers) / len(df_fst_full)
        print(f"  - Proportion of high Fst loci in this rearrangement: {proportion_rearrangement:.2%}")
        print(f"  - Proportion of high Fst loci in whole dataset: {proportion_genome:.2%}")
        if proportion_rearrangement > proportion_genome:
            print("  --> Conclusion: ENRICHED with high-Fst loci.")
        else:
            print("  --> Conclusion: NOT significantly enriched.")
    else:
        print("  - Not enough data to perform enrichment test for this rearrangement.")


# --- SECTION: Plotting the data ---
fig, ax = plt.subplots(figsize=(15, 8))

has_rearrangements_to_plot = not df_rearrangements.empty
if has_rearrangements_to_plot:
    for index, row in df_rearrangements.iterrows():
        ax.axvspan(row['START_cum'], row['END_cum'], color='pink', alpha=0.3, zorder=0)
    ax.axvspan(0, 0, color='pink', alpha=0.3, label=f'Rearrangements (>{SIZE_THRESHOLD} bp)')

ax.scatter(x=df_fst['BPcum'], y=df_fst['FST'], s=0.5, c=df_fst['color'], zorder=1, label='Fst Loci > 0.9')

# Use the 'df_outliers' from the full dataset for the red dots
ax.scatter(x=df_outliers['BPcum'], y=df_outliers['FST'], s=10, c='red', zorder=2, label=f'Top 1% Fst Loci (>{df_outliers["FST"].min():.2f})')

# Set x-axis ticks and labels
chrom_df = df_fst_full.groupby('CHR_int')['BPcum'].agg(['min', 'max'])
chrom_df['mid'] = (chrom_df['min'] + chrom_df['max']) / 2
ax.set_xticks(chrom_df['mid'])
ax.set_xticklabels(chrom_df.index)

ax.set_ylim(0.0, 1.0)
ax.set_xlabel("Chromosome", fontsize=12)
ax.set_ylabel("Fst", fontsize=12)
ax.set_title(f"Fst Values for POP1 vs POP2", fontsize=16)

ax.legend()
plt.tight_layout()
plt.savefig(output_plot_path, dpi=300)
print(f"\nPlot saved to {output_plot_path}")
