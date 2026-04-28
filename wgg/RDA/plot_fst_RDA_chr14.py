import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats

# --- USER-DEFINED FILE PATHS ---
# Updated file paths to use the new chromosome 14-specific files from the R script.
rda_scores_path = "rda_snp_scores_chr14.csv"
selected_snps_path = "selected_snps_for_plotting_chr14.csv"
syri_file_path = "/home/aescudero/syri/syri_final2.out"
fst_file_path = "admixture_final_ld_pruned_filtered_pop1_vs_pop2.fst.sorted"

# Output filenames
output_png = "multipanel_genomic_plot_chr14.png"
output_pdf = "multipanel_genomic_plot_chr14.pdf"

# --- CONFIGURATION FOR VISUAL SIGNIFICANCE ---
# Set the percentile threshold for highlighting SNPs in the plots (e.g., 0.95 for top 5%)
VISUAL_THRESHOLD_QUANTILE = 0.95

# --- SECTION: Load and process all data ---
print("Loading and preparing data for plotting...")

try:
    snp_scores = pd.read_csv(rda_scores_path)
    selected_snps_df = pd.read_csv(selected_snps_path)
    df_fst = pd.read_csv(fst_file_path, sep='\t', header=None, 
                             names=['CHR', 'blank', 'POS', 'N_INDV', 'FST'],
                             dtype={'CHR': int, 'POS': int, 'FST': float})
    df_fst = df_fst.drop(columns=['blank', 'N_INDV'])
except FileNotFoundError as e:
    print(f"Error: Required file not found. Missing file: {e.filename}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading files: {e}")
    exit()

# Filter and prepare RDA data
valid_snp_ids = selected_snps_df['SNP_ID'].astype(str).str.contains(r'^\d+_\d+$')
selected_snps_df = selected_snps_df[valid_snp_ids].copy()
snp_scores = snp_scores[snp_scores['SNP_ID'].isin(selected_snps_df['SNP_ID'])]
snp_scores.set_index('SNP_ID', inplace=True)

# --- REVISED CALCULATION: FOCUSING ONLY ON RDA1 MAGNITUDE ---
# Since RDA1 separates species (speciation signal), we use only the magnitude of RDA1.
# np.sqrt(x**2) is mathematically equivalent to abs(x)
snp_scores['distance'] = np.sqrt(snp_scores['RDA1']**2)
print("RDA distance calculated using ONLY the magnitude of RDA1 (speciation axis).")


# Create main plot dataframe
all_snps = selected_snps_df['SNP_ID'].tolist()
df_plot = pd.DataFrame(index=all_snps)
df_plot['SNP_ID'] = df_plot.index
df_plot['CHR'] = df_plot['SNP_ID'].str.split('_').str[0].astype(int)
df_plot['POS'] = df_plot['SNP_ID'].str.split('_').str[1].astype(int)
df_plot = df_plot.sort_values(['CHR', 'POS']).reset_index(drop=True)

# Merge Fst and RDA scores
df_plot = pd.merge(df_plot, df_fst[['CHR', 'POS', 'FST']], on=['CHR', 'POS'], how='left')
df_plot = pd.merge(df_plot, snp_scores[['distance']], on='SNP_ID', how='left')

# --- Significance Calculation (for plotting only) ---
print("\nDefining visual significance thresholds (Top 5% by default):")

# 1. RDA1 Magnitude Threshold (Top 5%)
if not df_plot['distance'].isnull().all():
    rda_threshold = df_plot['distance'].quantile(VISUAL_THRESHOLD_QUANTILE)
    df_plot['is_significant_rda'] = df_plot['distance'] > rda_threshold
    print(f"RDA1 Threshold ({VISUAL_THRESHOLD_QUANTILE*100:.0f}th percentile): {rda_threshold:.4f}")
    print(f"Identified {df_plot['is_significant_rda'].sum()} significant RDA SNPs for visualization.")
else:
    df_plot['is_significant_rda'] = False
    print("Warning: RDA data contains no valid values.")

# 2. Fst Threshold (Top 5%)
if not df_plot['FST'].isnull().all():
    fst_threshold = df_plot['FST'].quantile(VISUAL_THRESHOLD_QUANTILE)
    # Create a new column to highlight top Fst SNPs as well
    df_plot['is_significant_fst'] = df_plot['FST'] > fst_threshold
    print(f"Fst Threshold ({VISUAL_THRESHOLD_QUANTILE*100:.0f}th percentile): {fst_threshold:.4f}")
    print(f"Identified {df_plot['is_significant_fst'].sum()} significant Fst SNPs for visualization.")
else:
    df_plot['is_significant_fst'] = False
    print("Warning: Fst data contains no valid values.")


# Create a 'BPcum' column for the continuous x-axis
df_plot['base_pos'] = 0
last_pos = 0
for chr_num in df_plot['CHR'].unique():
    chr_indices = df_plot['CHR'] == chr_num
    # FIX: Define chr_df here so it's available for size checks and position calculations
    chr_df = df_plot[chr_indices]
    
    # Check if there are any SNPs for this chromosome before calculating max
    if chr_df.shape[0] > 0:
        df_plot.loc[chr_indices, 'base_pos'] = last_pos
        last_pos += chr_df['POS'].max()
df_plot['BPcum'] = df_plot['base_pos'] + df_plot['POS']


# --- Load and process SyRI data ---
print("Loading and processing SyRI data...")
try:
    df_raw_syri = pd.read_csv(syri_file_path, sep='\s+', header=None)
except FileNotFoundError:
    print(f"Warning: SyRI file not found at {syri_file_path}. Skipping rearrangement plotting.")
    has_rearrangements_to_plot = False
else:
    df_raw_syri = df_raw_syri.rename(columns={10: 'TYPE'})
    rearrangement_types = ['INV', 'DUP', 'TRANS', 'INVDP', 'INVTR']
    df_rearrangements = df_raw_syri[df_raw_syri['TYPE'].isin(rearrangement_types)].copy()
    df_rearrangements['CHR'] = np.where(df_rearrangements[0] != '-', df_rearrangements[0], df_rearrangements[5])
    df_rearrangements['START'] = np.where(df_rearrangements[1] != '-', df_rearrangements[1], df_rearrangements[6])
    df_rearrangements['END'] = np.where(df_rearrangements[2] != '-', df_rearrangements[2], df_rearrangements[7])
    df_rearrangements = df_rearrangements[df_rearrangements['CHR'].str.contains('scaffold_')].copy()
    df_rearrangements['START'] = pd.to_numeric(df_rearrangements['START'], errors='coerce')
    df_rearrangements['END'] = pd.to_numeric(df_rearrangements['END'], errors='coerce')
    df_rearrangements.dropna(subset=['START', 'END'], inplace=True)
    df_rearrangements['SIZE'] = df_rearrangements['END'] - df_rearrangements['START']
    SIZE_THRESHOLD = 80000
    df_rearrangements = df_rearrangements[df_rearrangements['SIZE'].abs() > SIZE_THRESHOLD].copy()
    df_rearrangements['CHR_int'] = df_rearrangements['CHR'].str.replace('scaffold_', '').astype(int)
    df_rearrangements = pd.merge(df_rearrangements, df_plot[['CHR', 'base_pos']].drop_duplicates(), left_on='CHR_int', right_on='CHR', how='left')
    df_rearrangements.dropna(subset=['base_pos'], inplace=True)
    df_rearrangements['START_cum'] = df_rearrangements['base_pos'] + df_rearrangements['START']
    df_rearrangements['END_cum'] = df_rearrangements['base_pos'] + df_rearrangements['END']
    has_rearrangements_to_plot = True


# --- SECTION: Statistical Analysis (Same as before) ---
print("\nPerforming statistical analysis...")
df_plot['is_in_in_rearrangement'] = False
if has_rearrangements_to_plot:
    for index, row in df_rearrangements.iterrows():
        df_plot.loc[(df_plot['BPcum'] >= row['START_cum']) & (df_plot['BPcum'] <= row['END_cum']), 'is_in_in_rearrangement'] = True

# --- Mann-Whitney U test for RDA1 distance values ---
print("\n--- RDA1 Magnitude vs. Rearrangements (Mann-Whitney U Test) ---")
if has_rearrangements_to_plot:
    rda_in_rearrangements = df_plot[df_plot['is_in_in_rearrangement']]['distance'].dropna()
    rda_not_in_rearrangements = df_plot[~df_plot['is_in_in_rearrangement']]['distance'].dropna()

    if len(rda_in_rearrangements) > 0 and len(rda_not_in_rearrangements) > 0:
        # We test if 'in rearrangements' has a GREATER distance score (stronger association with RDA1)
        stat_rda_mw, p_val_rda_mw = stats.mannwhitneyu(rda_in_rearrangements, rda_not_in_rearrangements, alternative='greater')
        print("Mann-Whitney U Test Results (RDA1 Magnitude):")
        print(f"U statistic: {stat_rda_mw:.4f}")
        print(f"P-value: {p_val_rda_mw:.4f}")

        if p_val_rda_mw < 0.05:
            print("\nConclusion: The difference is statistically significant.")
            print("RDA1 scores (speciation signal) are significantly higher in the rearranged areas compared to the rest of the genome.")
        else:
            print("\nConclusion: The difference is not statistically significant.")
            print("RDA1 scores are not significantly higher in the rearranged areas compared to the rest of the genome.")
    else:
        print("Could not perform Mann-Whitney U test (RDA1). One or both groups have no data.")
else:
    print("Could not perform Mann-Whitney U test (RDA1). No rearrangement data was found.")

# Mann-Whitney U test for Fst values (kept as is)
print("\n--- Fst vs. Rearrangements (Mann-Whitney U Test) ---")
if has_rearrangements_to_plot:
    fst_in_rearrangements = df_plot[df_plot['is_in_in_rearrangement']]['FST'].dropna()
    fst_not_in_rearrangements = df_plot[~df_plot['is_in_in_rearrangement']]['FST'].dropna()

    if len(fst_in_rearrangements) > 0 and len(fst_not_in_rearrangements) > 0:
        stat, p_val_mw = stats.mannwhitneyu(fst_in_rearrangements, fst_not_in_rearrangements, alternative='greater')
        print("Mann-Whitney U Test Results (Fst):")
        print(f"U statistic: {stat:.4f}")
        print(f"P-value: {p_val_mw:.4f}")

        if p_val_mw < 0.05:
            print("\nConclusion: The difference is statistically significant.")
            print("Fst values are significantly higher in the rearranged areas compared to the rest of the genome.")
        else:
            print("\nConclusion: The difference is not statistically significant.")
            print("Fst values are not significantly higher in the rearranged areas compared to the rest of the genome.")
    else:
        print("Could not perform Mann-Whitney U test (Fst). One or both groups have no data.")
else:
    print("Could not perform Mann-Whitney U test (Fst). No rearrangement data was found.")

# --- SECTION: Plotting the data ---
print("\nGenerating the multipanel plot...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
fig.suptitle("", fontsize=16)

# Define colors for alternating chromosomes
colors = ['#528585', '#78A0A0']
x_labels = []
x_label_pos = []

# --- TOP PLOT: Fst ---
for i, chr_num in enumerate(df_plot['CHR'].unique()):
    chr_df = df_plot[df_plot['CHR'] == chr_num]
    ax1.scatter(x=chr_df['BPcum'], y=chr_df['FST'], s=5, c=colors[i % 2], zorder=1)
    
    # Calculate label position for each chromosome
    x_label_pos.append(chr_df['BPcum'].mean())
    x_labels.append(f'Chr {chr_num}')

ax1.set_ylabel("Fst", fontsize=12)
ax1.set_title(f"Fst Manhattan Plot (Top {100-VISUAL_THRESHOLD_QUANTILE*100:.0f}% Outliers Highlighted)", fontsize=14)

# Highlight significant Fst SNPs (red dots)
df_significant_fst = df_plot[df_plot['is_significant_fst']]
ax1.scatter(x=df_significant_fst['BPcum'], y=df_significant_fst['FST'], s=30, c='red', zorder=3, label=f'Top {100-VISUAL_THRESHOLD_QUANTILE*100:.0f}% Fst Outliers')

# Plot rearrangement regions
if has_rearrangements_to_plot:
    if not df_rearrangements.empty:
        for index, row in df_rearrangements.iterrows():
            ax1.axvspan(row['START_cum'], row['END_cum'], color='pink', alpha=0.3, zorder=0)
        ax1.axvspan(0, 0, color='pink', alpha=0.3, label=f'Rearrangement')
ax1.legend(loc='lower left')


# --- BOTTOM PLOT: RDA ---
for i, chr_num in enumerate(df_plot['CHR'].unique()):
    chr_df = df_plot[df_plot['CHR'] == chr_num]
    ax2.scatter(x=chr_df['BPcum'], y=chr_df['distance'], s=5, c=colors[i % 2], zorder=1)
    
# Highlight significant RDA SNPs (red dots)
df_significant_rda = df_plot[df_plot['is_significant_rda']]
ax2.scatter(x=df_significant_rda['BPcum'], y=df_significant_rda['distance'], s=30, c='red', zorder=3, label=f'Top {100-VISUAL_THRESHOLD_QUANTILE*100:.0f}% RDA1 Outliers')

ax2.set_xlabel("Genomic Position (Chromosome 14)", fontsize=12)
ax2.set_ylabel("RDA1 Magnitude (Speciation Signal)", fontsize=12)
ax2.set_title(f"RDA1 Manhattan Plot (Top {100-VISUAL_THRESHOLD_QUANTILE*100:.0f}% Outliers Highlighted)", fontsize=14)

# Plot rearrangement regions
if has_rearrangements_to_plot:
    if not df_rearrangements.empty:
        for index, row in df_rearrangements.iterrows():
            ax2.axvspan(row['START_cum'], row['END_cum'], color='pink', alpha=0.3, zorder=0)
        ax2.axvspan(0, 0, color='pink', alpha=0.3, label=f'Rearrangement')
ax2.legend(loc='lower left')

# Set x-axis labels on the bottom plot only
plt.xticks(x_label_pos, x_labels)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# --- SAVE FIGURES ---
print(f"Saving plot to {output_png}...")
plt.savefig(output_png, dpi=300)

# The new PDF saving command
print(f"Saving plot to {output_pdf}...")
plt.savefig(output_pdf, format='pdf', dpi=300)
