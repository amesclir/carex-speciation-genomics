import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats

# --- USER-DEFINED FILE PATHS ---
rda_scores_path = "rda_snp_scores.csv"
selected_snps_path = "selected_snps_for_plotting.csv"
syri_file_path = "/home/aescudero/syri/syri_final2.out"
fst_file_path = "admixture_final_ld_pruned_filtered_pop1_vs_pop2.fst.sorted"

# Output filenames
output_png = "multipanel_genomic_plot.png"
output_pdf = "multipanel_genomic_plot.pdf"

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
#snp_scores['distance'] = np.sqrt(snp_scores['RDA1']**2 + snp_scores['RDA2']**2)
snp_scores['distance'] = snp_scores['RDA1']

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

# --- Significance Calculation (RDA) ---
if not df_plot['distance'].isnull().all():
    # Use a statistical cutoff (e.g., 2 standard deviations)
    rda_threshold = df_plot['distance'].mean() + 2 * df_plot['distance'].std()
    df_plot['is_significant_rda'] = df_plot['distance'] > rda_threshold
    print(f"Identified {df_plot['is_significant_rda'].sum()} significant SNPs based on RDA (Threshold: {rda_threshold:.2f}).")
else:
    df_plot['is_significant_rda'] = False
    print("Warning: RDA data contains no valid values.")

# Create a 'BPcum' column for the continuous x-axis
df_plot['base_pos'] = 0
last_pos = 0
for chr_num in df_plot['CHR'].unique():
    chr_indices = df_plot['CHR'] == chr_num
    df_plot.loc[chr_indices, 'base_pos'] = last_pos
    last_pos += df_plot.loc[chr_indices, 'POS'].max()
df_plot['BPcum'] = df_plot['base_pos'] + df_plot['POS']


# --- Load and process SyRI data (Rearrangements) ---
print("Loading and processing SyRI data...")
try:
    df_raw_syri = pd.read_csv(syri_file_path, sep='\s+', header=None)
except FileNotFoundError:
    print(f"Warning: SyRI file not found at {syri_file_path}. Skipping rearrangement analysis and plotting.")
    has_rearrangements_to_plot = False
else:
    # Rename column 10 to 'TYPE'
    df_raw_syri = df_raw_syri.rename(columns={10: 'TYPE'})
    rearrangement_types = ['INV', 'DUP', 'TRANS', 'INVDP', 'INVTR']
    df_rearrangements = df_raw_syri[df_raw_syri['TYPE'].isin(rearrangement_types)].copy()
    
    # Extract chromosome and positions, prioritizing the first set if available, otherwise the second
    df_rearrangements['CHR_STR'] = np.where(df_rearrangements[0] != '-', df_rearrangements[0], df_rearrangements[5])
    df_rearrangements['START'] = np.where(df_rearrangements[1] != '-', df_rearrangements[1], df_rearrangements[6])
    df_rearrangements['END'] = np.where(df_rearrangements[2] != '-', df_rearrangements[2], df_rearrangements[7])
    
    # Filter only scaffolds/chromosomes we can map
    df_rearrangements = df_rearrangements[df_rearrangements['CHR_STR'].str.contains('scaffold_')].copy()
    
    df_rearrangements['START'] = pd.to_numeric(df_rearrangements['START'], errors='coerce')
    df_rearrangements['END'] = pd.to_numeric(df_rearrangements['END'], errors='coerce')
    df_rearrangements.dropna(subset=['START', 'END'], inplace=True)
    
    df_rearrangements['SIZE'] = df_rearrangements['END'] - df_rearrangements['START']
    SIZE_THRESHOLD = 10000
    df_rearrangements = df_rearrangements[df_rearrangements['SIZE'].abs() > SIZE_THRESHOLD].copy()
    
    # Clean chromosome column to match main data (remove 'scaffold_')
    df_rearrangements['CHR'] = df_rearrangements['CHR_STR'].str.replace('scaffold_', '').astype(int)
    
    # Merge the base position data to the rearrangements dataframe
    df_plot_chr_info = df_plot[['CHR', 'base_pos']].drop_duplicates()
    df_rearrangements = pd.merge(df_rearrangements, df_plot_chr_info, on='CHR', how='left')
    df_rearrangements.dropna(subset=['base_pos'], inplace=True)
    df_rearrangements['START_cum'] = df_rearrangements['base_pos'] + df_rearrangements['START']
    df_rearrangements['END_cum'] = df_rearrangements['base_pos'] + df_rearrangements['END']
    has_rearrangements_to_plot = True

# --- SECTION: Statistical Analysis for Top 30 Rearrangements ---
if has_rearrangements_to_plot and not df_rearrangements.empty:
    print("\n--- Top 30 Chromosome Rearrangements ---")
    df_rearrangements['SIZE_ABS'] = df_rearrangements['SIZE'].abs()
    top_30_rearrangements = df_rearrangements.sort_values(by='SIZE_ABS', ascending=False).head(30).copy()
    
    # Print the top 30 list
    print(top_30_rearrangements[['CHR', 'START', 'END', 'TYPE', 'SIZE_ABS']])

    # --- Statistical Analysis for each of the top 30 rearrangements ---
    print("\nPerforming statistical analysis for each of the top 30 rearrangements...")
    
    # Pre-calculate boolean for all points outside any of the top 30 for efficiency
    # (Since we are testing against "rest of the genome")
    
    # Perform analysis for each of the top 30 individually
    for index, row in top_30_rearrangements.iterrows():
        # Identify SNPs within this specific rearrangement
        is_in_current_rearrangement = (df_plot['BPcum'] >= row['START_cum']) & (df_plot['BPcum'] <= row['END_cum'])
        
        # Mann-Whitney U test for Fst values within this rearrangement
        fst_in_rearrangement = df_plot[is_in_current_rearrangement]['FST'].dropna()
        fst_not_in_rearrangement = df_plot[~is_in_current_rearrangement]['FST'].dropna()
        
        print(f"\n--- Fst vs. Rearrangement (Chr: {row['CHR']}, Type: {row['TYPE']}, Size: {row['SIZE_ABS']:.0f}) ---")
        if len(fst_in_rearrangement) > 0 and len(fst_not_in_rearrangement) > 0:
            stat_top30, p_val_mw_top30 = stats.mannwhitneyu(fst_in_rearrangement, fst_not_in_rearrangement, alternative='greater')
            print("Mann-Whitney U Test Results:")
            print(f"P-value: {p_val_mw_top30:.4f}")
            if p_val_mw_top30 < 0.05:
                print("Conclusion: The difference is statistically significant (Fst higher in rearrangement).")
            else:
                print("Conclusion: The difference is not statistically significant.")
        else:
            print("Could not perform Mann-Whitney U test. Insufficient data in one or both groups.")

        # Chi-squared test for RDA loci within this rearrangement
        print(f"\n--- RDA Significant Loci vs. Rearrangement (Chr: {row['CHR']}, Type: {row['TYPE']}, Size: {row['SIZE_ABS']:.0f}) ---")
        sig_rda_in_rearr = df_plot[df_plot['is_significant_rda'] & is_in_current_rearrangement].shape[0]
        sig_rda_not_in_rearr = df_plot[df_plot['is_significant_rda'] & ~is_in_current_rearrangement].shape[0]
        non_sig_rda_in_rearr = df_plot[~df_plot['is_significant_rda'] & is_in_current_rearrangement].shape[0]
        non_sig_rda_not_in_rearr = df_plot[~df_plot['is_significant_rda'] & ~is_in_current_rearrangement].shape[0]

        contingency_table = np.array([[sig_rda_in_rearr, sig_rda_not_in_rearr],
                                     [non_sig_rda_in_rearr, non_sig_rda_not_in_rearr]])
        
        try:
            chi2, p_val_chi2, _, _ = stats.chi2_contingency(contingency_table)
            print("Chi-squared Test Results:")
            print(f"P-value: {p_val_chi2:.4f}")
            if p_val_chi2 < 0.05:
                print("Conclusion: The difference is statistically significant (RDA loci significantly associated with this rearrangement).")
            else:
                print("Conclusion: The difference is not statistically significant.")
        except ValueError:
            print("Could not perform Chi-squared test. One or more categories may have zero observations.")

    # Mark all SNPs within *any* of the top 30 rearrangements for the overall plot display
    df_plot['is_in_rearrangement'] = False
    for index, row in top_30_rearrangements.iterrows():
        df_plot.loc[(df_plot['BPcum'] >= row['START_cum']) & (df_plot['BPcum'] <= row['END_cum']), 'is_in_rearrangement'] = True


# --- SECTION: Statistical Analysis for ALL rearrangements ---
print("\n--- Fst vs. ALL Rearrangements ---")
if has_rearrangements_to_plot:
    # Mark all SNPs within *all* rearrangements for the final analysis
    df_plot['is_in_rearrangement'] = False
    for index, row in df_rearrangements.iterrows():
        df_plot.loc[(df_plot['BPcum'] >= row['START_cum']) & (df_plot['BPcum'] <= row['END_cum']), 'is_in_rearrangement'] = True
        
    fst_in_rearrangements = df_plot[df_plot['is_in_rearrangement']]['FST'].dropna()
    fst_not_in_rearrangements = df_plot[~df_plot['is_in_rearrangement']]['FST'].dropna()

    if len(fst_in_rearrangements) > 0 and len(fst_not_in_rearrangements) > 0:
        stat, p_val_mw = stats.mannwhitneyu(fst_in_rearrangements, fst_not_in_rearrangements, alternative='greater')
        print("Mann-Whitney U Test Results (All Rearrangements):")
        print(f"U statistic: {stat:.4f}")
        print(f"P-value: {p_val_mw:.4f}")
        if p_val_mw < 0.05:
            print("\nConclusion: The difference is statistically significant.")
            print("Fst values are significantly higher in the rearranged areas compared to the rest of the genome.")
        else:
            print("\nConclusion: The difference is not statistically significant.")
    else:
        print("Could not perform Mann-Whitney U test. One or both groups have no data.")
else:
    print("Could not perform Mann-Whitney U test. No rearrangement data was found.")

# --- SECTION: Plotting the data ---
print("\nGenerating the multipanel plot...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
fig.suptitle("Genomic Loci: Fst, RDA, and Rearrangements", fontsize=16)

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
ax1.set_title("Fst Manhattan Plot", fontsize=14)

# Plot rearrangement regions
if has_rearrangements_to_plot:
    if not df_rearrangements.empty:
        # Use only the top 30 for coloring, or all if preferred, but for plot clarity, all is usually better
        for index, row in df_rearrangements.iterrows():
            ax1.axvspan(row['START_cum'], row['END_cum'], color='pink', alpha=0.3, zorder=0)
        # Add legend placeholder (using the threshold from the SyRI processing)
        ax1.axvspan(0, 0, color='pink', alpha=0.3, label=f'Rearrangements (>{SIZE_THRESHOLD} bp)')
ax1.legend(loc='upper left')

# --- BOTTOM PLOT: RDA ---
for i, chr_num in enumerate(df_plot['CHR'].unique()):
    chr_df = df_plot[df_plot['CHR'] == chr_num]
    ax2.scatter(x=chr_df['BPcum'], y=chr_df['distance'], s=5, c=colors[i % 2], zorder=1)
    
# Highlight significant RDA SNPs (red dots)
df_significant_rda = df_plot[df_plot['is_significant_rda']]
ax2.scatter(x=df_significant_rda['BPcum'], y=df_significant_rda['distance'], s=30, c='red', zorder=3, label='RDA-Significant SNPs')

# Plot rearrangement regions
if has_rearrangements_to_plot:
    if not df_rearrangements.empty:
        for index, row in df_rearrangements.iterrows():
            ax2.axvspan(row['START_cum'], row['END_cum'], color='pink', alpha=0.3, zorder=0)
        ax2.axvspan(0, 0, color='pink', alpha=0.3, label=f'Rearrangements (>{SIZE_THRESHOLD} bp)')
ax2.legend(loc='upper left')
ax2.set_xlabel("Genomic Position", fontsize=12)
ax2.set_ylabel("RDA Score", fontsize=12)
ax2.set_title("RDA Manhattan Plot", fontsize=14)

# Set x-axis labels on the bottom plot only
plt.xticks(x_label_pos, x_labels)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# --- SAVE FIGURES ---
print(f"Saving plot to {output_png}...")
plt.savefig(output_png, dpi=300)

# The new PDF saving command
print(f"Saving plot to {output_pdf}...")
plt.savefig(output_pdf, format='pdf', dpi=300) 

print("Plot saved as multipanel_genomic_plot.png and multipanel_genomic_plot.pdf")
