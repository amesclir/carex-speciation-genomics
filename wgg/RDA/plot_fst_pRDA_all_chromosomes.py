import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats

# --- CONFIGURATION ---
TARGET_CHROMOSOMES = [3, 14, 18, 20, 28, 32]
VISUAL_THRESHOLD_QUANTILE = 0.95
SIZE_THRESHOLD = 80000

# Global file paths
syri_file_path = "/home/aescudero/syri/syri_final2.out"
fst_file_path = "admixture_final_ld_pruned_filtered_pop1_vs_pop2.fst.sorted"

# To store summary statistics for the final printout
summary_results = []

# --- STEP 1: LOAD GLOBAL DATASETS ONCE ---
print("Loading global Fst and SyRI datasets (this only happens once)...")

# Load FST
try:
    df_fst_global = pd.read_csv(fst_file_path, sep='\t', header=None, 
                             names=['CHR', 'blank', 'POS', 'N_INDV', 'FST'],
                             dtype={'CHR': int, 'POS': int, 'FST': float})
    df_fst_global = df_fst_global.drop(columns=['blank', 'N_INDV'])
except Exception as e:
    print(f"Error loading Fst file: {e}")
    exit()

# Load and Process SyRI
has_rearrangements = False
try:
    df_raw_syri = pd.read_csv(syri_file_path, sep=r'\s+', header=None, low_memory=False)
    df_raw_syri = df_raw_syri.rename(columns={10: 'TYPE'})
    rearrangement_types = ['INV', 'DUP', 'TRANS', 'INVDP', 'INVTR']
    df_syri_global = df_raw_syri[df_raw_syri['TYPE'].isin(rearrangement_types)].copy()
    
    df_syri_global['CHR'] = np.where(df_syri_global[0] != '-', df_syri_global[0], df_syri_global[5])
    df_syri_global['START'] = np.where(df_syri_global[1] != '-', df_syri_global[1], df_syri_global[6])
    df_syri_global['END'] = np.where(df_syri_global[2] != '-', df_syri_global[2], df_syri_global[7])
    
    df_syri_global = df_syri_global[df_syri_global['CHR'].str.contains('scaffold_')].copy()
    df_syri_global['START'] = pd.to_numeric(df_syri_global['START'], errors='coerce')
    df_syri_global['END'] = pd.to_numeric(df_syri_global['END'], errors='coerce')
    df_syri_global.dropna(subset=['START', 'END'], inplace=True)
    df_syri_global['SIZE'] = df_syri_global['END'] - df_syri_global['START']
    
    df_syri_global = df_syri_global[df_syri_global['SIZE'].abs() > SIZE_THRESHOLD].copy()
    df_syri_global['CHR_int'] = df_syri_global['CHR'].str.replace('scaffold_', '').astype(int)
    has_rearrangements = True
except Exception as e:
    print(f"Warning: SyRI data could not be processed: {e}")


# --- STEP 2: LOOP THROUGH EACH CHROMOSOME ---
for chrom in TARGET_CHROMOSOMES:
    print(f"\n{'='*50}")
    print(f"PROCESSING CHROMOSOME {chrom}")
    print(f"{'='*50}")
    
    rda_scores_path = f"rda_snp_scores_chr{chrom}.csv"
    selected_snps_path = f"selected_snps_for_plotting_chr{chrom}.csv"
    
    if not os.path.exists(rda_scores_path) or not os.path.exists(selected_snps_path):
        print(f"Skipping Chr {chrom}: Missing RDA CSV files.")
        continue

    # Load RDA data
    snp_scores = pd.read_csv(rda_scores_path)
    selected_snps_df = pd.read_csv(selected_snps_path)
    
    valid_snp_ids = selected_snps_df['SNP_ID'].astype(str).str.contains(r'^\d+_\d+$')
    selected_snps_df = selected_snps_df[valid_snp_ids].copy()
    snp_scores = snp_scores[snp_scores['SNP_ID'].isin(selected_snps_df['SNP_ID'])]
    snp_scores.set_index('SNP_ID', inplace=True)
    
    # Calculate Multidimensional pRDA Distance (Environmental Signal)
    snp_scores['distance'] = np.sqrt(snp_scores['RDA1']**2 + snp_scores['RDA2']**2)
    
    # Setup Plot Dataframe
    all_snps = selected_snps_df['SNP_ID'].tolist()
    df_plot = pd.DataFrame(index=all_snps)
    df_plot['SNP_ID'] = df_plot.index
    df_plot['CHR'] = df_plot['SNP_ID'].str.split('_').str[0].astype(int)
    df_plot['POS'] = df_plot['SNP_ID'].str.split('_').str[1].astype(int)
    df_plot = df_plot.sort_values(['CHR', 'POS']).reset_index(drop=True)
    
    # Merge with Global Fst
    df_plot = pd.merge(df_plot, df_fst_global[['CHR', 'POS', 'FST']], on=['CHR', 'POS'], how='left')
    df_plot = pd.merge(df_plot, snp_scores[['distance']], on='SNP_ID', how='left')
    
    # Significance Thresholds
    rda_threshold = df_plot['distance'].quantile(VISUAL_THRESHOLD_QUANTILE)
    df_plot['is_significant_rda'] = df_plot['distance'] > rda_threshold
    fst_threshold = df_plot['FST'].quantile(VISUAL_THRESHOLD_QUANTILE) if not df_plot['FST'].isnull().all() else 0
    df_plot['is_significant_fst'] = df_plot['FST'] > fst_threshold
    
    # Map Rearrangements for this specific chromosome
    df_plot['is_in_in_rearrangement'] = False
    chr_rearrangements = pd.DataFrame()
    
    if has_rearrangements:
        chr_rearrangements = df_syri_global[df_syri_global['CHR_int'] == chrom]
        for index, row in chr_rearrangements.iterrows():
            # Since it's a single chromosome plot, BPcum is just POS
            df_plot.loc[(df_plot['POS'] >= row['START']) & (df_plot['POS'] <= row['END']), 'is_in_in_rearrangement'] = True

    # --- STATISTICAL TESTS ---
    pval_rda, pval_fst = "N/A", "N/A"
    
    if has_rearrangements and not chr_rearrangements.empty:
        # RDA Test
        rda_in = df_plot[df_plot['is_in_in_rearrangement']]['distance'].dropna()
        rda_out = df_plot[~df_plot['is_in_in_rearrangement']]['distance'].dropna()
        if len(rda_in) > 0 and len(rda_out) > 0:
            stat, p_val = stats.mannwhitneyu(rda_in, rda_out, alternative='greater')
            pval_rda = p_val
            
        # FST Test
        fst_in = df_plot[df_plot['is_in_in_rearrangement']]['FST'].dropna()
        fst_out = df_plot[~df_plot['is_in_in_rearrangement']]['FST'].dropna()
        if len(fst_in) > 0 and len(fst_out) > 0:
            stat, p_val = stats.mannwhitneyu(fst_in, fst_out, alternative='greater')
            pval_fst = p_val
            
    # Save to summary
    summary_results.append({
        'Chromosome': chrom,
        'SVs_Found': len(chr_rearrangements),
        'P-Value_Fst': f"{pval_fst:.4f}" if isinstance(pval_fst, float) else pval_fst,
        'P-Value_pRDA': f"{pval_rda:.4f}" if isinstance(pval_rda, float) else pval_rda
    })

    # --- PLOTTING ---
    print(f"Generating multipanel plot for Chromosome {chrom}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # Plot FST
    ax1.scatter(x=df_plot['POS'], y=df_plot['FST'], s=5, c='#528585', zorder=1)
    sig_fst = df_plot[df_plot['is_significant_fst']]
    ax1.scatter(x=sig_fst['POS'], y=sig_fst['FST'], s=30, c='red', zorder=3, label=f'Top {100-VISUAL_THRESHOLD_QUANTILE*100:.0f}% Fst Outliers')
    
    # Plot RDA
    ax2.scatter(x=df_plot['POS'], y=df_plot['distance'], s=5, c='#528585', zorder=1)
    sig_rda = df_plot[df_plot['is_significant_rda']]
    ax2.scatter(x=sig_rda['POS'], y=sig_rda['distance'], s=30, c='red', zorder=3, label=f'Top {100-VISUAL_THRESHOLD_QUANTILE*100:.0f}% pRDA Outliers')
    
    # Add Rearrangement Highlights
    if not chr_rearrangements.empty:
        for index, row in chr_rearrangements.iterrows():
            ax1.axvspan(row['START'], row['END'], color='pink', alpha=0.3, zorder=0)
            ax2.axvspan(row['START'], row['END'], color='pink', alpha=0.3, zorder=0)
        ax1.axvspan(0, 0, color='pink', alpha=0.3, label='Rearrangement')
        ax2.axvspan(0, 0, color='pink', alpha=0.3, label='Rearrangement')
        
    ax1.set_ylabel("Fst", fontsize=12)
    ax1.set_title(f"Chromosome {chrom}: Fst Manhattan Plot", fontsize=14)
    ax1.legend(loc='upper right')
    
    ax2.set_xlabel(f"Genomic Position (Chromosome {chrom})", fontsize=12)
    ax2.set_ylabel("pRDA Distance (Environmental Signal)", fontsize=12)
    ax2.set_title(f"Chromosome {chrom}: Partial RDA Manhattan Plot", fontsize=14)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"multipanel_genomic_plot_chr{chrom}.png", dpi=300)
    plt.savefig(f"multipanel_genomic_plot_chr{chrom}.pdf", format='pdf', dpi=300)
    plt.close(fig) # Frees up memory after each loop iteration

# --- STEP 3: PRINT SUMMARY TABLE ---
print("\n\n" + "="*60)
print("FINAL GENOME-WIDE STATISTICAL SUMMARY")
print("="*60)
print(f"{'Chrom':<10} | {'SVs Found':<12} | {'Fst P-Value (Spec)':<20} | {'pRDA P-Value (Env)':<20}")
print("-" * 60)
for res in summary_results:
    print(f"{res['Chromosome']:<10} | {res['SVs_Found']:<12} | {res['P-Value_Fst']:<20} | {res['P-Value_pRDA']:<20}")
print("="*60)
print("Analysis complete. All PDFs and PNGs have been saved to your directory.")

