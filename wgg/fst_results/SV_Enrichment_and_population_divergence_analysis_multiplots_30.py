#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import os
from scipy import stats

def compare_single_sv(sv_data, bg_data, stat_col):
    """
    Compares the distribution of a statistic (FST or PI_RATIO) for a single SV
    against the genomic background using the Mann-Whitney U test.
    """
    sv_values = sv_data[stat_col].dropna()
    bg_values = bg_data[stat_col].dropna()
    
    # Ensure all required keys are returned even for insufficient data
    if len(sv_values) < 5 or len(bg_values) < 5:
        return {
            'mean_SV': np.nan, 
            'mean_BG': np.nan, 
            'p_value': np.nan, 
            'p_format': 'N/A',
            'conclusion': 'Insufficient data'
        }

    # Determine alternative hypothesis based on the statistic
    if stat_col == 'FST':
        # Expect FST to be higher in SVs compared to background
        test_alternative = 'greater' 
    else: # PI_RATIO
        # Expect PI_RATIO distribution to be different (higher or lower) from background. 
        test_alternative = 'two-sided' 

    u_stat, p_value = stats.mannwhitneyu(sv_values, bg_values, alternative=test_alternative, use_continuity=False)
    
    # Format the conclusion
    if p_value < 0.001:
        p_format = "< 0.001"
        conclusion = "Significant"
    elif p_value < 0.05:
        p_format = f"{p_value:.3f}"
        conclusion = "Significant"
    else:
        p_format = f"{p_value:.3f}"
        conclusion = "Not Significant"

    return {
        'mean_SV': sv_values.mean(),
        'mean_BG': bg_values.mean(),
        'p_value': p_value,
        'p_format': p_format,
        'conclusion': conclusion
    }

def main():
    """
    Loads windowed Fst and Pi data, structural variant (SyRI) data, 
    and performs population divergence analysis, including individual SV tests using PI_RATIO.
    """
    
    # --- USER-DEFINED FILE PATHS (Absolute for robustness) ---
    BASE_WGG_DIR = "/home/aescudero/wgg/" 
    FST_WINDOW_FILE = os.path.join(BASE_WGG_DIR, "fst_results/windowed_stats/windowed_fst_1000bp.windowed.weir.fst")
    PI_POP1_WINDOW_FILE = os.path.join(BASE_WGG_DIR, "fst_results/windowed_stats/windowed_pi_pop1_1000bp.windowed.pi")
    PI_POP2_WINDOW_FILE = os.path.join(BASE_WGG_DIR, "fst_results/windowed_stats/windowed_pi_pop2_1000bp.windowed.pi")
    SYRI_FILE_PATH = "/home/aescudero/syri/syri.out" 

    # --- CONFIGURATION ---
    WINDOW_SIZE = 1000
    SV_SIZE_THRESHOLD = 10000
    K_TOP_SVS = 30
    PLOTS_OUTPUT_DIR = "sv_density_plots" # New directory for individual plots

    # Create output directory for density plots
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)


    # --- SECTION 1: Load and Process Windowed Statistics ---
    print("Loading and processing windowed statistics...")

    # 1a. Load Fst windows 
    if not os.path.exists(FST_WINDOW_FILE):
        print(f"Error: FST file not found at {FST_WINDOW_FILE}. Please check the path.")
        return
        
    df_fst = pd.read_csv(FST_WINDOW_FILE, sep='\t')
    df_fst.dropna(subset=['WEIGHTED_FST'], inplace=True) 
    df_fst = df_fst.rename(columns={'WEIGHTED_FST': 'FST', 'BIN_START': 'POS'})
    df_fst['CHR_int'] = df_fst['CHROM'].astype(str).str.replace('scaffold_', '', regex=False).astype(int)

    # 1b. Load Pi windows
    if not os.path.exists(PI_POP1_WINDOW_FILE) or not os.path.exists(PI_POP2_WINDOW_FILE):
        print("Error: One or both PI files not found. Please check paths.")
        return

    df_pi1 = pd.read_csv(PI_POP1_WINDOW_FILE, sep='\t')
    df_pi2 = pd.read_csv(PI_POP2_WINDOW_FILE, sep='\t')

    df_pi_merged = pd.merge(df_pi1[['CHROM', 'BIN_START', 'PI']], 
                            df_pi2[['CHROM', 'BIN_START', 'PI']], 
                            on=['CHROM', 'BIN_START'], 
                            suffixes=('_POP1', '_POP2'))

    # NEW METRIC: Calculate the PI Ratio (Pop1 / Pop2)
    # We add a small epsilon to the denominator to prevent division by zero
    EPSILON = 1e-6
    df_pi_merged['PI_RATIO'] = df_pi_merged['PI_POP1'] / (df_pi_merged['PI_POP2'] + EPSILON)
    
    df_pi_merged = df_pi_merged.rename(columns={'BIN_START': 'POS'})
    df_pi_merged['CHR_int'] = df_pi_merged['CHROM'].astype(str).str.replace('scaffold_', '', regex=False).astype(int)

    # 1c. Combine Fst and Pi data
    df_windows = pd.merge(df_fst[['CHR_int', 'POS', 'FST']], 
                          df_pi_merged[['CHR_int', 'POS', 'PI_RATIO']], # Use PI_RATIO here
                          on=['CHR_int', 'POS'], 
                          how='inner')

    print(f"Total {WINDOW_SIZE}bp windows analyzed: {len(df_windows)}")

    # --- SECTION 2: Load and Process SyRI Data (Top K Largest SVs) ---
    print("\nLoading and processing SyRI data...")
    
    if not os.path.exists(SYRI_FILE_PATH):
        print(f"Error: SyRI file not found at {SYRI_FILE_PATH}. Please check the path.")
        return

    # Added low_memory=False to potentially handle DtypeWarning
    df_raw_syri = pd.read_csv(SYRI_FILE_PATH, sep='\s+', header=None, dtype={0: str, 5: str}, on_bad_lines='skip', low_memory=False)
    df_raw_syri = df_raw_syri.rename(columns={10: 'TYPE'})

    rearrangement_types = ['INV', 'DUP', 'TRANS', 'INVTR', 'INVDP']
    df_rearrangements = df_raw_syri[df_raw_syri['TYPE'].isin(rearrangement_types)].copy()

    df_rearrangements['CHR'] = np.where(df_rearrangements[0].astype(str) != '-', df_rearrangements[0], df_rearrangements[5])
    df_rearrangements['START'] = np.where(df_rearrangements[1].astype(str) != '-', df_rearrangements[1], df_rearrangements[6])
    df_rearrangements['END'] = np.where(df_rearrangements[2].astype(str) != '-', df_rearrangements[2], df_rearrangements[7])

    df_rearrangements = df_rearrangements[df_rearrangements['CHR'].astype(str).str.contains('scaffold_', na=False)].copy()
    df_rearrangements['START'] = pd.to_numeric(df_rearrangements['START'], errors='coerce').astype('Int64')
    df_rearrangements['END'] = pd.to_numeric(df_rearrangements['END'], errors='coerce').astype('Int64')
    df_rearrangements.dropna(subset=['START', 'END'], inplace=True)

    df_rearrangements['SIZE'] = (df_rearrangements['END'] - df_rearrangements['START']).abs()
    df_rearrangements = df_rearrangements[df_rearrangements['SIZE'] > SV_SIZE_THRESHOLD].copy()
    df_rearrangements['CHR_int'] = df_rearrangements['CHR'].astype(str).str.replace('scaffold_', '', regex=False).astype(int) 

    df_top_svs = df_rearrangements.sort_values('SIZE', ascending=False).head(K_TOP_SVS).copy().reset_index(drop=True)
    df_top_svs['SV_ID'] = [f"SV{i+1}" for i in df_top_svs.index]


    if df_top_svs.empty:
        print("Warning: No large SVs found to analyze. Exiting analysis and plotting steps.")
        return

    print(f"Analyzing enrichment within the top {K_TOP_SVS} largest SVs (>{SV_SIZE_THRESHOLD} bp):")
    print(df_top_svs[['SV_ID', 'TYPE', 'CHR_int', 'START', 'END', 'SIZE']].to_string(index=False))

    # --- SECTION 3: Overlap Windows with Top SVs (Original logic for whole SV group) ---

    def is_window_in_sv(row, sv_df):
        """Checks if a window (defined by CHR_int and POS) overlaps with any SV region."""
        window_start = row['POS']
        window_end = row['POS'] + WINDOW_SIZE
        
        svs_on_chr = sv_df[sv_df['CHR_int'] == row['CHR_int']]
        
        for _, sv_row in svs_on_chr.iterrows():
            sv_start = min(sv_row['START'], sv_row['END'])
            sv_end = max(sv_row['START'], sv_row['END'])
            
            if max(window_start, sv_start) < min(window_end, sv_end):
                return True
        return False

    # Flag windows that fall within the top K SV regions
    df_windows['In_Top_SV'] = df_windows.apply(lambda row: is_window_in_sv(row, df_top_svs), axis=1)

    # Split data into SV and Background groups
    df_sv_group = df_windows[df_windows['In_Top_SV']]
    df_bg = df_windows[~df_windows['In_Top_SV']]

    print(f"\nWindows In Top SVs (Group): {len(df_sv_group)}")
    print(f"Windows In Background: {len(df_bg)}")

    # --- SECTION 4: Individual SV Statistical Analysis and Table Generation ---
    print("\n" + "="*80)
    print(f"--- Individual Statistical Tests (Mann-Whitney U) for Top {K_TOP_SVS} SVs vs. Background ---")
    print("="*80)
    
    individual_test_results = []
    
    # Note: We iterate through the original top 10 list to ensure all are processed.
    for _, sv_row in df_top_svs.iterrows():
        # Get windows fully contained within this specific SV
        sv_start_safe = min(sv_row['START'], sv_row['END'])
        sv_end_safe = max(sv_row['START'], sv_row['END'])
        
        sv_loci = df_windows[
            (df_windows['CHR_int'] == sv_row['CHR_int']) &
            (df_windows['POS'] >= sv_start_safe) &
            (df_windows['POS'] + WINDOW_SIZE <= sv_end_safe) 
        ]

        # FST Test
        fst_res = compare_single_sv(sv_loci, df_bg, 'FST')
        # PI RATIO Test
        pi_res = compare_single_sv(sv_loci, df_bg, 'PI_RATIO')

        individual_test_results.append({
            'SV_ID': sv_row['SV_ID'],
            'TYPE': sv_row['TYPE'],
            'CHR': sv_row['CHR_int'],
            'SIZE_kb': sv_row['SIZE'] / 1000,
            'Windows': len(sv_loci),
            'FST_Mean': fst_res['mean_SV'],
            'FST_P': fst_res['p_format'],
            'FST_Sig': fst_res['conclusion'],
            'PI_Ratio_Mean': pi_res['mean_SV'], 
            'PI_Ratio_P': pi_res['p_format'],   
            'PI_Ratio_Sig': pi_res['conclusion'], 
        })

    df_individual_tests = pd.DataFrame(individual_test_results)
    # Only keep SVs that had enough windows for at least one meaningful test
    df_individual_tests = df_individual_tests[
        (df_individual_tests['FST_P'] != 'N/A') | (df_individual_tests['PI_Ratio_P'] != 'N/A')
    ]

    # Print the new table
    print(df_individual_tests.to_string(index=False, float_format="%.4f"))
    print("\nNote: FST test alternative: SV > Background. PI_RATIO test alternative: SV $\\ne$ Background (two-sided).")
    print("Windows column indicates windows fully contained within the SV region.")

    # --- SECTION 5: Density Plotting for Individual SVs ---
    print("\n" + "="*80)
    print(f"--- Generating Individual Density Plots in '{PLOTS_OUTPUT_DIR}/' ---")
    print("="*80)
    
    # Pre-calculate the background 95th percentile once for standardization
    bg_pi_ratios = df_bg['PI_RATIO'].dropna()
    q95_bg = bg_pi_ratios.quantile(0.95) if not bg_pi_ratios.empty else 6.0
    # Ensure a reasonable minimum viewing limit (e.g., 2.0)
    standard_x_limit = max(q95_bg, 2.0) 
    
    # --- CONFIRMATION FOR USER ---
    print(f"\nPI Ratio X-axis Standardization:")
    print(f"95th Percentile of Genomic Background: {q95_bg:.4f}")
    print(f"Standardized X-axis Upper Limit (max of 95th percentile or 2.0): {standard_x_limit:.4f}")
    print("----------------------------------------------------------------")
    
    for _, sv_row in df_top_svs.iterrows():
        sv_start_safe = min(sv_row['START'], sv_row['END'])
        sv_end_safe = max(sv_row['START'], sv_row['END'])
        
        sv_loci = df_windows[
            (df_windows['CHR_int'] == sv_row['CHR_int']) &
            (df_windows['POS'] >= sv_start_safe) &
            (df_windows['POS'] + WINDOW_SIZE <= sv_end_safe) 
        ]

        # Check if the SV was tested (i.e., had enough windows)
        if len(sv_loci) < 5:
            print(f"Skipping {sv_row['SV_ID']} (Chr{sv_row['CHR_int']}, {sv_row['TYPE']}) due to insufficient data ({len(sv_loci)} windows).")
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        sv_name = f"{sv_row['SV_ID']} ({sv_row['TYPE']} on Chr{sv_row['CHR_int']})"
        
        # Plot 1: FST Distribution
        sns.kdeplot(df_bg['FST'].dropna(), ax=axes[0], color='gray', fill=True, alpha=0.5, label='Genomic Background')
        sns.kdeplot(sv_loci['FST'].dropna(), ax=axes[0], color='darkred', fill=True, alpha=0.7, label=sv_name)
        
        # Add means as vertical lines
        axes[0].axvline(df_bg['FST'].mean(), color='gray', linestyle='--', linewidth=1)
        axes[0].axvline(sv_loci['FST'].mean(), color='darkred', linestyle='-', linewidth=1)

        axes[0].set_title(f"FST Distribution: {sv_name}")
        axes[0].set_xlabel("FST")
        axes[0].legend()

        # Plot 2: PI Ratio Distribution
        sns.kdeplot(df_bg['PI_RATIO'].dropna(), ax=axes[1], color='gray', fill=True, alpha=0.5, label='Genomic Background')
        sns.kdeplot(sv_loci['PI_RATIO'].dropna(), ax=axes[1], color='darkblue', fill=True, alpha=0.7, label=sv_name)

        # Add means and the neutral ratio (1.0) as vertical lines
        axes[1].axvline(1.0, color='black', linestyle=':', linewidth=1, label='Neutral Ratio (1.0)')
        axes[1].axvline(df_bg['PI_RATIO'].mean(), color='gray', linestyle='--', linewidth=1)
        axes[1].axvline(sv_loci['PI_RATIO'].mean(), color='darkblue', linestyle='-', linewidth=1)
        
        # Set the standardized x-axis limit based on background 95th percentile
        axes[1].set_xlim(0, standard_x_limit)
        
        # Update title to reflect the new, standardized cap
        axes[1].set_title(f"$\\pi_{{Pop1}}/\\pi_{{Pop2}}$ Ratio Distribution: {sv_name} (Axis capped at BG 95th percentile: {standard_x_limit:.2f})")
        axes[1].set_xlabel("$\\pi_{\\text{Pop1}}/\\pi_{\\text{Pop2}}$ Ratio")
        
        axes[1].legend()

        plt.suptitle(f'Divergence and Diversity Ratio Profile of {sv_name} ({sv_row["SIZE"]/1000:.1f} kb)', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        filename = os.path.join(PLOTS_OUTPUT_DIR, f"{sv_row['SV_ID']}_{sv_row['TYPE']}_Chr{sv_row['CHR_int']}_ratio_density.png")
        plt.savefig(filename, dpi=300)
        plt.close(fig) # Close the figure to free memory
        print(f"Generated {filename}")

    print("\n--- Analysis Complete ---")
    
if __name__ == "__main__":
    main()
