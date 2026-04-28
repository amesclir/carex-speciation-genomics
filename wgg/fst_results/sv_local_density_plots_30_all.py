#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import os
from scipy import stats

def compare_distributions(sv_data, bg_data, stat_col, alternative='two-sided'):
    """
    Compares the distribution of a statistic for an SV against its local background
    using the Mann-Whitney U test, with a customizable alternative hypothesis.
    
    Args:
        sv_data (pd.DataFrame): Windows contained within the SV.
        bg_data (pd.DataFrame): Windows on the same chromosome outside the SV (local background).
        stat_col (str): The column name (e.g., 'FST', 'PI_RATIO').
        alternative (str): The alternative hypothesis ('greater', 'less', or 'two-sided').
    
    Returns:
        dict: Test results including means, p-value, and conclusion.
    """
    sv_values = sv_data[stat_col].dropna()
    bg_values = bg_data[stat_col].dropna()
    
    # Ensure sufficient data for a robust test
    if len(sv_values) < 5 or len(bg_values) < 5:
        return {
            'mean_SV': np.nan, 
            'p_value': np.nan, 
            'p_format': 'N/A',
            'conclusion': 'Insufficient data'
        }

    # Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(sv_values, bg_values, alternative=alternative, use_continuity=False)
    
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

def get_windows_outside_sv(df_chr, sv_start, sv_end, window_size):
    """Filters windows on a chromosome that are entirely outside the given SV region."""
    window_start = df_chr['POS']
    window_end = df_chr['POS'] + window_size
    # A window is OUTSIDE the SV if (window_end <= sv_start) OR (window_start >= sv_end)
    is_outside = (window_end <= sv_start) | (window_start >= sv_end)
    return df_chr[is_outside].copy()

def main():
    """
    Loads windowed data, performs local background comparison for top SVs, 
    and generates 4-panel density plots.
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
    PLOTS_OUTPUT_DIR = "sv_local_density_plots" # Updated directory name

    # Create output directory for density plots
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    
    # Define the alternative hypothesis for each metric (for Mann-Whitney U test)
    # FST is tested for enrichment (SV > BG)
    # Pi values and Ratio are tested for difference (SV != BG)
    METRICS_TO_TEST = {
        'FST': {'label': 'FST', 'alt': 'greater', 'color': 'darkred'}, 
        'PI_POP1': {'label': '$\\pi_{\\text{Pop1}}$', 'alt': 'two-sided', 'color': 'cyan'}, 
        'PI_POP2': {'label': '$\\pi_{\\text{Pop2}}$', 'alt': 'two-sided', 'color': 'orange'}, 
        'PI_RATIO': {'label': '$\\pi_{\\text{Pop1}}/\\pi_{\\text{Pop2}}$ Ratio', 'alt': 'two-sided', 'color': 'darkblue'}
    }

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

    # Calculate the PI Ratio (Pop1 / Pop2)
    EPSILON = 1e-6
    df_pi_merged['PI_RATIO'] = df_pi_merged['PI_POP1'] / (df_pi_merged['PI_POP2'] + EPSILON)
    
    df_pi_merged = df_pi_merged.rename(columns={'BIN_START': 'POS', 'PI_POP1': 'PI_POP1', 'PI_POP2': 'PI_POP2'})
    df_pi_merged['CHR_int'] = df_pi_merged['CHROM'].astype(str).str.replace('scaffold_', '', regex=False).astype(int)

    # 1c. Combine Fst and Pi data
    df_windows = pd.merge(df_fst[['CHR_int', 'POS', 'FST']], 
                            df_pi_merged[['CHR_int', 'POS', 'PI_RATIO', 'PI_POP1', 'PI_POP2']], 
                            on=['CHR_int', 'POS'], 
                            how='inner')

    print(f"Total {WINDOW_SIZE}bp windows analyzed: {len(df_windows)}")

    # --- SECTION 2: Load and Process SyRI Data (Top K Largest SVs) ---
    print("\nLoading and processing SyRI data...")
    
    if not os.path.exists(SYRI_FILE_PATH):
        print(f"Error: SyRI file not found at {SYRI_FILE_PATH}. Please check the path.")
        return

    df_raw_syri = pd.read_csv(SYRI_FILE_PATH, sep='\s+', header=None, dtype={0: str, 5: str}, on_bad_lines='skip', low_memory=False)
    df_raw_syri = df_raw_syri.rename(columns={10: 'TYPE'})

    rearrangement_types = ['INV', 'DUP', 'TRANS', 'INVTR', 'INVDP']
    df_rearrangements = df_raw_syri[df_raw_syri['TYPE'].isin(rearrangement_types)].copy()

    # Determine CHR, START, END robustly across translocations and inversions
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

    # --- SECTION 3: Individual SV Statistical Analysis and Table Generation (vs. Local Background) ---
    print("\n" + "="*80)
    print(f"--- Individual Statistical Tests (Mann-Whitney U) for Top {K_TOP_SVS} SVs vs. Local Chromosome Background ---")
    print("="*80)
    
    individual_test_results = []
    
    # Pre-fetch all windows grouped by chromosome
    df_windows_by_chr = {chr_int: df_windows[df_windows['CHR_int'] == chr_int].copy() 
                          for chr_int in df_top_svs['CHR_int'].unique()}

    for _, sv_row in df_top_svs.iterrows():
        chr_int = sv_row['CHR_int']
        sv_start_safe = min(sv_row['START'], sv_row['END'])
        sv_end_safe = max(sv_row['START'], sv_row['END'])
        
        # 1. SV Loci: Windows fully contained within this specific SV
        df_chr_windows = df_windows_by_chr.get(chr_int)
        
        if df_chr_windows is None:
            print(f"Warning: No windows found for chromosome {chr_int}. Skipping {sv_row['SV_ID']}.")
            continue
            
        sv_loci = df_chr_windows[
            (df_chr_windows['POS'] >= sv_start_safe) &
            (df_chr_windows['POS'] + WINDOW_SIZE <= sv_end_safe) 
        ]
        
        # 2. Local Background: Windows on the same chromosome OUTSIDE the SV
        df_bg_local = get_windows_outside_sv(df_chr_windows, sv_start_safe, sv_end_safe, WINDOW_SIZE)

        # Skip if either SV or local background data is too small
        if len(sv_loci) < 5 or len(df_bg_local) < 5:
             print(f"Skipping {sv_row['SV_ID']} (Chr{chr_int}, {sv_row['TYPE']}) due to insufficient data (SV: {len(sv_loci)}, BG: {len(df_bg_local)} windows).")
             continue

        
        # 3. Perform Tests for all 4 metrics
        test_results = {
            'SV_ID': sv_row['SV_ID'],
            'TYPE': sv_row['TYPE'],
            'CHR': chr_int,
            'SIZE_kb': sv_row['SIZE'] / 1000,
            'SV_Windows': len(sv_loci),
            'BG_Windows': len(df_bg_local)
        }
        
        for stat_col, config in METRICS_TO_TEST.items():
            res = compare_distributions(sv_loci, df_bg_local, stat_col, config['alt']) 
            test_results[f'{stat_col}_Mean'] = res['mean_SV']
            test_results[f'{stat_col}_P'] = res['p_format']
            test_results[f'{stat_col}_Sig'] = res['conclusion']
            
        individual_test_results.append(test_results)

    df_individual_tests = pd.DataFrame(individual_test_results)

    # Print the new table
    print(df_individual_tests.to_string(index=False, float_format="%.4f"))
    print("\nNote on Mann-Whitney U Test Alternatives:")
    print(f"- FST: SV > Local Background (Alternative: 'greater')")
    print(f"- PI metrics and Ratio: SV $\\ne$ Local Background (Alternative: 'two-sided')")
    
    # --- ADDED SECTION 3B: Global SV vs Non-SV Comparison ---
    print("\n" + "="*80)
    print("--- 3B. Global Test: All Large SVs vs. All Non-SV Regions (>10 kbp SVs) ---")
    print("="*80)
    
    # 1. Identify all windows covered by ANY large SV
    all_sv_windows = []
    
    # Iterate through all large rearrangements (not just top K)
    for _, sv_row in df_rearrangements.iterrows():
        chr_int = sv_row['CHR_int']
        sv_start_safe = min(sv_row['START'], sv_row['END'])
        sv_end_safe = max(sv_row['START'], sv_row['END'])
        
        # Windows on this chromosome
        df_chr_windows = df_windows[df_windows['CHR_int'] == chr_int]
        
        # Windows fully contained within this SV
        sv_loci = df_chr_windows[
            (df_chr_windows['POS'] >= sv_start_safe) &
            (df_chr_windows['POS'] + WINDOW_SIZE <= sv_end_safe) 
        ]
        all_sv_windows.append(sv_loci)

    if not all_sv_windows:
        print("Warning: No windows found inside any large SV for global test. Skipping.")
        return

    df_all_sv = pd.concat(all_sv_windows).drop_duplicates(subset=['CHR_int', 'POS'])
    
    # 2. Identify all windows *outside* all large SVs (Non-SV Background)
    # Start with all windows, then remove the SV windows
    df_all_bg = df_windows[~df_windows.set_index(['CHR_int', 'POS']).index.isin(
        df_all_sv.set_index(['CHR_int', 'POS']).index
    )].copy()

    # Perform Tests for all 4 metrics (Global Test)
    global_test_results = {'Group': 'All Large SVs'}
    
    if len(df_all_sv) < 5 or len(df_all_bg) < 5:
        print("Warning: Insufficient window count for robust global test. SV Windows:", len(df_all_sv), "BG Windows:", len(df_all_bg))
        return

    print(f"Total SV Windows (>10 kbp SVs): {len(df_all_sv):,}")
    print(f"Total Non-SV Windows: {len(df_all_bg):,}")

    for stat_col, config in METRICS_TO_TEST.items():
        res = compare_distributions(df_all_sv, df_all_bg, stat_col, config['alt']) 
        global_test_results[f'{stat_col}_Mean_SV'] = res['mean_SV']
        global_test_results[f'{stat_col}_Mean_BG'] = res['mean_BG']
        global_test_results[f'{stat_col}_P'] = res['p_format']
        global_test_results[f'{stat_col}_Sig'] = res['conclusion']

    # Custom output table for global results
    print("\nGlobal Statistical Comparison (Mann-Whitney U Test):")
    
    header = ["Metric", "Mean SV", "Mean Non-SV", "P-value", "Significance"]
    data_rows = []
    
    for stat_col, config in METRICS_TO_TEST.items():
        data_rows.append([
            config['label'],
            f"{global_test_results[f'{stat_col}_Mean_SV']:.4f}",
            f"{global_test_results[f'{stat_col}_Mean_BG']:.4f}",
            global_test_results[f'{stat_col}_P'],
            global_test_results[f'{stat_col}_Sig']
        ])
    
    # Print the table structure
    col_widths = [max(len(h), max(len(row[i]) for row in data_rows)) for i, h in enumerate(header)]
    
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(header, col_widths)) + " |"
    separator_line = "|-" + "-|-".join('-' * w for w in col_widths) + "-|"
    
    print(header_line)
    print(separator_line)
    
    for row in data_rows:
        row_str = "| "
        row_str += row[0].ljust(col_widths[0]) + " | "
        row_str += row[1].ljust(col_widths[1]) + " | "
        row_str += row[2].ljust(col_widths[2]) + " | "
        row_str += row[3].ljust(col_widths[3]) + " | "
        row_str += row[4].ljust(col_widths[4]) + " |"
        print(row_str)
        
    print("\nNote: FST test alternative hypothesis is SV > Non-SV.")

    # --- SECTION 4: Density Plotting for Individual SVs (vs. Local Background) ---
    print("\n" + "="*80)
    print(f"--- Generating 4-Panel Density Plots in '{PLOTS_OUTPUT_DIR}/' ---")
    print("="*80)
    
    for _, sv_row in df_top_svs.iterrows():
        chr_int = sv_row['CHR_int']
        sv_start_safe = min(sv_row['START'], sv_row['END'])
        sv_end_safe = max(sv_row['START'], sv_row['END'])
        
        df_chr_windows = df_windows_by_chr.get(chr_int)
        if df_chr_windows is None: continue
            
        sv_loci = df_chr_windows[
            (df_chr_windows['POS'] >= sv_start_safe) &
            (df_chr_windows['POS'] + WINDOW_SIZE <= sv_end_safe) 
        ]
        df_bg_local = get_windows_outside_sv(df_chr_windows, sv_start_safe, sv_end_safe, WINDOW_SIZE)

        # Skip if insufficient windows for plotting
        if len(sv_loci) < 5 or len(df_bg_local) < 5:
            continue
            
        sv_name = f"{sv_row['SV_ID']} ({sv_row['TYPE']} on Chr{chr_int})"
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        # Calculate local 95th percentile for PI_RATIO axis cap
        local_q95_bg_ratio = df_bg_local['PI_RATIO'].dropna().quantile(0.95)
        # Ensure a reasonable minimum viewing limit
        local_standard_x_limit = max(local_q95_bg_ratio, 2.0)
        
        for i, (stat_col, config) in enumerate(METRICS_TO_TEST.items()):
            ax = axes[i]
            
            # Plot Local Background Distribution
            sns.kdeplot(df_bg_local[stat_col].dropna(), ax=ax, color='gray', fill=True, alpha=0.5, label='Local Background')
            # Plot SV Distribution
            sns.kdeplot(sv_loci[stat_col].dropna(), ax=ax, color=config['color'], fill=True, alpha=0.7, label=sv_name)
            
            # Add means as vertical lines
            ax.axvline(df_bg_local[stat_col].mean(), color='gray', linestyle='--', linewidth=1, label='BG Mean')
            ax.axvline(sv_loci[stat_col].mean(), color=config['color'], linestyle='-', linewidth=1, label='SV Mean')
            
            ax.set_title(f"{config['label']} Distribution: {sv_name}")
            ax.set_xlabel(config['label'])
            
            # Special handling for PI Ratio plot
            if stat_col == 'PI_RATIO':
                ax.axvline(1.0, color='black', linestyle=':', linewidth=1, label='Neutral Ratio (1.0)')
                ax.set_xlim(0, local_standard_x_limit)
                ax.set_title(f"{config['label']} Distribution (Cap at Local BG 95th: {local_standard_x_limit:.2f})")
            
            ax.legend(loc='upper right')

        plt.suptitle(f'Comparative Profile of {sv_name} ({sv_row["SIZE"]/1000:.1f} kb) vs. Local Chromosome Background', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        filename = os.path.join(PLOTS_OUTPUT_DIR, f"{sv_row['SV_ID']}_{sv_row['TYPE']}_Chr{chr_int}_local_analysis.png")
        plt.savefig(filename, dpi=300)
        plt.close(fig) 
        print(f"Generated {filename}")

    print("\n--- Analysis Complete ---")
    
if __name__ == "__main__":
    main()
