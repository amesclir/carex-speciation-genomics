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
    using the Mann-Whitney U test.
    """
    sv_values = sv_data[stat_col].dropna()
    bg_values = bg_data[stat_col].dropna()
    
    # Ensure sufficient data for a robust test
    if len(sv_values) < 5 or len(bg_values) < 5:
        return {
            'mean_SV': np.nan, 
            'mean_BG': np.nan, 
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
    is_outside = (window_end <= sv_start) | (window_start >= sv_end)
    return df_chr[is_outside].copy()

def main():
    """
    Loads windowed data (Fst, Pi, Dxy), performs local background comparison for top SVs, 
    and generates 5-panel vertical density plots.
    """
    
    # --- USER-DEFINED FILE PATHS ---
    BASE_WGG_DIR = "/home/aescudero/wgg/" 
    FST_WINDOW_FILE = os.path.join(BASE_WGG_DIR, "fst_results/windowed_stats/windowed_fst_1000bp.windowed.weir.fst")
    PI_POP1_WINDOW_FILE = os.path.join(BASE_WGG_DIR, "fst_results/windowed_stats/windowed_pi_pop1_1000bp.windowed.pi")
    PI_POP2_WINDOW_FILE = os.path.join(BASE_WGG_DIR, "fst_results/windowed_stats/windowed_pi_pop2_1000bp.windowed.pi")
    DXY_WINDOW_FILE = os.path.join(BASE_WGG_DIR, "fst_results/windowed_stats/windowed_dxy_1000bp.txt")
    SYRI_FILE_PATH = "/home/aescudero/syri/syri.out" 

    # --- CONFIGURATION ---
    WINDOW_SIZE = 1000
    SV_SIZE_THRESHOLD = 10000
    K_TOP_SVS = 30
    PLOTS_OUTPUT_DIR = "sv_local_density_plots"

    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    
    # Define Metrics, Colors, and Labels
    METRICS_TO_TEST = {
        'FST':      {'label': 'FST', 'alt': 'greater', 'color': 'darkred'}, 
        'DXY':      {'label': '$D_{xy}$', 'alt': 'greater', 'color': 'red'}, 
        'PI_POP1':  {'label': r'$\pi_{\text{borbonica}}$', 'alt': 'two-sided', 'color': 'cyan'}, 
        'PI_POP2':  {'label': r'$\pi_{\text{boryana}}$', 'alt': 'two-sided', 'color': 'orange'}, 
        'PI_RATIO': {'label': r'$\pi_{\text{borb}}/\pi_{\text{bory}}$ Ratio', 'alt': 'two-sided', 'color': 'darkblue'}
    }

    # --- SECTION 1: Load and Process Windowed Statistics ---
    print("Loading and processing windowed statistics...")

    # 1a. Load Fst 
    if not os.path.exists(FST_WINDOW_FILE):
        print(f"Error: FST file not found at {FST_WINDOW_FILE}.")
        return
        
    df_fst = pd.read_csv(FST_WINDOW_FILE, sep='\t')
    df_fst.dropna(subset=['WEIGHTED_FST'], inplace=True) 
    df_fst = df_fst.rename(columns={'WEIGHTED_FST': 'FST', 'BIN_START': 'POS'})
    df_fst['CHR_int'] = df_fst['CHROM'].astype(str).str.replace('scaffold_', '', regex=False).astype(int)

    # 1b. Load Pi 
    if not os.path.exists(PI_POP1_WINDOW_FILE) or not os.path.exists(PI_POP2_WINDOW_FILE):
        print("Error: PI files not found.")
        return

    df_pi1 = pd.read_csv(PI_POP1_WINDOW_FILE, sep='\t')
    df_pi2 = pd.read_csv(PI_POP2_WINDOW_FILE, sep='\t')

    df_pi_merged = pd.merge(df_pi1[['CHROM', 'BIN_START', 'PI']], 
                            df_pi2[['CHROM', 'BIN_START', 'PI']], 
                            on=['CHROM', 'BIN_START'], 
                            suffixes=('_POP1', '_POP2'))

    # Calculate PI Ratio
    EPSILON = 1e-6
    df_pi_merged['PI_RATIO'] = df_pi_merged['PI_POP1'] / (df_pi_merged['PI_POP2'] + EPSILON)
    
    df_pi_merged = df_pi_merged.rename(columns={'BIN_START': 'POS', 'PI_POP1': 'PI_POP1', 'PI_POP2': 'PI_POP2'})
    df_pi_merged['CHR_int'] = df_pi_merged['CHROM'].astype(str).str.replace('scaffold_', '', regex=False).astype(int)

    # 1c. Load Dxy 
    if not os.path.exists(DXY_WINDOW_FILE):
        print(f"Error: Dxy file not found at {DXY_WINDOW_FILE}.")
        return

    df_dxy = pd.read_csv(DXY_WINDOW_FILE, sep='\t')
    df_dxy = df_dxy.rename(columns={'BIN_START': 'POS'}) 
    df_dxy['CHR_int'] = df_dxy['CHROM'].astype(str).str.replace('scaffold_', '', regex=False).astype(int)
    if 'DXY' not in df_dxy.columns:
        print("Error: 'DXY' column not found. Columns are:", df_dxy.columns)
        return
    df_dxy = df_dxy[['CHR_int', 'POS', 'DXY']] 

    # 1d. Combine All Data
    df_temp = pd.merge(df_fst[['CHR_int', 'POS', 'FST']], 
                       df_pi_merged[['CHR_int', 'POS', 'PI_RATIO', 'PI_POP1', 'PI_POP2']], 
                       on=['CHR_int', 'POS'], 
                       how='inner')
    
    df_windows = pd.merge(df_temp, 
                          df_dxy, 
                          on=['CHR_int', 'POS'], 
                          how='inner')

    print(f"Total {WINDOW_SIZE}bp windows with intersection of Fst, Pi, and Dxy: {len(df_windows)}")

    # --- SECTION 2: Load and Process SyRI Data ---
    print("\nLoading and processing SyRI data...")
    if not os.path.exists(SYRI_FILE_PATH):
        print(f"Error: SyRI file not found at {SYRI_FILE_PATH}.")
        return

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
        print("Warning: No large SVs found to analyze.")
        return

    print(f"Analyzing enrichment within the top {K_TOP_SVS} largest SVs:")

    # --- SECTION 3: Individual SV Statistical Analysis ---
    print("\n" + "="*80)
    print(f"--- Individual Statistical Tests (Mann-Whitney U) for Top {K_TOP_SVS} SVs ---")
    print("="*80)
    
    individual_test_results = []
    df_windows_by_chr = {chr_int: df_windows[df_windows['CHR_int'] == chr_int].copy() 
                          for chr_int in df_top_svs['CHR_int'].unique()}

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

        if len(sv_loci) < 5 or len(df_bg_local) < 5:
             print(f"Skipping {sv_row['SV_ID']} due to insufficient data.")
             continue

        test_results = {
            'SV_ID': sv_row['SV_ID'],
            'TYPE': sv_row['TYPE'],
            'CHR': chr_int,
            'SIZE_kb': sv_row['SIZE'] / 1000
        }
        
        for stat_col, config in METRICS_TO_TEST.items():
            res = compare_distributions(sv_loci, df_bg_local, stat_col, config['alt']) 
            test_results[f'{stat_col}_Mean'] = res['mean_SV']
            test_results[f'{stat_col}_P'] = res['p_format']
            test_results[f'{stat_col}_Sig'] = res['conclusion']
            
        individual_test_results.append(test_results)

    df_individual_tests = pd.DataFrame(individual_test_results)
    print(df_individual_tests.to_string(index=False, float_format="%.4f"))
    
    # --- SECTION 3B: Global SV vs Non-SV Comparison ---
    print("\n" + "="*80)
    print("--- 3B. Global Test: All Large SVs vs. All Non-SV Regions ---")
    print("="*80)
    
    all_sv_windows = []
    for _, sv_row in df_rearrangements.iterrows():
        chr_int = sv_row['CHR_int']
        sv_start_safe = min(sv_row['START'], sv_row['END'])
        sv_end_safe = max(sv_row['START'], sv_row['END'])
        
        df_chr_windows = df_windows[df_windows['CHR_int'] == chr_int]
        sv_loci = df_chr_windows[
            (df_chr_windows['POS'] >= sv_start_safe) &
            (df_chr_windows['POS'] + WINDOW_SIZE <= sv_end_safe) 
        ]
        all_sv_windows.append(sv_loci)

    if not all_sv_windows:
        print("Warning: No windows found inside any large SV for global test.")
        return

    df_all_sv = pd.concat(all_sv_windows).drop_duplicates(subset=['CHR_int', 'POS'])
    df_all_bg = df_windows[~df_windows.set_index(['CHR_int', 'POS']).index.isin(
        df_all_sv.set_index(['CHR_int', 'POS']).index
    )].copy()

    global_test_results = {'Group': 'All Large SVs'}
    
    print(f"Total SV Windows (>10 kbp SVs): {len(df_all_sv):,}")
    print(f"Total Non-SV Windows: {len(df_all_bg):,}")

    for stat_col, config in METRICS_TO_TEST.items():
        res = compare_distributions(df_all_sv, df_all_bg, stat_col, config['alt']) 
        global_test_results[f'{stat_col}_Mean_SV'] = res['mean_SV']
        global_test_results[f'{stat_col}_Mean_BG'] = res['mean_BG']
        global_test_results[f'{stat_col}_P'] = res['p_format']
        global_test_results[f'{stat_col}_Sig'] = res['conclusion']

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
    
    col_widths = [max(len(h), max(len(row[i]) for row in data_rows)) for i, h in enumerate(header)]
    print("| " + " | ".join(h.ljust(w) for h, w in zip(header, col_widths)) + " |")
    print("|-" + "-|-".join('-' * w for w in col_widths) + "-|")
    for row in data_rows:
        print("| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row))) + " |")

    # --- SECTION 4: Density Plotting ---
    print("\n" + "="*80)
    print(f"--- Generating 5-Panel Vertical Density Plots in '{PLOTS_OUTPUT_DIR}/' ---")
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

        if len(sv_loci) < 5 or len(df_bg_local) < 5: continue
            
        sv_name = f"{sv_row['SV_ID']} ({sv_row['TYPE']} on Chr{chr_int})"
        
        # CHANGED: Vertical layout (5 rows x 1 column)
        # Increased figure height to 20 to accommodate the vertical stack
        fig, axes = plt.subplots(5, 1, figsize=(10, 20))
        axes = axes.flatten()
        
        local_q95_bg_ratio = df_bg_local['PI_RATIO'].dropna().quantile(0.95)
        local_standard_x_limit = max(local_q95_bg_ratio, 2.0)
        
        for i, (stat_col, config) in enumerate(METRICS_TO_TEST.items()):
            ax = axes[i]
            
            # Plot Local Background Distribution (Darker Grey)
            sns.kdeplot(df_bg_local[stat_col].dropna(), ax=ax, color='#333333', fill=True, alpha=0.6, label='Local Background')
            
            # Plot SV Distribution
            sns.kdeplot(sv_loci[stat_col].dropna(), ax=ax, color=config['color'], fill=True, alpha=0.6, label=sv_name)
            
            # Add means
            ax.axvline(df_bg_local[stat_col].mean(), color='#333333', linestyle='--', linewidth=1, label='BG Mean')
            ax.axvline(sv_loci[stat_col].mean(), color=config['color'], linestyle='-', linewidth=1, label='SV Mean')
            
            ax.set_title(f"{config['label']}") 
            ax.set_xlabel(config['label'])
            
            if stat_col == 'PI_RATIO':
                ax.axvline(1.0, color='black', linestyle=':', linewidth=1)
                ax.set_xlim(0, local_standard_x_limit)
            
            ax.legend(loc='upper right', fontsize='small')

        plt.suptitle(f'Comparative Profile: {sv_name} ({sv_row["SIZE"]/1000:.1f} kb)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save as PNG
        filename_png = os.path.join(PLOTS_OUTPUT_DIR, f"{sv_row['SV_ID']}_{sv_row['TYPE']}_Chr{chr_int}_profile.png")
        plt.savefig(filename_png, dpi=300)
        
        # Save as PDF (New Request)
        filename_pdf = os.path.join(PLOTS_OUTPUT_DIR, f"{sv_row['SV_ID']}_{sv_row['TYPE']}_Chr{chr_int}_profile.pdf")
        plt.savefig(filename_pdf, format='pdf')
        
        plt.close(fig) 
        print(f"Generated {filename_png} and {filename_pdf}")

    print("\n--- Analysis Complete ---")
    
if __name__ == "__main__":
    main()
