#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import os

def load_data_and_svs():
    """
    Loads all required windowed statistics and the top SV data.
    """
    # --- USER-DEFINED FILE PATHS ---
    BASE_WGG_DIR = "/home/aescudero/wgg/" 
    FST_WINDOW_FILE = os.path.join(BASE_WGG_DIR, "fst_results/windowed_stats/windowed_fst_1000bp.windowed.weir.fst")
    PI_POP1_WINDOW_FILE = os.path.join(BASE_WGG_DIR, "fst_results/windowed_stats/windowed_pi_pop1_1000bp.windowed.pi")
    PI_POP2_WINDOW_FILE = os.path.join(BASE_WGG_DIR, "fst_results/windowed_stats/windowed_pi_pop2_1000bp.windowed.pi")
    SYRI_FILE_PATH = "/home/aescudero/syri/syri.out" 
    WINDOW_SIZE = 1000
    SV_SIZE_THRESHOLD = 10000

    print("Loading and merging windowed statistics...")

    # 1. Load Fst windows
    df_fst = pd.read_csv(FST_WINDOW_FILE, sep='\t').rename(columns={'WEIGHTED_FST': 'FST', 'BIN_START': 'POS'})
    df_fst['CHR_int'] = df_fst['CHROM'].astype(str).str.replace('scaffold_', '', regex=False).astype(int)

    # 2. Load Pi windows and calculate Ratio
    df_pi1 = pd.read_csv(PI_POP1_WINDOW_FILE, sep='\t')
    df_pi2 = pd.read_csv(PI_POP2_WINDOW_FILE, sep='\t')

    df_pi_merged = pd.merge(df_pi1[['CHROM', 'BIN_START', 'PI']], 
                            df_pi2[['CHROM', 'BIN_START', 'PI']], 
                            on=['CHROM', 'BIN_START'], 
                            suffixes=('_POP1', '_POP2'))

    EPSILON = 1e-6 # Avoid division by zero
    df_pi_merged['PI_RATIO'] = df_pi_merged['PI_POP1'] / (df_pi_merged['PI_POP2'] + EPSILON)
    df_pi_merged = df_pi_merged.rename(columns={'BIN_START': 'POS'})
    df_pi_merged['CHR_int'] = df_pi_merged['CHROM'].astype(str).str.replace('scaffold_', '', regex=False).astype(int)

    # 3. Combine Fst and Pi data
    df_windows = pd.merge(df_fst[['CHR_int', 'POS', 'FST']], 
                          df_pi_merged[['CHR_int', 'POS', 'PI_RATIO', 'PI_POP1', 'PI_POP2']], 
                          on=['CHR_int', 'POS'], 
                          how='inner')
    df_windows.dropna(subset=['FST', 'PI_RATIO'], inplace=True)

    print(f"Total windows loaded: {len(df_windows)}")

    # 4. Load SyRI data and identify top SVs
    df_raw_syri = pd.read_csv(SYRI_FILE_PATH, sep='\s+', header=None, dtype={0: str, 5: str}, on_bad_lines='skip', low_memory=False)
    df_raw_syri = df_raw_syri.rename(columns={10: 'TYPE'})

    rearrangement_types = ['INV', 'DUP', 'TRANS']
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

    df_top_svs = df_rearrangements.sort_values('SIZE', ascending=False).head(10).copy().reset_index(drop=True)
    df_top_svs['SV_ID'] = [f"SV{i+1}" for i in df_top_svs.index]

    return df_windows, df_top_svs

def plot_chromosome_data(df_chr, sv_info, chr_int, output_dir):
    """
    Generates a 4-panel plot for a single chromosome, highlighting the SV location.
    
    Args:
        df_chr (pd.DataFrame): Windowed data for the specific chromosome.
        sv_info (pd.Series): Row from df_top_svs corresponding to the SV on this chromosome.
        chr_int (int): Chromosome identifier.
        output_dir (str): Directory to save the plot.
    """
    
    SV_ID = sv_info['SV_ID']
    SV_TYPE = sv_info['TYPE']
    SV_START = min(sv_info['START'], sv_info['END'])
    SV_END = max(sv_info['START'], sv_info['END'])
    
    # Define metrics and their properties for plotting
    METRICS = [
        {'col': 'FST', 'label': 'FST (Divergence)', 'color': 'darkred', 'ylim_max': 1.0},
        {'col': 'PI_POP1', 'label': '$\\pi$ Pop1 (Diversity)', 'color': 'darkgreen', 'ylim_max_q': 0.99},
        {'col': 'PI_POP2', 'label': '$\\pi$ Pop2 (Diversity)', 'color': 'orange', 'ylim_max_q': 0.99},
        {'col': 'PI_RATIO', 'label': '$\\pi_{\\text{Pop1}}/\\pi_{\\text{Pop2}}$ Ratio', 'color': 'darkblue', 'hline': 1.0, 'ylim_max_q': 0.99}
    ]

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(16, 10), sharex=True)
    
    # Calculate global max Y-limits for Pi plots based on 99th percentile for better visibility
    pi_max = df_chr[['PI_POP1', 'PI_POP2']].stack().quantile(0.99)
    ratio_max = df_chr['PI_RATIO'].quantile(0.99)

    for i, metric in enumerate(METRICS):
        ax = axes[i]
        col = metric['col']
        
        # Scatter plot of the statistic
        ax.scatter(df_chr['POS'], df_chr[col], s=5, alpha=0.7, color=metric['color'])
        
        # Highlight the SV region
        ax.axvspan(SV_START, SV_END, color='gray', alpha=0.2, label=f'{SV_ID} ({SV_TYPE})')
        
        # Horizontal line for PI Ratio (neutral expectation)
        if 'hline' in metric:
            ax.axhline(metric['hline'], color='black', linestyle=':', linewidth=1)
        
        # Set Y-axis limits
        if col == 'FST':
            ax.set_ylim(0, metric['ylim_max'])
        elif col in ['PI_POP1', 'PI_POP2']:
            ax.set_ylim(0, pi_max)
        elif col == 'PI_RATIO':
             ax.set_ylim(0, ratio_max)

        ax.set_ylabel(metric['label'], fontsize=12)
        ax.legend(loc='upper right', fontsize=10)

    # Set X-axis label only for the bottom plot
    axes[-1].set_xlabel(f'Position (bp) on Chromosome {chr_int}', fontsize=14)
    
    # Title for the entire figure
    fig.suptitle(f'Window-by-Window Divergence and Diversity on Chr{chr_int} (Highlighting {SV_ID}: {SV_TYPE})', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save the plot
    filename = os.path.join(output_dir, f"{SV_ID}_{SV_TYPE}_Chr{chr_int}_window_plot.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Generated {filename}")


def main():
    """Main function to run the visualization pipeline."""
    
    # --- Configuration ---
    PLOTS_OUTPUT_DIR = "sv_chromosome_plots"
    # Target chromosomes based on user's important SVs (SV1, SV3, SV4, SV8)
    TARGET_CHRS = {
        14: 'SV1',
        20: 'SV3',
        3: 'SV4',
        32: 'SV8'
    }

    # Create output directory
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df_windows, df_top_svs = load_data_and_svs()
    
    print("\nStarting chromosome-wide plotting...")
    
    for chr_int, sv_id_target in TARGET_CHRS.items():
        # 1. Get SV metadata
        sv_info = df_top_svs[df_top_svs['SV_ID'] == sv_id_target].iloc[0]
        
        # 2. Filter windowed data for the target chromosome
        df_chr = df_windows[df_windows['CHR_int'] == chr_int].copy()
        
        if df_chr.empty:
            print(f"Warning: No window data found for Chr{chr_int}. Skipping.")
            continue
            
        # 3. Generate and save the plot
        plot_chromosome_data(df_chr, sv_info, chr_int, PLOTS_OUTPUT_DIR)

    print("\n--- Visualization Complete ---")

if __name__ == "__main__":
    main()
