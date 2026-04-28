#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy.stats import wilcoxon

def process_ld_file(ld_file, bin_size=1000, max_dist=100000):
    """Reads PLINK .ld file, calculates distance, and bins by distance."""
    if not os.path.exists(ld_file) or os.path.getsize(ld_file) == 0:
        return None
        
    try:
        # Read in chunks to save memory
        chunks = pd.read_csv(ld_file, sep=r'\s+', chunksize=500000, engine='c')
        binned_data = []
        
        for chunk in chunks:
            chunk['DIST'] = (chunk['BP_B'] - chunk['BP_A']).abs()
            chunk = chunk[chunk['DIST'] <= max_dist]
            chunk['BIN'] = (chunk['DIST'] // bin_size) * bin_size
            
            grouped = chunk.groupby('BIN').agg(
                r2_sum=('R2', 'sum'),
                count=('R2', 'count')
            ).reset_index()
            binned_data.append(grouped)
            
        if not binned_data:
            return None
            
        df_combined = pd.concat(binned_data)
        final_bins = df_combined.groupby('BIN').agg(
            total_r2=('r2_sum', 'sum'),
            total_count=('count', 'sum')
        ).reset_index()
        
        final_bins['MEAN_R2'] = final_bins['total_r2'] / final_bins['total_count']
        return final_bins.sort_values('BIN')
        
    except Exception as e:
        print(f"Error processing {ld_file}: {e}")
        return None

def main():
    LD_DIR = "/home/aescudero/wgg/fst_results/windowed_stats/ld_decay"
    BIN_SIZE = 1000     
    MAX_DIST = 100000   
    
    # Store stats for the supplementary table
    stats_results = []
    
    inside_files = glob.glob(os.path.join(LD_DIR, "*_inside.ld"))
    
    for inside_file in inside_files:
        sv_base = os.path.basename(inside_file).replace("_inside.ld", "")
        outside_file = os.path.join(LD_DIR, f"{sv_base}_outside.ld")
        
        print(f"Processing {sv_base}...")
        df_inside = process_ld_file(inside_file, BIN_SIZE, MAX_DIST)
        df_outside = process_ld_file(outside_file, BIN_SIZE, MAX_DIST)
        
        if df_inside is None or df_outside is None or len(df_inside) == 0 or len(df_outside) == 0:
            print(f"Skipping {sv_base} due to missing data.")
            continue
            
        # Merge bins to ensure we are comparing exact same distances
        merged = pd.merge(df_inside[['BIN', 'MEAN_R2']], df_outside[['BIN', 'MEAN_R2']], 
                          on='BIN', suffixes=('_in', '_out')).dropna()
        
        if len(merged) < 5:
            print(f"Not enough overlapping bins to test {sv_base}.")
            continue
            
        # Formal Statistical Test: Wilcoxon signed-rank test
        # Alternative='greater' tests if INSIDE > OUTSIDE
        stat, p_val = wilcoxon(merged['MEAN_R2_in'], merged['MEAN_R2_out'], alternative='greater')
        
        # Format p-value for plot
        p_str = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        
        # Save stats
        stats_results.append({
            'SV_ID': sv_base,
            'Bins_Compared': len(merged),
            'Mean_R2_Inside': merged['MEAN_R2_in'].mean(),
            'Mean_R2_Outside': merged['MEAN_R2_out'].mean(),
            'Wilcoxon_Statistic': stat,
            'P_Value': p_val
        })
            
        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(merged['BIN'] / 1000, merged['MEAN_R2_in'], label=f'Inside SV', color='darkred', linewidth=2)
        plt.plot(merged['BIN'] / 1000, merged['MEAN_R2_out'], label='Collinear Background', color='#555555', linewidth=2)
        
        plt.title(f'LD Decay Comparison: {sv_base}', fontsize=14)
        plt.xlabel('Physical Distance (kb)', fontsize=12)
        plt.ylabel('Average $r^2$', fontsize=12)
        
        # Add stats text to the plot
        plt.annotate(f"Wilcoxon Test (Inside > Outside):\n{p_str}", 
                     xy=(0.95, 0.95), xycoords='axes fraction', 
                     ha='right', va='top', fontsize=11, 
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.8))
        
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        out_plot = os.path.join(LD_DIR, f"{sv_base}_LD_Decay.png")
        plt.savefig(out_plot, dpi=300)
        plt.close()

    # Save summary table
    if stats_results:
        df_stats = pd.DataFrame(stats_results)
        df_stats = df_stats.sort_values('SV_ID')
        csv_out = os.path.join(LD_DIR, "LD_Decay_Statistical_Summary.csv")
        df_stats.to_csv(csv_out, index=False)
        print(f"\nSaved statistical summary to: {csv_out}")

if __name__ == "__main__":
    main()
