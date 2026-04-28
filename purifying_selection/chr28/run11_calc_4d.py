import os
import pandas as pd
from Bio import AlignIO
import sys
from scipy.stats import mannwhitneyu, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
INPUT_DIRS = ["paml_inputs_INSIDE", "paml_inputs_OUTSIDE"]

# --- Gene List Files (Output from run9_parse_paml.py) ---
# THESE FILES MUST EXIST AND CONTAIN THE LIST OF FILTERED GENES
GENE_LIST_INSIDE = "gene_ids_aggregate_inside.txt"
GENE_LIST_OUTSIDE = "gene_ids_aggregate_outside.txt"

OUTPUT_FILE = "p4d_filtered_results.csv" # Output file name reflecting filtering
PLOTS_PDF = "p4d_filtered_distribution.pdf" # Plot file name reflecting filtering
# ---------------------

# --- Universal Genetic Code Four-Fold Degenerate Codon Prefixes ---
FOUR_FOLD_CODONS = {
    "CC",  # Proline (Pro)
    "AC",  # Threonine (Thr)
    "GC",  # Alanine (Ala)
    "GG",  # Glycine (Gly)
    "GU",  # Valine (Val)
    "CU",  # Leucine (Leu)
    "UC",  # Serine (Ser)
    "CG"    # Arginine (Arg)
}
# ---------------------

def load_filtered_genes(file_path):
    """
    Loads the list of gene IDs that passed the strict PAML filter from a text file.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: Filter list file not found: {file_path}. Cannot guarantee consistent gene sets.")
        return set()
        
    with open(file_path, 'r') as f:
        # Read lines and strip whitespace/newlines
        gene_ids = {line.strip() for line in f if line.strip()}
    
    return gene_ids


def calculate_4d_metrics(alignment_path):
    """
    Reads a PHYLIP alignment (assumed to be 2 sequences) and calculates:
    1. Total substitutions at 4D sites (SNPs).
    2. Total comparable 4D sites.
    3. The ratio (p-distance at 4D sites, P4D).
    """
    try:
        # PAML uses PHYLIP Sequential format
        alignment = AlignIO.read(alignment_path, "phylip-sequential")
    except Exception:
        return {'Substitutions_4D': None, 'Total_4D_Sites': 0, 'P4D_Ratio': None}

    if len(alignment) != 2 or alignment.get_alignment_length() % 3 != 0:
        return {'Substitutions_4D': None, 'Total_4D_Sites': 0, 'P4D_Ratio': None}
    
    seq0 = str(alignment[0].seq).upper()
    seq1 = str(alignment[1].seq).upper()

    substitutions_4d = 0
    comparable_sites_4d = 0 # Sites used for P4D calculation (non-ambiguous in both seqs)

    # Iterate through the alignment by codons
    for i in range(0, alignment.get_alignment_length(), 3):
        codon0 = seq0[i:i+3]
        codon1 = seq1[i:i+3]

        # Skip codons with incomplete length or gaps in either sequence
        if len(codon0) < 3 or '-' in codon0 or '-' in codon1: 
            continue
        
        first_two_bases = codon0[0:2]
        third_base0 = codon0[2]
        third_base1 = codon1[2]
        
        # 1. Check for Four-Fold Degenerate Site based on Codon 0
        if first_two_bases in FOUR_FOLD_CODONS:
            
            # 2. P4D (Substitution Rate) calculation - Raw p-distance components
            # Must have unambiguous bases in BOTH sequences for comparison
            if third_base0 in 'ATGC' and third_base1 in 'ATGC':
                comparable_sites_4d += 1
                if third_base0 != third_base1:
                    substitutions_4d += 1

    
    p4d_ratio = substitutions_4d / comparable_sites_4d if comparable_sites_4d > 0 else None

    metrics = {
        'Substitutions_4D': substitutions_4d,
        'Total_4D_Sites': comparable_sites_4d,
        'P4D_Ratio': p4d_ratio
    }
    
    return metrics


def analyze_4d_metrics(df):
    """
    Performs statistical comparison (Mann-Whitney U) and generates a plot for the P4D ratio
    on the STRICTLY FILTERED dataset. Also calculates the aggregate P4D ratio and runs a Chi-squared test on pooled counts.
    """
    print("\n--- Starting 4D Metrics Statistical Analysis and Plotting (P4D, PAML Filtered) ---")
    
    # Metrics to analyze and their labels for printing
    metrics_to_print = {
        'Substitutions_4D': 'Total 4D Substitutions (SNPs)',
        'Total_4D_Sites': 'Total 4D Sites',
        'P4D_Ratio': r'P$_{4D}$ Ratio (Substitutions/Sites)'
    }
    
    # --- 1. Gene-by-Gene Statistical Analysis (MWU Test on ALL FILTERED genes) ---
    print("\n--- 1. Gene-by-Gene P4D Ratio Analysis (ALL Filtered Genes) ---")
    
    # Initialize chi2_p_value outside the loop for use in plotting function
    chi2_p_value = None 
    
    for metric, label in metrics_to_print.items():
        
        data_inside = df[df['Group'] == 'Inside Inversion'][metric].dropna()
        data_outside = df[df['Group'] == 'Outside Inversion'][metric].dropna()

        N_in = len(data_inside)
        N_out = len(data_outside)
        
        # Check if the data is numerical for stats
        if not pd.api.types.is_numeric_dtype(data_inside) or not pd.api.types.is_numeric_dtype(data_outside):
            print(f"\nMetric: {label} (Non-numeric data, skipping MWU)")
            continue
            
        print(f"\nMetric: {label}")
        print(f"  Inside Inversion (N={N_in}): Median = {data_inside.median():.6f}, Mean = {data_inside.mean():.6f}")
        print(f"  Outside Inversion (N={N_out}): Median = {data_outside.median():.6f}, Mean = {data_outside.mean():.6f}")

        if N_in < 2 or N_out < 2:
            print("  Not enough data points for statistical testing. Skipping MWU.")
        else:
            # Perform Mann-Whitney U test (non-parametric two-sided test)
            try:
                statistic, p_value = mannwhitneyu(data_inside, data_outside, alternative='two-sided')
                print(f"  Mann-Whitney U Test P-value: {p_value:.4f}")
            except ValueError:
                print("  Mann-Whitney U Test Error.")


    # --- 2. Gene-by-Gene Statistical Analysis (Excluding Genes with 0 Substitutions) ---
    print("\n--- 2. Gene-by-Gene P4D Ratio Analysis (Filtered, Excluding Genes with 0 Substitutions) ---")
    
    # Filter the DataFrame to include only genes with at least one 4D substitution
    df_nonzero = df[df['Substitutions_4D'] > 0].copy()
    
    metric = 'P4D_Ratio'
    label = r'P$_{4D}$ Ratio (Substitutions/Sites)'

    data_inside = df_nonzero[df_nonzero['Group'] == 'Inside Inversion'][metric].dropna()
    data_outside = df_nonzero[df_nonzero['Group'] == 'Outside Inversion'][metric].dropna()

    N_in = len(data_inside)
    N_out = len(data_outside)
    
    print(f"\nMetric: {label} (Non-Zero Subs Only)")
    print(f"  Inside Inversion (N={N_in}): Median = {data_inside.median():.6f}, Mean = {data_inside.mean():.6f}")
    print(f"  Outside Inversion (N={N_out}): Median = {data_outside.median():.6f}, Mean = {data_outside.mean():.6f}")

    if N_in < 2 or N_out < 2:
        print("  Not enough non-zero data points for statistical testing. Skipping MWU.")
    else:
        try:
            # Perform Mann-Whitney U test (non-parametric two-sided test)
            statistic, p_value = mannwhitneyu(data_inside, data_outside, alternative='two-sided')
            print(f"  Mann-Whitney U Test P-value: {p_value:.4f}")
        except ValueError:
            print("  Mann-Whitney U Test Error.")


    # --- 3. Aggregate P4D Ratio Calculation (Pooling all sites) ---
    print("\n\n--- 3. Aggregate P4D Analysis (Pooling all 4D Sites) ---")
    
    # Group by inversion location and sum the raw counts (Substitutions and Total Sites)
    aggregate_data = df.groupby('Group').agg(
        Total_Substitutions=('Substitutions_4D', 'sum'),
        Total_Sites=('Total_4D_Sites', 'sum')
    )

    # Calculate the aggregate P4D ratio (Pooled P4D)
    aggregate_data['Aggregate_P4D_Ratio'] = aggregate_data['Total_Substitutions'] / aggregate_data['Total_Sites']

    print("Aggregate P4D Results:")
    for group in aggregate_data.index:
        sites = aggregate_data.loc[group, 'Total_Sites']
        subs = aggregate_data.loc[group, 'Total_Substitutions']
        ratio = aggregate_data.loc[group, 'Aggregate_P4D_Ratio']
        print(f"  {group}:")
        print(f"    Total Substitutions (SNPs): {subs:,.0f}")
        print(f"    Total Comparable 4D Sites: {sites:,.0f}")
        print(f"    Aggregate P4D Ratio: {ratio:.6f}")

    # --- 4. Chi-squared Test on Pooled Counts (for increased power) ---
    print("\n--- 4. Chi-squared Test on Total Counts (Substitutions vs Non-Substitutions) ---")
    
    try:
        subs_in = aggregate_data.loc['Inside Inversion', 'Total_Substitutions']
        sites_in = aggregate_data.loc['Inside Inversion', 'Total_Sites']
        subs_out = aggregate_data.loc['Outside Inversion', 'Total_Substitutions']
        sites_out = aggregate_data.loc['Outside Inversion', 'Total_Sites']

        if sites_in > 0 and sites_out > 0:
            non_subs_in = sites_in - subs_in
            non_subs_out = sites_out - subs_out

            # Contingency table: [Substitutions, Non-Substitutions] for each group
            contingency_table = [
                [subs_in, subs_out],
                [non_subs_in, non_subs_out]
            ]
            
            # Perform Chi-squared test to compare the two proportions (P4D ratios)
            chi2, chi2_p_value, dof, expected = chi2_contingency(contingency_table)
            print(f"  Chi-squared Statistic: {chi2:.4f}")
            print(f"  Degrees of Freedom: {dof}")
            p_text = f"{chi2_p_value:.4e}" if chi2_p_value < 0.001 else f"{chi2_p_value:.4f}"
            print(f"  P-value: {p_text}")
        else:
            print("  Not enough aggregated site data to perform Chi-squared test.")
    except KeyError:
        print("  Error accessing aggregate data. Check group names.")
    except ValueError:
        print("  Chi-squared Test Error (Likely due to zero counts or insufficient degrees of freedom).")


    # --- Plotting P4D Ratio Distribution ---
    
    metric = 'P4D_Ratio'
    y_label = r'P$_{4D}$ Ratio (Substitutions/Sites)'
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    plot_df = df.copy()
    
    # P4D, like K4D, can be zero-inflated, so we use log transformation for better visualization
    min_p4d_val = df[metric][df[metric] > 0].min()
    log_offset = min_p4d_val / 10 if pd.notna(min_p4d_val) else 1e-6
    
    # Add N to group label for plot clarity
    N_in_total = len(df[df['Group'] == 'Inside Inversion'])
    N_out_total = len(df[df['Group'] == 'Outside Inversion'])
    plot_df.loc[plot_df['Group'] == 'Inside Inversion', 'Group_N'] = f'Inside Inversion (N={N_in_total})'
    plot_df.loc[plot_df['Group'] == 'Outside Inversion', 'Group_N'] = f'Outside Inversion (N={N_out_total})'
    
    plot_df[f'log_{metric}'] = np.log10(plot_df[metric].fillna(0) + log_offset)
    plot_metric = f'log_{metric}'
    plot_y_label = r'Log$_{10}$' + f'({y_label} + {log_offset:.2e})' 

    # Plot the data
    sns.violinplot(data=plot_df, x='Group_N', y=plot_metric, 
                   hue='Group_N', 
                   palette=['#3498db', '#e74c3c'], 
                   inner="quartile", 
                   linewidth=1.5,
                   legend=False, 
                   ax=ax)
    sns.despine(left=True, ax=ax)

    # Updated Title to reflect PAML Filtering
    ax.set_title(r'Distribution of Neutral Sequence Divergence ($\mathbf{P_{4D}}$) (PAML Filtered Genes)', fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel(plot_y_label, fontsize=12)
    
    # Add P-value annotation for P4D_Ratio (MWU for gene-by-gene, ALL filtered genes)
    data_inside_p4d = df[df['Group'] == 'Inside Inversion']['P4D_Ratio'].dropna()
    data_outside_p4d = df[df['Group'] == 'Outside Inversion']['P4D_Ratio'].dropna()
    
    if len(data_inside_p4d) >= 2 and len(data_outside_p4d) >= 2:
        try:
            statistic, p_value = mannwhitneyu(data_inside_p4d, data_outside_p4d, alternative='two-sided')
            p_text = f"MWU P (Filtered) = {p_value:.3e}" if p_value < 0.001 else f"MWU P (Filtered) = {p_value:.4f}"
            
            # Add Chi2 result if available (Note: chi2_p_value must be initialized/calculated above)
            # We access the most recently calculated chi2_p_value from the try/except block above
            if 'chi2_p_value' in locals() and chi2_p_value is not None:
                chi2_p_text = f"{chi2_p_value:.4e}" if chi2_p_value < 0.001 else f"{chi2_p_value:.4f}"
                p_text += f"\nChi2 P (Aggregate) = {chi2_p_text}"
                
            ax.text(0.5, 0.95, p_text, ha='center', fontsize=10, 
                     transform=ax.transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        except ValueError:
            pass # Skip if MWU fails
            

    plt.tight_layout()
    plt.savefig(PLOTS_PDF)
    print(f"\n✅ Analysis complete. Distribution plot saved to {PLOTS_PDF}")


def process_directories():
    """
    1. Loads the strictly filtered gene IDs from PAML text files.
    2. Iterates through input directories and aggregates 4D metrics results ONLY for those genes.
    """
    
    print("--- 4D Metrics Calculator Started (Applying PAML Filter) ---")
    
    # 1. Load the gene ID filters from files created by run9
    inside_filter_set = load_filtered_genes(GENE_LIST_INSIDE)
    outside_filter_set = load_filtered_genes(GENE_LIST_OUTSIDE)
    
    if not inside_filter_set and not outside_filter_set:
        print("FATAL ERROR: Could not load any genes from PAML filter lists. Run run9_parse_paml.py first.")
        return
    
    print(f"Loaded {len(inside_filter_set)} Inside genes and {len(outside_filter_set)} Outside genes from PAML filter lists.")
    
    all_results = []
    processed_genes = set()
    
    # Define mapping to assign group names and look up correct filter set
    GROUP_MAPPING = {
        "paml_inputs_INSIDE": ("Inside Inversion", inside_filter_set),
        "paml_inputs_OUTSIDE": ("Outside Inversion", outside_filter_set)
    }
    
    for input_dir in INPUT_DIRS:
        if not os.path.isdir(input_dir):
            print(f"❌ Directory not found: {input_dir}. Skipping.")
            continue
        
        group_name, filter_set = GROUP_MAPPING.get(input_dir, ("Unknown Group", set()))
        print(f"Processing directory: {input_dir} (Targeted Genes: {len(filter_set)})")
        
        for filename in os.listdir(input_dir):
            if filename.endswith(".phy"):
                gene_id = filename.replace(".phy", "")
                file_path = os.path.join(input_dir, filename)
                
                # --- CRITICAL FILTER STEP: Only process genes that passed the PAML filter ---
                if gene_id not in filter_set:
                    continue
                if gene_id in processed_genes: # Prevent processing duplicates if necessary
                    continue
                    
                metrics = calculate_4d_metrics(file_path)
                
                # Only include genes where 4D sites could be calculated
                if metrics and metrics['Total_4D_Sites'] > 0:
                    all_results.append({
                        'gene_id': gene_id,
                        'Group': group_name,
                        'Substitutions_4D': metrics['Substitutions_4D'],
                        'Total_4D_Sites': metrics['Total_4D_Sites'],
                        'P4D_Ratio': metrics['P4D_Ratio']
                    })
                    processed_genes.add(gene_id)


    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Successfully processed {len(all_results)} genes (PAML filtered).")
        print(f"Results saved to {OUTPUT_FILE}")
        
        # Analyze the data after successful saving, now including aggregate tests
        analyze_4d_metrics(df)
        
    else:
        print("\n❌ No valid PHYLIP files processed. Check input directories and PAML filter lists.")


if __name__ == "__main__":
    process_directories()
