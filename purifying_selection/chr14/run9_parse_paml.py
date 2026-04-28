import os
import re
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration (Keep these the same) ---
INSIDE_DIR = "paml_inputs_INSIDE"
OUTSIDE_DIR = "paml_inputs_OUTSIDE"
OUTPUT_FILE_INSIDE = "results_omega_inside.csv"
OUTPUT_FILE_OUTSIDE = "results_omega_outside.csv"
PLOTS_PDF = "omega_distributions_extended.pdf"
# ---------------------------------------------

# --- Filtering Thresholds ---
# Genes with dS and dN outside these bounds are excluded from the main FILTERED analysis and plotting.
DS_MIN_THRESHOLD = 0.01
DS_MAX_THRESHOLD = 1.5
# ----------------------------

# --- Gene List Outputs (For use in run11_calc_4d_prueba.py) ---
# These files contain the gene IDs that passed the dS saturation filter (dS <= 1.5).
GENE_LIST_INSIDE_AGGREGATE = "gene_ids_aggregate_inside.txt"
GENE_LIST_OUTSIDE_AGGREGATE = "gene_ids_aggregate_outside.txt"

# These files contain the gene IDs that passed the STRICT dS/dN filter (0.01 <= d <= 1.5).
GENE_LIST_INSIDE_STRICT = "gene_ids_strict_inside.txt"
GENE_LIST_OUTSIDE_STRICT = "gene_ids_strict_outside.txt"
# ---------------------------------------------


def extract_paml_metrics(mlc_file_path):
    """
    Parses a PAML .mlc file to extract omega (dN/dS), dN, dS, and the total
    nonsynonymous (N) and synonymous (S) site counts for the whole tree (Model 0).
    
    Returns: A dictionary with 'omega', 'dN_tree', 'dS_tree', 'N_sites', and 'S_sites',
             or None if values not found.
    """
    metrics = {}
    try:
        with open(mlc_file_path, 'r') as f:
            content = f.read()

        # 1. Extract omega (dN/dS)
        omega_match = re.search(r"omega \(dN\/dS\) = \s*([\d\.]+e?[-+]?\d*)", content)
        if omega_match:
            try:
                metrics['omega'] = float(omega_match.group(1))
            except ValueError:
                metrics['omega'] = 999.0
        
        # 2. Extract dN and dS (tree lengths) - These represent the total divergence.
        dN_match = re.search(r"tree length for dN:\s*([\d\.]+e?[-+]?\d*)", content)
        dS_match = re.search(r"tree length for dS:\s*([\d\.]+e?[-+]?\d*)", content)
        
        if dN_match and dS_match:
            metrics['dN_tree'] = float(dN_match.group(1))
            metrics['dS_tree'] = float(dS_match.group(1))

        # 3. Extract N and S site counts from the dN & dS for each branch table.
        # This regex finds the line containing branch information and captures N and S values.
        table_block_match = re.search(r"dN & dS for each branch([\s\S]*?)Time used:", content)
        if table_block_match:
            table_block = table_block_match.group(1)
            # Find a line starting with a branch index (e.g., '3..1' or '3..2') and extract N and S
            # We look for two floats (\d\.) after a float representing branch length (t)
            branch_line_match = re.search(r"\d+\.\.\d+\s+[\d\.]+\s+([\d\.]+)\s+([\d\.]+)", table_block)
            
            if branch_line_match:
                metrics['N_sites'] = float(branch_line_match.group(1))
                metrics['S_sites'] = float(branch_line_match.group(2))
        
    except Exception as e:
        # print(f"Error reading {mlc_file_path}: {e}")
        return None

    if all(key in metrics for key in ['omega', 'dN_tree', 'dS_tree', 'N_sites', 'S_sites']):
        return metrics
    return None


def process_directory(input_dir, output_file):
    """Iterates through a directory, processes all .mlc files, and writes results to a CSV."""
    # print(f"Processing files in: {input_dir}")
    results = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".mlc"):
            gene_id = filename.replace(".mlc", "")
            file_path = os.path.join(input_dir, filename)
            
            metrics = extract_paml_metrics(file_path)
            
            if metrics:
                results.append({
                    'gene_id': gene_id,
                    'omega': metrics['omega'],
                    'dN': metrics['dN_tree'],
                    'dS': metrics['dS_tree'],
                    'N_sites': metrics['N_sites'],
                    'S_sites': metrics['S_sites']
                })

    if results:
        df_raw = pd.DataFrame(results)
        df_raw.to_csv(output_file, index=False)
        print(f"✅ Successfully wrote {len(results)} raw results to {output_file}")
        return df_raw
    else:
        print(f"❌ No valid results found in {input_dir}. Skipping CSV creation.")
        return pd.DataFrame()


def get_aggregate_stats_and_changes(df, group_name):
    """Calculates aggregate rates, N and S site totals, and total N and S changes."""
    df_copy = df.copy()
    
    # Calculate total nonsynonymous changes (Total DN) and total synonymous changes (Total DS)
    # Total changes = Rate * Sites (dN * N_sites or dS * S_sites)
    df_copy['N_changes'] = df_copy['dN'] * df_copy['N_sites']
    df_copy['S_changes'] = df_copy['dS'] * df_copy['S_sites']

    total_N_sites = df_copy['N_sites'].sum()
    total_S_sites = df_copy['S_sites'].sum()
    total_N_changes = df_copy['N_changes'].sum()
    total_S_changes = df_copy['S_changes'].sum()

    # Aggregate rates for the entire gene group
    aggregate_dN = total_N_changes / total_N_sites if total_N_sites else 0
    aggregate_dS = total_S_changes / total_S_sites if total_S_sites else 0
    
    # Calculate the final aggregate omega (dN/dS)
    aggregate_omega = aggregate_dN / aggregate_dS if aggregate_dS > 0 else 0

    return {
        'group': group_name, 
        'total_N_sites': total_N_sites, 
        'total_S_sites': total_S_sites, 
        'aggregate_dN': aggregate_dN, 
        'aggregate_dS': aggregate_dS, 
        'aggregate_omega': aggregate_omega,
        'total_N_changes': total_N_changes,
        'total_S_changes': total_S_changes,
    }


def calculate_unfiltered_aggregate_omega(df_inside_raw, df_outside_raw):
    """
    Calculates the aggregate omega after filtering out genes with saturated dS (> DS_MAX_THRESHOLD)
    and performs the Chi-Squared test for significance between the aggregate dN/dS ratios.
    *** Also exports the gene list used for this analysis. ***
    """
    # Filter: Remove genes where dS is saturated (> DS_MAX_THRESHOLD) to avoid bias in aggregate calculations.
    df_inside_ds_filtered = df_inside_raw[df_inside_raw['dS'] <= DS_MAX_THRESHOLD].copy()
    df_outside_ds_filtered = df_outside_raw[df_outside_raw['dS'] <= DS_MAX_THRESHOLD].copy()
    
    # --- EXPORT GENE LISTS ---
    df_inside_ds_filtered['gene_id'].to_csv(GENE_LIST_INSIDE_AGGREGATE, index=False, header=False)
    df_outside_ds_filtered['gene_id'].to_csv(GENE_LIST_OUTSIDE_AGGREGATE, index=False, header=False)
    
    print(f"\n✅ Exported {len(df_inside_ds_filtered)} Inside gene IDs to {GENE_LIST_INSIDE_AGGREGATE} (for P4D aggregate analysis).")
    print(f"✅ Exported {len(df_outside_ds_filtered)} Outside gene IDs to {GENE_LIST_OUTSIDE_AGGREGATE} (for P4D aggregate analysis).")
    # -------------------------

    # Use raw string to fix SyntaxWarning
    print(r"\n--- AGGREGATE Divergence Rate Analysis (Filtered $d_S \leq 1.5$ Only) ---")
    genes_removed_inside = len(df_inside_raw) - len(df_inside_ds_filtered)
    genes_removed_outside = len(df_outside_raw) - len(df_outside_ds_filtered)
    # Use f-string, ensuring no problematic escape sequences remain
    print(f"Filtering: Removed {genes_removed_inside} inside and {genes_removed_outside} outside genes due to $d_S > {DS_MAX_THRESHOLD}$.")
    print(f"Remaining genes: Inside={len(df_inside_ds_filtered)}, Outside={len(df_outside_ds_filtered)}.")

    inside_data = get_aggregate_stats_and_changes(df_inside_ds_filtered, "Inside Inversion")
    outside_data = get_aggregate_stats_and_changes(df_outside_ds_filtered, "Outside Inversion")
    
    # 1. Custom Print Table (NOW includes site totals)
    # Header uses unicode for omega
    header = ["Group", "Total LN Sites", "Total LS Sites", "Aggregate dN", "Aggregate dS", "Aggregate \u03c9 (dN/dS)"]
    data_list = [
        (inside_data['group'], inside_data['total_N_sites'], inside_data['total_S_sites'], inside_data['aggregate_dN'], inside_data['aggregate_dS'], inside_data['aggregate_omega']),
        (outside_data['group'], outside_data['total_N_sites'], outside_data['total_S_sites'], outside_data['aggregate_dN'], outside_data['aggregate_dS'], outside_data['aggregate_omega'])
    ]

    # Pre-calculate widths for alignment
    col_widths = [len(h) for h in header]
    
    for row in data_list:
        for i, item in enumerate(row):
            if i == 0:
                length = len(str(item))
            elif i < 3:
                length = len(f"{item:.0f}")
            else:
                length = len(f"{item:.4f}")
            col_widths[i] = max(col_widths[i], length)
    
    # Print Header
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(header, col_widths)) + " |"
    print(header_line)
    
    # Print Separator
    separator_line = "|-" + "-|-".join('-' * w for w in col_widths) + "-|"
    print(separator_line)
    
    # Print Data Rows
    for row in data_list:
        row_str = "| "
        row_str += row[0].ljust(col_widths[0]) + " | "
        row_str += f"{row[1]:.0f}".ljust(col_widths[1]) + " | "
        row_str += f"{row[2]:.0f}".ljust(col_widths[2]) + " | "
        row_str += f"{row[3]:.4f}".ljust(col_widths[3]) + " | "
        row_str += f"{row[4]:.4f}".ljust(col_widths[4]) + " | "
        row_str += f"{row[5]:.4f}".ljust(col_widths[5]) + " |"
        print(row_str)

    # 2. Ratio Calculation
    inside_omega = inside_data['aggregate_omega']
    outside_omega = outside_data['aggregate_omega']
    
    if outside_omega > 0:
        ratio_of_omegas = inside_omega / outside_omega
        # Use raw string for the title to prevent SyntaxWarning
        print(r"\n--- Ratio of FILTERED Aggregate Divergence Rates ($\omega_{\text{Inside}} / \omega_{\text{Outside}}$) ---")
        print(f"Ratio: {ratio_of_omegas:.4f}")
        # Use raw f-string to fix SyntaxWarning on L208
        print(rf"Interpretation: The aggregated divergence rate ($\omega$) is {ratio_of_omegas:.2f} times higher inside the inversion (using $d_S \leq 1.5$ genes).")
    else:
        # Use raw string to fix SyntaxWarning
        print(r"\nCannot calculate the ratio of aggregate omegas as the outside inversion aggregate $\omega$ is zero.")

    # 3. Significance Test (Chi-Squared Test on Total Changes)
    # Use raw string to fix SyntaxWarning on L214
    print(r"\n--- Significance Test for Aggregate $\omega$ Ratios (Chi-Squared Test on Total Substitutions) ---")
    
    # Contingency table of Total Substitutions
    # This tests if the ratio of N changes to S changes is significantly different between groups.
    # We round the changes to the nearest integer for Chi-Squared test, as it requires counts.
    # *** SYNTAX ERROR FIX HERE (removed 'cont') ***
    contingency_table = np.array([
        [int(round(inside_data['total_N_changes'])), int(round(inside_data['total_S_changes']))],
        [int(round(outside_data['total_N_changes'])), int(round(outside_data['total_S_changes']))]
    ])
    
    try:
        # *** SYNTAX ERROR FIX HERE (removed 'cont') ***
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    except ValueError:
        print("Error: Chi-squared test failed due to zero or negative values in the substitution table.")
        return

    print("Contingency Table (Total N changes vs Total S changes):")
    # *** SYNTAX ERROR FIX HERE (removed 'cont') ***
    print(contingency_table)
    print(f"  Chi-Squared Statistic: {chi2:.4f}")
    print(f"  P-value: {p_value:.5f}")
    # Use raw string to fix SyntaxWarning
    print(r"  Interpretation: {'Significant difference' if p_value < 0.05 else 'No significant difference'} in the aggregated $\omega$ ratio.")


# The remaining functions (get_summary_stats, proportion_analysis, calculate_aggregate_omega, and analyze_and_plot_results) 
# remain the same and operate on the FILTERED data to ensure consistency with the plotting step.

def get_summary_stats(data_series, group_name):
    """Calculates median and IQR for a given series and returns a formatted string."""
    median = data_series.median()
    q1 = data_series.quantile(0.25)
    q3 = data_series.quantile(0.75)
    return (
        f"  {group_name} (N={len(data_series)}): Median = {median:.4f}, "
        f"IQR = [{q1:.4f} - {q3:.4f}]"
    )

def proportion_analysis(df_inside, df_outside):
    """Performs chi-squared test on the proportion of genes with omega > 1."""
    
    inside_positive = (df_inside['omega'] > 1).sum()
    inside_other = len(df_inside) - inside_positive
    
    outside_positive = (df_outside['omega'] > 1).sum()
    outside_other = len(df_outside) - outside_positive

    contingency_table = np.array([
        [inside_positive, outside_positive],
        [inside_other, outside_other]
    ])
    
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    except ValueError:
        print("Error: Chi-squared test failed due to small sample size or zero cells.")
        return

    print("\n--- Proportion Analysis (Omega > 1 vs Omega <= 1) on Filtered Data ---")
    print(f"Inside Inversion ({len(df_inside)} genes):")
    print(f"  Genes with \u03c9 > 1 (Positive/Relaxed): {inside_positive} ({inside_positive/len(df_inside):.2%})")
    print(f"  Genes with \u03c9 \u2264 1 (Purifying/Neutral): {inside_other} ({inside_other/len(df_inside):.2%})")
    
    print(f"Outside Inversion ({len(df_outside)} genes):")
    print(f"  Genes with \u03c9 > 1 (Positive/Relaxed): {outside_positive} ({outside_positive/len(df_outside):.2%})")
    print(f"  Genes with \u03c9 \u2264 1 (Purifying/Neutral): {outside_other} ({outside_other/len(df_outside):.2%})")
    
    print("\nChi-Squared Test (Difference in Proportions):")
    print(f"  Chi-Squared Statistic: {chi2:.4f}")
    print(f"  P-value: {p_value:.5f}")
    print(f"  Interpretation: {'Significant difference' if p_value < 0.05 else 'No significant difference'} in the proportion of genes with \u03c9 > 1.")


def calculate_aggregate_omega_filtered(df_inside, df_outside):
    """
    Calculates the total aggregated omega (dN/dS) for the STRICTLY FILTERED gene group
    and prints the results.
    """
    # Use raw string to fix SyntaxWarning
    print(r"\n--- STRICTLY FILTERED Aggregate Divergence Rate Analysis (Total N, S, dN/dS) ---")

    inside_stats = get_aggregate_stats_and_changes(df_inside, "Inside Inversion")
    outside_stats = get_aggregate_stats_and_changes(df_outside, "Outside Inversion")

    # Displaying Site Totals and Rates for the filtered analysis
    header = ["Group", "Total LN Sites", "Total LS Sites", "Aggregate dN", "Aggregate dS", "Aggregate \u03c9 (dN/dS)"]
    data_list = [
        (inside_stats['group'], inside_stats['total_N_sites'], inside_stats['total_S_sites'], inside_stats['aggregate_dN'], inside_stats['aggregate_dS'], inside_stats['aggregate_omega']),
        (outside_stats['group'], outside_stats['total_N_sites'], outside_stats['total_S_sites'], outside_stats['aggregate_dN'], outside_stats['aggregate_dS'], outside_stats['aggregate_omega'])
    ]
    
    # Custom Print Table (using the original logic with site counts)
    col_widths = [len(h) for h in header]
    for row in data_list:
        for i, item in enumerate(row):
            if i == 0:
                length = len(str(item))
            elif i < 3:
                length = len(f"{item:.0f}")
            else:
                length = len(f"{item:.4f}")
            col_widths[i] = max(col_widths[i], length)
    
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(header, col_widths)) + " |"
    print(header_line)
    
    separator_line = "|-" + "-|-".join('-' * w for w in col_widths) + "-|"
    print(separator_line)
    
    for row in data_list:
        row_str = "| "
        row_str += row[0].ljust(col_widths[0]) + " | "
        row_str += f"{row[1]:.0f}".ljust(col_widths[1]) + " | "
        row_str += f"{row[2]:.0f}".ljust(col_widths[2]) + " | "
        row_str += f"{row[3]:.4f}".ljust(col_widths[3]) + " | "
        row_str += f"{row[4]:.4f}".ljust(col_widths[4]) + " | "
        row_str += f"{row[5]:.4f}".ljust(col_widths[5]) + " |"
        print(row_str)

    # Ratio calculation (only for display, no test needed here)
    inside_omega = inside_stats['aggregate_omega']
    outside_omega = outside_stats['aggregate_omega']
    
    if outside_omega > 0:
        ratio_of_omegas = inside_omega / outside_omega
        # Use raw string to fix SyntaxWarning
        print(r"$\omega_{\text{Inside}} / \omega_{\text{Outside}}$ Ratio (Strictly Filtered Data):")
        print(f"Ratio: {ratio_of_omegas:.4f}")


def analyze_and_plot_results(df_inside_raw, df_outside_raw, plot_file):
    """
    Combines data, runs dS max filtered aggregate test, applies strict dS/dN filtering, 
    runs statistical tests on strictly filtered data, and generates a multi-panel plot.
    """
    
    # --- STEP 1: dS Max Filtered Aggregate Analysis (UPDATED TO EXPORT GENE LIST) ---
    calculate_unfiltered_aggregate_omega(df_inside_raw, df_outside_raw)


    # --- STEP 2: Filtering based on dS and dN values for PLOTTING and individual tests ---
    
    # Combine dataframes for unified filtering
    df_inside_raw['Group'] = 'Inside Inversion'
    df_outside_raw['Group'] = 'Outside Inversion'
    df = pd.concat([df_inside_raw, df_outside_raw], ignore_index=True)

    initial_count = len(df)

    # Filter: dS and dN must be within the specified bounds
    df_filtered = df[
        (df['dS'] >= DS_MIN_THRESHOLD) & (df['dS'] <= DS_MAX_THRESHOLD) &
        (df['dN'] >= DS_MIN_THRESHOLD) & (df['dN'] <= DS_MAX_THRESHOLD)
    ].copy()
    
    df_inside_filtered = df_filtered[df_filtered['Group'] == 'Inside Inversion'].copy()
    df_outside_filtered = df_filtered[df_filtered['Group'] == 'Outside Inversion'].copy()

    N_inside = len(df_inside_filtered)
    N_outside = len(df_outside_filtered)
    
    print("\n--- Strictly Filtered Data Summary (Used for subsequent tests and plotting) ---")
    print(f"Initial Total Gene Count: {initial_count}")
    # Use f-string with unicode for \leq
    print(f"Filter used: {DS_MIN_THRESHOLD} \u2264 dS \u2264 {DS_MAX_THRESHOLD} AND {DS_MIN_THRESHOLD} \u2264 dN \u2264 {DS_MAX_THRESHOLD}")
    print(f"Final Filtered Genes Inside: {N_inside}")
    print(f"Final Filtered Genes Outside: {N_outside}")
    
    if N_inside == 0 or N_outside == 0:
        print("Error: One or both gene groups are empty after filtering. Analysis aborted.")
        return

    # --- EXPORT STRICT GENE LISTS (Original purpose) ---
    df_inside_filtered['gene_id'].to_csv(GENE_LIST_INSIDE_STRICT, index=False, header=False)
    df_outside_filtered['gene_id'].to_csv(GENE_LIST_OUTSIDE_STRICT, index=False, header=False)
    print(f"✅ Exported {N_inside} Inside gene IDs to {GENE_LIST_INSIDE_STRICT} (for previous P4D analysis).")
    print(f"✅ Exported {N_outside} Outside gene IDs to {GENE_LIST_OUTSIDE_STRICT} (for previous P4D analysis).")
    # ----------------------------------------------------
    
    # Update the 'Group' labels to include the final sample size (N) for plotting
    df_filtered.loc[df_filtered['Group'] == 'Inside Inversion', 'Group'] = f'Inside Inversion (N={N_inside})'
    df_filtered.loc[df_filtered['Group'] == 'Outside Inversion', 'Group'] = f'Outside Inversion (N={N_outside})'


    # --- STEP 3: Strictly Filtered Data Analysis ---
    
    # 3a. Descriptive Stats (Original Analysis)
    # Use raw strings for the dictionary keys
    metrics = {
        'omega': r'$\omega$ (dN/dS)',
        'dN': r'$d_N$ (Non-synonymous Rate)',
        'dS': r'$d_S$ (Synonymous Rate)'
    }
    
    print("\n--- Descriptive Statistics (Median, IQR) on Strictly Filtered Data ---")
    
    for col, title in metrics.items():
        print(f"Metric: {col}")
        data_inside = df_inside_filtered[col].dropna()
        data_outside = df_outside_filtered[col].dropna()
        
        print(get_summary_stats(data_inside, "Inside Inversion"))
        print(get_summary_stats(data_outside, "Outside Inversion"))
    
    # 3b. Aggregate Site Count and Omega Calculation (Strictly Filtered Data)
    calculate_aggregate_omega_filtered(df_inside_filtered, df_outside_filtered)
    
    # 3c. Mann-Whitney U Test (Original Analysis)
    print("\n--- Mann-Whitney U Test Results on Strictly Filtered Data (Two-Sided) ---")
    
    # 3d. Proportion Analysis (Original Analysis)
    proportion_analysis(df_inside_filtered, df_outside_filtered)
    
    
    # --- STEP 4: Plotting (Extended to 4 panels) ---
    
    # Set up the plot 
    sns.set_theme(style="whitegrid", rc={"figure.figsize": (20, 5)}) 
    fig, axes = plt.subplots(1, 4)
    axes_flat = axes.flatten() 
    
    # Use raw string for the main title to fix SyntaxWarning
    fig.suptitle(r'Extended Evolutionary Analysis (Strictly Filtered by $0.01 \leq d \leq 1.5$ for both $d_S$ and $d_N$)', fontsize=16)

    palette = sns.color_palette()
    
    # Index 0, 1, 2: KDE plots for omega, dN, dS (Original Analysis)
    for i, (col, title) in enumerate(metrics.items()):
        
        data_inside = df_inside_filtered[col].dropna()
        data_outside = df_outside_filtered[col].dropna()

        # Run MWU test again to print p-values in the plot section
        stat_text = "Test Skipped (Low N)"
        try:
            # 1. Two-sided test (General difference)
            statistic_two_sided, p_value_two_sided = mannwhitneyu(data_inside, data_outside, alternative='two-sided', method='auto')
            
            # Print the one-sided results only for omega
            if col == 'omega':
                statistic_less, p_value_less = mannwhitneyu(data_inside, data_outside, alternative='less', method='auto')
                statistic_greater, p_value_greater = mannwhitneyu(data_inside, data_outside, alternative='greater', method='auto')
                print(f"Metric: {col} | Two-Sided P-value: {p_value_two_sided:.5f}")
                print(f"Metric: {col} | One-Sided P-value (Inside < Outside - Stronger Purifying): {p_value_less:.5f}")
                print(f"Metric: {col} | One-Sided P-value (Inside > Outside - Stronger Positive/Relaxed): {p_value_greater:.5f}")
            else:
                print(f"Metric: {col} | Two-Sided P-value: {p_value_two_sided:.5f}")

            # Determine the text for the plot based on the two-sided P-value
            if p_value_two_sided < 0.001:
                stat_text = f"MWU P < 0.001***"
            elif p_value_two_sided < 0.01:
                stat_text = f"MWU P = {p_value_two_sided:.3f}**"
            elif p_value_two_sided < 0.05:
                stat_text = f"MWU P = {p_value_two_sided:.3f}*"
            else:
                stat_text = f"MWU P = {p_value_two_sided:.3f}"
            
        except ValueError:
            pass # stat_text remains "Test Skipped"


        # Plotting logic remains the same for the first three plots
        if col == 'omega':
            x_min_plot = max(df_filtered['omega'].min(), 0.0001)  
            xlim_max = 5.0 
            
            df_plot = df_filtered[df_filtered['omega'] <= xlim_max]
            
            sns.kdeplot(
                data=df_plot, 
                x=col, 
                hue="Group", 
                fill=True, 
                alpha=.3, 
                ax=axes_flat[i], 
                common_norm=False, 
                linewidth=2,
                log_scale=True,
                cut=0
            )
            axes_flat[i].set_xscale('log')  
            axes_flat[i].set_xlim(x_min_plot, xlim_max)
            axes_flat[i].axvline(1.0, color='grey', linestyle=':', alpha=0.7, linewidth=1.5, zorder=0)

        else:
            df_plot = df_filtered
            sns.kdeplot(
                data=df_plot, 
                x=col, 
                hue="Group", 
                fill=True, 
                alpha=.3, 
                ax=axes_flat[i], 
                common_norm=False, 
                linewidth=2,
                clip=(DS_MIN_THRESHOLD, DS_MAX_THRESHOLD * 1.5)
            )
            axes_flat[i].set_xlim(DS_MIN_THRESHOLD - 0.01, DS_MAX_THRESHOLD)
            axes_flat[i].axvline(df_plot[col].median(), color='grey', linestyle=':', alpha=0.7, linewidth=1.5, zorder=0)

        # Add p-value text box
        axes_flat[i].text(0.95, 0.95, stat_text, 
                              transform=axes_flat[i].transAxes, 
                              fontsize=10, 
                              verticalalignment='top', 
                              horizontalalignment='right', 
                              bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        axes_flat[i].set_title(title)
        axes_flat[i].set_xlabel(title)
        
        # Add Median lines
        median_in = df_inside_filtered[col].median()
        median_out = df_outside_filtered[col].median()
        
        axes_flat[i].axvline(median_in, color=palette[0], linestyle='--', alpha=0.8, linewidth=1.5, label=f'Inside Median')
        axes_flat[i].axvline(median_out, color=palette[1], linestyle='--', alpha=0.8, linewidth=1.5, label=f'Outside Median')

        if axes_flat[i].legend_:
            axes_flat[i].legend_.remove()  
        
        sns.despine(ax=axes_flat[i], left=True)

    
    # --- NEW PLOT: dN vs dS Scatter Plot (axes_flat[3]) ---
    ax_scatter = axes_flat[3]
    sns.scatterplot(
        data=df_filtered,
        x='dS',
        y='dN',
        hue='Group',
        palette=palette[:2],
        alpha=0.6,
        s=50,
        ax=ax_scatter
    )

    # Add the line for neutral evolution (omega = 1, so dN = dS)
    max_val = max(df_filtered['dS'].max(), df_filtered['dN'].max()) * 1.05
    min_val = min(df_filtered['dS'].min(), df_filtered['dN'].min()) * 0.95
    
    # Use raw string for LaTeX labels
    ax_scatter.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label=r'$\omega = 1$ (Neutral)')

    # Add the line for omega = 0.5 (dN = 0.5 * dS)
    ax_scatter.plot([0, max_val], [0, max_val * 0.5], 'k:', alpha=0.5, label=r'$\omega = 0.5$')
    
    # Add the line for omega = 2 (dN = 2 * dS)
    ax_scatter.plot([0, max_val * 0.5], [0, max_val], 'k:', alpha=0.5, label=r'$\omega = 2$')

    ax_scatter.set_title(r'$d_N$ vs $d_S$ Scatter Plot')
    ax_scatter.set_xlabel(r'$d_S$ (Synonymous Rate)')
    ax_scatter.set_ylabel(r'$d_N$ (Non-synonymous Rate)')
    ax_scatter.set_xlim(0, max_val)
    ax_scatter.set_ylim(0, max_val)
    ax_scatter.legend_.remove()
    sns.despine(ax=ax_scatter, left=False, bottom=False)

    # Get the legend handles/labels from the scatter plot (including the omega lines)
    handles, labels = ax_scatter.get_legend_handles_labels()
    
    group_labels = [label for label in labels if 'Inversion' in label or r'$\omega' in label]
    group_handles = [handle for handle, label in zip(handles, labels) if 'Inversion' in label or r'$\omega' in label]

    # Add a single overall legend at the bottom center
    # Use raw string for the title to fix SyntaxWarning
    fig.legend(group_handles, group_labels, loc='lower center', ncol=5, 
               title=r"Gene Group (Filtered Sample Size) and $\omega$ Lines", frameon=True, bbox_to_anchor=(0.5, -0.05))
    
    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(plot_file)
    print(f"\n✅ Analysis complete. Extended distribution plot saved to {plot_file}")


if __name__ == "__main__":
    
    print("--- PAML Result Aggregator Started ---")
    
    # 1. Aggregate the data (returns the raw dataframes)
    df_inside_raw = process_directory(INSIDE_DIR, OUTPUT_FILE_INSIDE)
    df_outside_raw = process_directory(OUTSIDE_DIR, OUTPUT_FILE_OUTSIDE)
    
    print(f"\nRaw total genes: Inside={len(df_inside_raw)}, Outside={len(df_outside_raw)}")
    
    if len(df_inside_raw) > 0 and len(df_outside_raw) > 0:
        # 2. Run analysis, including dS max filtered aggregate test and STRICTLY FILTERED tests/plotting
        analyze_and_plot_results(df_inside_raw, df_outside_raw, PLOTS_PDF)
    else:
        print("Cannot proceed with analysis: One or both raw data files are empty.")
