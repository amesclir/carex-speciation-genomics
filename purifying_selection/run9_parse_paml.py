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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          45.log
  [SUCCESS] g16545.ctl completed.
Processing g16546.ctl -> Output logged to g16546.log
  [SUCCESS] g16546.ctl completed.
Processing g16547.ctl -> Output logged to g16547.log
  [SUCCESS] g16547.ctl completed.
--- PAML Batch Run Complete ---
]0;aescudero@login3:~/purifying_selection/paml_inputs_OUTSIDE(mafft) [aescudero@nodo4143 paml_inputs_OUTSIDE]$ ./run8_paml.sh [Kcd ..
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ python run9_parse_paml.py 
  File [35m"/lustre/home/aescudero/purifying_selection/run9_parse_paml.py"[0m, line [35m1[0m
    Script started on 2025-11-25 12:[1;31m0[0m3:58+01:00
                                    [1;31m^[0m
[1;35mSyntaxError[0m: [35mleading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers[0m
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ python run9_parse_paml.py 
  File [35m"/lustre/home/aescudero/purifying_selection/run9_parse_paml.py"[0m, line [35m1[0m
    Script started on 2025-11-25 12:[1;31m0[0m3:58+01:00
                                    [1;31m^[0m
[1;35mSyntaxError[0m: [35mleading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers[0m
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ python run9_parse_paml.py [1P[1P[1P[1P[1P[1P[1@n[1@a[1@n[1@o
[?1049h[22;0;0t[1;45r(B[m[4l[?7h[39;49m[?1h=[?1h=[?1h=[?25l[39;49m(B[m[H[2J[43;85H(B[0;7m[ Reading File ](B[m[43;83H(B[0;7m[ Read 2010 lines ](B[m[H(B[0;7m  GNU nano 2.9.8                                                                      run9_parse_paml.py                                                                                [1;183H(B[m[44d(B[0;7m^G(B[m Get Help     (B[0;7m^O(B[m Write Out    (B[0;7m^W(B[m Where Is     (B[0;7m^K(B[m Cut Text     (B[0;7m^J(B[m Justify	(B[0;7m^C(B[m Cur Pos	(B[0;7mM-U(B[m Undo[44;113H(B[0;7mM-A(B[m Mark Text   (B[0;7mM-](B[m To Bracket  (B[0;7mM-▲(B[m Previous    (B[0;7m^B(B[m Back[45d(B[0;7m^X(B[m Exit[45;17H(B[0;7m^R(B[m Read File    (B[0;7m^\(B[m Replace	(B[0;7m^U(B[m Uncut Text   (B[0;7m^T(B[m To Linter    (B[0;7m^_(B[m Go To Line   (B[0;7mM-E(B[m Redo[45;113H(B[0;7mM-6(B[m Copy Text   (B[0;7mM-W(B[m WhereIs Next(B[0;7mM-▼(B[m Next[45;161H(B[0;7m^F(B[m Forward[43d[3d[39;49m(B[mScript started on 2025-11-25 12:03:58+01:00[4d^X^Z^Z^[]0;aescudero@login3:~/purifying_selection^G(base) [aescudero@nodo4143 purifying_selection]$ ^C[5d^[]0;aescudero@login3:~/purifying_selection^G(base) [aescudero@nodo4143 purifying_selection]$ ^[[A^H^H^H^Hsalloc --mem=16G -c 4 -t 02:00:00 srun --pty /bin/bash -i^M^[[C^[[C^[[C^[[C^[$[6;14HJOBID PARTITION     NAME     USER ST	TIME  NODES NODELIST(REASON)[7;12H1050602  standard interact aescuder  R    1:52:18	  1 nodo4143[8d^[]0;aescudero@login3:~/purifying_selection^G(base) [aescudero@nodo4143 purifying_selection]$ conda activate mafft[9d^[]0;aescudero@login3:~/purifying_selection^G(mafft) [aescudero@nodo4143 purifying_selection]$ conda activate mafft^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^Hsqueue^[[K^H^H^H^H^Halloc --$[10;1H^[]0;aescudero@login3:~/purifying_selection^G(mafft) [aescudero@nodo4143 purifying_selection]$ nano run9_parse_paml.py[42m [11d[49m(B[m^[[?1049h^[[22;0;0t^[[1;45r^[(B^[[m^[[4l^[[?7h^[[39;49m^[[?1h^[=^[[?1h^[=^[[?1h^[=^[[?25l^[[39;49m^[(B^[[m^[[H^[[2J^[[43;85H^[(B^[[0;7m[ Reading File ]^[(B^[[m^[[43;91H^[(B^[[0;7m 0 l$[12;1H^[]0;aescudero@login3:~/purifying_selection^G(mafft) [aescudero@nodo4143 purifying_selection]$ ls[13d^[[0m^[[38;5;33maligned_pairs_INSIDE^[[0m[13;68H^[[38;5;33mchr18^[[0m[13;124H^[[38;5;40mrun1b_extract_outside_genes.sh^[[0m[181Grun$[14;1H^[[38;5;33maligned_pairs_OUTSIDE^[[0m[14;63H^[[38;5;33mchr20^[[0m[14;119H^[[38;5;40mrun1_extract_inversion_genes_boryana.sh^[[0m  run7_pre$[15;1Hborbonica_ortholog_cds_INSIDE.fasta[15;47H^[[38;5;33mchr3^[[0m[15;103H^[[38;5;40mrun1_extract_inversion_genes.sh^[[0m[15;160H^[[38;5;40mrun8_paml.sh^$[16;1Hborbonica_ortholog_cds_OUTSIDE.fasta[16;47H^[[38;5;33mcodon_aligned_dnds_INSIDE^[[0m[16;103Hrun2b_ortholog_mapper.py[16;144Hrun9_parse_paml.py[17dboryana_ortholog_cds_INSIDE.fasta[17;47H^[[38;5;33mcodon_aligned_dnds_OUTSIDE^[[0m[17;103Hrun2_ortholog_mapper.py[17;144Htarget_borbonica_genes_INSIDE.txt[18dboryana_ortholog_cds_OUTSIDE.fasta[18;47H^[[38;5;33mfiltered_alignments_INSIDE^[[0m[18;103Hrun3b_extract_cds.py[18;144Htarget_borbonica_genes_OUTSIDE.txt[19dCarex_borbonica.faa[19;47H^[[38;5;33mfiltered_alignments_OUTSIDE^[[0m[19;103Hrun3_extract_cds.py[19;144Htarget_boryana_genes_INSIDE.txt[20dCarex_borbonica.fna[20;47Hfinal_ortholog_list_INSIDE.txt[20;87Hrun4_align_orthologs.py[20;128Htarget_boryana_genes_OUTSIDE.txt[21dCarex_borbonica.gff3[21;47Hfinal_ortholog_list_OUTSIDE.txt[21;87Hrun4_align_orthologs_rc.py[21;128Htarget_boryana_orthologs_INSIDE.txt[22dCarex_boryana.faa[22;47H^[[38;5;33mpaml_inputs_INSIDE^[[0m[22;103Hrun4b_align_orthologs.py[22;144Htarget_boryana_orthologs_OUTSIDE.txt[23dCarex_boryana.fna[23;47H^[[38;5;33mpaml_inputs_OUTSIDE^[[0m[23;103Hrun5b_prepare_dnds_data.py[23;144H^[[38;5;33mtemp_pep_alignments_INSIDE^[[$[24;1HCarex_boryana.gff3[24;47Hrun10_gene_lists_strict.py[24;87Hrun5_prepare_dnds_data.py[24;128H^[[38;5;33mtemp_pep_alignments_OUTSIDE^[[0m[25dCarex_boryana_vs_Carex_borbonica.synHits.txt  run11_calc_4d.py[25;87Hrun6b_pal2nal_prep.py[26d^[[38;5;33mchr14^[[0m[26;63H^[[38;5;40mrun1b_extract_outside_genes_boryana.sh^[[0m  run6_pal2nal_prep.py[27d^[]0;aescudero@login3:~/purifying_selection^G(mafft) [aescudero@nodo4143 purifying_selection]$ ls^H^Hcp ./chr14/run9_parse_paml.py ./^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H^H$[28;1H--- PAML Result Aggregator Started ---[29d✅ Successfully wrote 6 raw results to results_omega_inside.csv[30d✅ Successfully wrote 335 raw results to results_omega_outside.csv[32dRaw total genes: Inside=6, Outside=335[34d✅ Exported 6 Inside gene IDs to gene_ids_aggregate_inside.txt ((B[0;1m[36mfor[39m(B[m P4D aggregate analysis).[35d✅ Exported 331 Outside gene IDs to gene_ids_aggregate_outside.txt ((B[0;1m[36mfor[39m(B[m P4D aggregate analysis).[36d\n--- AGGREGATE Divergence Rate Analysis (Filtered $d_S \leq 1.5$ Only) ---[37dFiltering: Removed 0 inside (B[0;1m[36mand[39m(B[m 4 outside genes due to $d_S > 1.5$.[38dRemaining genes: Inside=6, Outside=331.[39d| Group[39;21H| Total LN Sites | Total LS Sites | Aggregate dN | Aggregate dS | Aggregate ω (dN/dS) |[40d|-------------------|----------------|----------------|--------------|--------------|---------------------|[41d| Inside Inversion  | 6538[41;38H| 2048[41;55H| 0.0054[70G| 0.0061[85G| 0.8927[41;107H|[42d| Outside Inversion | 276230[42;38H| 104563[42;55H| 0.0089[70G| 0.0211[85G| 0.4240[42;107H|[3d[?12l[?25h[?25l[43d[J[45;184H[?12l[?25h[45;1H[?1049l[23;0;0t[?1l>]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ cp /hoe[Kme/aescudero/py[Kurifying_selection/chr14/run9_parse_paml.py ./
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ cp /home/aescudero/purifying_selection/chr14/run9_parse_paml.py ./[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[42Pnano run9_parse_paml.py [2@python[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
--- PAML Result Aggregator Started ---
✅ Successfully wrote 4 raw results to results_omega_inside.csv
✅ Successfully wrote 456 raw results to results_omega_outside.csv

Raw total genes: Inside=4, Outside=456

✅ Exported 3 Inside gene IDs to gene_ids_aggregate_inside.txt (for P4D aggregate analysis).
✅ Exported 451 Outside gene IDs to gene_ids_aggregate_outside.txt (for P4D aggregate analysis).
\n--- AGGREGATE Divergence Rate Analysis (Filtered $d_S \leq 1.5$ Only) ---
Filtering: Removed 1 inside and 5 outside genes due to $d_S > 1.5$.
Remaining genes: Inside=3, Outside=451.
| Group             | Total LN Sites | Total LS Sites | Aggregate dN | Aggregate dS | Aggregate ω (dN/dS) |
|-------------------|----------------|----------------|--------------|--------------|---------------------|
| Inside Inversion  | 1215           | 465            | 0.0008       | 0.0046       | 0.1772              |
| Outside Inversion | 414250         | 153554         | 0.0058       | 0.0214       | 0.2731              |
\n--- Ratio of FILTERED Aggregate Divergence Rates ($\omega_{\text{Inside}} / \omega_{\text{Outside}}$) ---
Ratio: 0.6488
Interpretation: The aggregated divergence rate ($\omega$) is 0.65 times higher inside the inversion (using $d_S \leq 1.5$ genes).
\n--- Significance Test for Aggregate $\omega$ Ratios (Chi-Squared Test on Total Substitutions) ---
Contingency Table (Total N changes vs Total S changes):
[[   1    2]
 [2421 3286]]
  Chi-Squared Statistic: 0.0000
  P-value: 1.00000
  Interpretation: {'Significant difference' if p_value < 0.05 else 'No significant difference'} in the aggregated $\omega$ ratio.

--- Strictly Filtered Data Summary (Used for subsequent tests and plotting) ---
Initial Total Gene Count: 460
Filter used: 0.01 ≤ dS ≤ 1.5 AND 0.01 ≤ dN ≤ 1.5
Final Filtered Genes Inside: 0
Final Filtered Genes Outside: 31
Error: One or both gene groups are empty after filtering. Analysis aborted.
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ python run9_parse_paml.py [K[K[K[K[K[K[K[K[K[K[K[K[K[K[K[K11_calc_4d.py 
--- 4D Metrics Calculator Started (Applying PAML Filter) ---
Loaded 3 Inside genes and 451 Outside genes from PAML filter lists.
Processing directory: paml_inputs_INSIDE (Targeted Genes: 3)
Processing directory: paml_inputs_OUTSIDE (Targeted Genes: 451)

✅ Successfully processed 454 genes (PAML filtered).
Results saved to p4d_filtered_results.csv

--- Starting 4D Metrics Statistical Analysis and Plotting (P4D, PAML Filtered) ---

--- 1. Gene-by-Gene P4D Ratio Analysis (ALL Filtered Genes) ---

Metric: Total 4D Substitutions (SNPs)
  Inside Inversion (N=3): Median = 0.000000, Mean = 0.333333
  Outside Inversion (N=451): Median = 0.000000, Mean = 1.529933
  Mann-Whitney U Test P-value: 0.8593

Metric: Total 4D Sites
  Inside Inversion (N=3): Median = 37.000000, Mean = 53.666667
  Outside Inversion (N=451): Median = 98.000000, Mean = 110.871397
  Mann-Whitney U Test P-value: 0.0666

Metric: P$_{4D}$ Ratio (Substitutions/Sites)
  Inside Inversion (N=3): Median = 0.000000, Mean = 0.003623
  Outside Inversion (N=451): Median = 0.000000, Mean = 0.011783
  Mann-Whitney U Test P-value: 0.7433

--- 2. Gene-by-Gene P4D Ratio Analysis (Filtered, Excluding Genes with 0 Substitutions) ---

Metric: P$_{4D}$ Ratio (Substitutions/Sites) (Non-Zero Subs Only)
  Inside Inversion (N=1): Median = 0.010870, Mean = 0.010870
  Outside Inversion (N=113): Median = 0.010638, Mean = 0.047029
  Not enough non-zero data points for statistical testing. Skipping MWU.


--- 3. Aggregate P4D Analysis (Pooling all 4D Sites) ---
Aggregate P4D Results:
  Inside Inversion:
    Total Substitutions (SNPs): 1
    Total Comparable 4D Sites: 161
    Aggregate P4D Ratio: 0.006211
  Outside Inversion:
    Total Substitutions (SNPs): 690
    Total Comparable 4D Sites: 50,003
    Aggregate P4D Ratio: 0.013799

--- 4. Chi-squared Test on Total Counts (Substitutions vs Non-Substitutions) ---
  Chi-squared Statistic: 0.2363
  Degrees of Freedom: 1
  P-value: 0.6269

✅ Analysis complete. Distribution plot saved to p4d_filtered_distribution.pdf
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ ls
[0m[38;5;33maligned_pairs_INSIDE[0m                          [38;5;33mchr3[0m                             run10_gene_lists_strict.py               run6_pal2nal_prep.py
[38;5;33maligned_pairs_OUTSIDE[0m                         [38;5;33mchr32[0m                            run11_calc_4d.py                         run7b_prepare_paml_input.py
borbonica_ortholog_cds_INSIDE.fasta           [38;5;33mcodon_aligned_dnds_INSIDE[0m        [38;5;40mrun1b_extract_outside_genes_boryana.sh[0m   run7_prepare_paml_input.py
borbonica_ortholog_cds_OUTSIDE.fasta          [38;5;33mcodon_aligned_dnds_OUTSIDE[0m       [38;5;40mrun1b_extract_outside_genes.sh[0m           [38;5;40mrun8_paml.sh[0m
boryana_ortholog_cds_INSIDE.fasta             [38;5;33mfiltered_alignments_INSIDE[0m       [38;5;40mrun1_extract_inversion_genes_boryana.sh[0m  run9_parse_paml.py
boryana_ortholog_cds_OUTSIDE.fasta            [38;5;33mfiltered_alignments_OUTSIDE[0m      [38;5;40mrun1_extract_inversion_genes.sh[0m          target_borbonica_genes_INSIDE.txt
Carex_borbonica.faa                           final_ortholog_list_INSIDE.txt   run2b_ortholog_mapper.py                 target_borbonica_genes_OUTSIDE.txt
Carex_borbonica.fna                           final_ortholog_list_OUTSIDE.txt  run2_ortholog_mapper.py                  target_boryana_genes_INSIDE.txt
Carex_borbonica.gff3                          gene_ids_aggregate_inside.txt    run3b_extract_cds.py                     target_boryana_genes_OUTSIDE.txt
Carex_boryana.faa                             gene_ids_aggregate_outside.txt   run3_extract_cds.py                      target_boryana_orthologs_INSIDE.txt
Carex_boryana.fna                             p4d_filtered_distribution.pdf    run4_align_orthologs.py                  target_boryana_orthologs_OUTSIDE.txt
Carex_boryana.gff3                            p4d_filtered_results.csv         run4_align_orthologs_rc.py               [38;5;33mtemp_pep_alignments_INSIDE[0m
Carex_boryana_vs_Carex_borbonica.synHits.txt  [38;5;33mpaml_inputs_INSIDE[0m               run4b_align_orthologs.py                 [38;5;33mtemp_pep_alignments_OUTSIDE[0m
[38;5;33mchr14[0m                                         [38;5;33mpaml_inputs_OUTSIDE[0m              run5b_prepare_dnds_data.py
[38;5;33mchr18[0m                                         results_omega_inside.csv         run5_prepare_dnds_data.py
[38;5;33mchr20[0m                                         results_omega_outside.csv        run6b_pal2nal_prep.py
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ mkdir chr28
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ mv aligned_pairs_* ./ch28[K[Kr28/
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ mv aligned_pairs_* ./chr28/[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1@b[1@o[1@r
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ mv bor* ./chr28/[1P[1P[1P[1@c[1@o[17@don_aligned_dnds_
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ mv codon_aligned_dnds_* ./chr28/[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1@f
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ mv f* ./chr28/[1P[1@g
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ mv g* ./chr28/[1P[1@p
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ mv p* ./chr28/[1P[1@r[1@e[12@sults_omega_
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ mv results_omega_* ./chr28/[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1P[1@t
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ ls
Carex_borbonica.faa                           [0m[38;5;33mchr14[0m                       run11_calc_4d.py                         run3b_extract_cds.py        run6b_pal2nal_prep.py
Carex_borbonica.fna                           [38;5;33mchr18[0m                       [38;5;40mrun1b_extract_outside_genes_boryana.sh[0m   run3_extract_cds.py         run6_pal2nal_prep.py
Carex_borbonica.gff3                          [38;5;33mchr20[0m                       [38;5;40mrun1b_extract_outside_genes.sh[0m           run4_align_orthologs.py     run7b_prepare_paml_input.py
Carex_boryana.faa                             [38;5;33mchr28[0m                       [38;5;40mrun1_extract_inversion_genes_boryana.sh[0m  run4_align_orthologs_rc.py  run7_prepare_paml_input.py
Carex_boryana.fna                             [38;5;33mchr3[0m                        [38;5;40mrun1_extract_inversion_genes.sh[0m          run4b_align_orthologs.py    [38;5;40mrun8_paml.sh[0m
Carex_boryana.gff3                            [38;5;33mchr32[0m                       run2b_ortholog_mapper.py                 run5b_prepare_dnds_data.py  run9_parse_paml.py
Carex_boryana_vs_Carex_borbonica.synHits.txt  run10_gene_lists_strict.py  run2_ortholog_mapper.py                  run5_prepare_dnds_data.py
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ lsmv t* ./chr28/[1P[1@r[1@u[1P[1P[1@c[1@p
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ ls
Carex_borbonica.faa                           [0m[38;5;33mchr14[0m                       run11_calc_4d.py                         run3b_extract_cds.py        run6b_pal2nal_prep.py
Carex_borbonica.fna                           [38;5;33mchr18[0m                       [38;5;40mrun1b_extract_outside_genes_boryana.sh[0m   run3_extract_cds.py         run6_pal2nal_prep.py
Carex_borbonica.gff3                          [38;5;33mchr20[0m                       [38;5;40mrun1b_extract_outside_genes.sh[0m           run4_align_orthologs.py     run7b_prepare_paml_input.py
Carex_boryana.faa                             [38;5;33mchr28[0m                       [38;5;40mrun1_extract_inversion_genes_boryana.sh[0m  run4_align_orthologs_rc.py  run7_prepare_paml_input.py
Carex_boryana.fna                             [38;5;33mchr3[0m                        [38;5;40mrun1_extract_inversion_genes.sh[0m          run4b_align_orthologs.py    [38;5;40mrun8_paml.sh[0m
Carex_boryana.gff3                            [38;5;33mchr32[0m                       run2b_ortholog_mapper.py                 run5b_prepare_dnds_data.py  run9_parse_paml.py
Carex_boryana_vs_Carex_borbonica.synHits.txt  run10_gene_lists_strict.py  run2_ortholog_mapper.py                  run5_prepare_dnds_data.py
]0;aescudero@login3:~/purifying_selection(mafft) [aescudero@nodo4143 purifying_selection]$ cd ..
]0;aescudero@login3:~(mafft) [aescudero@nodo4143 ~]$ ls
[0m[38;5;33mbeast[0m  [38;5;33mbin[0m  [38;5;33mgenome_assembly[0m  [38;5;33mhelodes[0m  [38;5;33miqtree[0m  [38;5;33mminiforge3[0m  [38;5;33mpurifying_selection[0m  [38;5;33mR[0m  [38;5;33mradseq[0m  [38;5;33msoftware[0m  [38;5;33mSTRUCTURE[0m  [38;5;33msyri[0m  syri.log  [38;5;33mwgg[0m
]0;aescudero@login3:~(mafft) [aescudero@nodo4143 ~]$ cd wgg/
]0;aescudero@login3:~/wgg(mafft) [aescudero@nodo4143 wgg]$ ls
[0m[38;5;33madmixture_data[0m                  [38;5;33miqtree[0m                         [38;5;33msv_gene_lists[0m                       wgg_bcftools_call.sbatch        wgg_run_fst_2pops.sbatch
align_and_process_array.sbatch  [38;5;33mlogs[0m                           wgg_admixture_prep_chr14_ld.sbatch  wgg_run_admixture_chr14.sbatch  wgg_run_fst_final_RDAprep.sbatch
[38;5;33mbams[0m                            maptophylo.R                   wgg_admixture_prep_chr18_ld.sbatch  wgg_run_admixture_chr18.sbatch  wgg_run_fst_final_RDAprep_sumsampling.sbatch
[38;5;40mcheck_file_structure.sh[0m         prep_go_annotation_chrom14.py  wgg_admixture_prep_chr20_ld.sbatch  wgg_run_admixture_chr20.sbatch  wgg_run_fst_final.sbatch
fst_analysis_1039219.err        [38;5;33mrawdata[0m                        wgg_admixture_prep_chr28_ld.sbatch  wgg_run_admixture_chr28.sbatch  wgg_run_windowed_pi_and_fst.sbatch
fst_analysis_1039219.out        [38;5;33mRDA[0m                            wgg_admixture_prep_chr32_ld.sbatch  wgg_run_admixture_chr32.sbatch  wgg_vcf2phylip_snpsites_maf05.sbatch
[38;5;33mfst_results[0m                     samples_list_from_vcf.txt      wgg_admixture_prep_chr3_ld.sbatch   wgg_run_admixture_chr3.sbatch   wgg_vcf_filter.sbatch
[38;5;33mgenomes[0m                         samples.txt                    wgg_admixture_prep_final2.sbatch    wgg_run_admixture_cv.sbatch
[38;5;33mgo_annotation[0m                   sv_gene_extraction.py          wgg_admixture_prep_final.sbatch     wgg_run_admixture_final.sbatch
]0;aescudero@login3:~/wgg(mafft) [aescudero@nodo4143 wgg]$ cat wgg_run_windowed_pi_and_fst.sbatch
#!/bin/bash
#SBATCH --job-name=windowed_pi_fst_analysis
#SBATCH --output=windowed_pi_fst_analysis_%j.out
#SBATCH --error=windowed_pi_fst_analysis_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=long

set -e

echo "--- Starting Windowed pi/Fst Analysis Workflow ---"
date +"%Y-%m-%d %H:%M:%S"

# --- Configuration ---
CONDA_ENV_NAME="bcftools"
PLINK_FILTERED_PREFIX="/home/aescudero/wgg/admixture_data/admixture_final_ld_pruned_filtered"
POP_FILE_RAW="/home/aescudero/wgg/admixture_data/population_list.txt"
UPDATE_IDS_FILE="/home/aescudero/wgg/admixture_data/update_ids.txt"
VCF_OUTPUT_BCF_DIRTY="${PLINK_FILTERED_PREFIX}.bcf" # Temporary BCF with dirty IDs
VCF_OUTPUT_BCF_CLEAN="${PLINK_FILTERED_PREFIX}_clean.bcf" # Final BCF with clean IDs
RESULTS_DIR="/home/aescudero/wgg/fst_results/windowed_stats"

WINDOW_SIZE=1000
CHR_SET=34

# --- Activate Conda environment ---
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source "/home/aescudero/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}" || { echo "Error: Failed to activate Conda environment '${CONDA_ENV_NAME}'. Exiting." >&2; exit 1; }
echo "Conda environment '${CONDA_ENV_NAME}' activated."

# --- Create Output Directories ---
echo "Creating output directories..."
mkdir -p "${RESULTS_DIR}"

# --- 1. CONVERT PLINK to VCF/BCF (Outputting VCF with dirty IDs) ---
echo "1. Converting PLINK to BCF format..."
plink --bfile "${PLINK_FILTERED_PREFIX}" \
      --recode vcf \
      --out "${PLINK_FILTERED_PREFIX}" \
      --chr-set ${CHR_SET} no-xy

bcftools view -O b -o "${VCF_OUTPUT_BCF_DIRTY}" "${PLINK_FILTERED_PREFIX}.vcf"

# --- 2. CRITICAL STEP: CLEAN SAMPLE IDs IN THE BCF HEADER ---
echo "2.1 Reheadering BCF to clean sample IDs (matching 184 samples)..."
NEW_SAMPLE_LIST="${RESULTS_DIR}/new_sample_names_184.txt"
DIRTY_ORDER_LIST="${RESULTS_DIR}/dirty_vcf_order.txt"
ID_MAP="${RESULTS_DIR}/id_map.txt"

# A. Extract the 184 dirty IDs currently in the BCF, in their precise order.
echo "A. Extracting 184 sample IDs from the BCF header..."
bcftools query -l "${VCF_OUTPUT_BCF_DIRTY}" > "${DIRTY_ORDER_LIST}"
DIRTY_COUNT=$(wc -l < "${DIRTY_ORDER_LIST}")
echo "BCF contains ${DIRTY_COUNT} samples."

# B. Create the map file (Dirty_IID \t Clean_IID) from update_ids.txt
# Assumes Col 2 (Dirty IID) maps to Col 4 (Clean IID).
echo "B. Creating map file from update_ids.txt (Dirty_IID from Col 2 -> Clean_IID from Col 4)..."
awk '{print $2 "\t" $4}' "${UPDATE_IDS_FILE}" > "${ID_MAP}"

# C. Generate list of clean IDs, correcting for mangled names using sequential lookup.
echo "C. Generating list of clean IDs in the correct BCF order (using sequential lookup for mangled IDs)..."
> "${NEW_SAMPLE_LIST}" # Clear the output list

# Process the dirty IDs one by one and look up the clean ID from the map file
while read DIRTY_VCF_ID; do
    
    CLEAN_ID=""
    
    # 1. Try a direct lookup first
    CLEAN_ID=$(grep -w "${DIRTY_VCF_ID}" "${ID_MAP}" | awk '{print $2}')
    
    if [[ -z "${CLEAN_ID}" ]]; then
        
        # 2. Sequential attempt for mangled IDs (e.g., C12ME24_1_C12ME24_1 -> C12ME24_1)
        
        # Method A: Strip the shortest suffix matching '_*' (e.g., C12ME24_1_C12ME24_1 -> C12ME24_1)
        BASE_DIRTY_ID_A="${DIRTY_VCF_ID%_*}"
        CLEAN_ID=$(grep -w "${BASE_DIRTY_ID_A}" "${ID_MAP}" | awk '{print $2}')
        
        # If still not found, try Method B: Force split on the first two underscore fields.
        if [[ -z "${CLEAN_ID}" ]]; then
            # This handles the case where the ID has two parts separated by an underscore (ID_P1_ID_P1)
            BASE_DIRTY_ID_B=$(echo "${DIRTY_VCF_ID}" | awk -F'_' '{print $1"_"$2}')
            CLEAN_ID=$(grep -w "${BASE_DIRTY_ID_B}" "${ID_MAP}" | awk '{print $2}')
        fi
    fi

    # Final result check
    if [[ -n "${CLEAN_ID}" ]]; then
        echo "${CLEAN_ID}" >> "${NEW_SAMPLE_LIST}"
    else
        echo "Error: Could not find clean ID for dirty ID: ${DIRTY_VCF_ID}. Exiting." >&2
        exit 1
    fi
    
done < "${DIRTY_ORDER_LIST}"

# D. Final check and Reheader the BCF
NEW_SAMPLE_COUNT=$(wc -l < "${NEW_SAMPLE_LIST}")
if [ "${NEW_SAMPLE_COUNT}" -ne "${DIRTY_COUNT}" ]; then
    echo "CRITICAL ERROR: Mismatch! New sample list count is ${NEW_SAMPLE_COUNT}, but BCF has ${DIRTY_COUNT}. Exiting." >&2
    exit 1
fi
echo "New sample list of ${NEW_SAMPLE_COUNT} samples generated successfully."

bcftools reheader -s "${NEW_SAMPLE_LIST}" -o "${VCF_OUTPUT_BCF_CLEAN}" "${VCF_OUTPUT_BCF_DIRTY}"
echo "BCF reheadered successfully to ${VCF_OUTPUT_BCF_CLEAN}."

# Set the main BCF variable to the newly cleaned file for subsequent steps
VCF_OUTPUT_BCF="${VCF_OUTPUT_BCF_CLEAN}"

# --- 2.2 GENERATE VCFtools Population Lists (using the clean IDs that now match the BCF) ---
echo "2.2 Generating VCFtools population files from the clean population_list.txt."
tail -n +2 "${POP_FILE_RAW}" | awk '$2 == "POP1" {print $1}' > "${RESULTS_DIR}/pop1_samples_cleaned.txt"
tail -n +2 "${POP_FILE_RAW}" | awk '$2 == "POP2" {print $1}' > "${RESULTS_DIR}/pop2_samples_cleaned.txt"

POP1_LIST="${RESULTS_DIR}/pop1_samples_cleaned.txt"
POP2_LIST="${RESULTS_DIR}/pop2_samples_cleaned.txt"

# --- 3. CALCULATE Fst (Weir & Cockerham) in Windows ---
echo "3. Calculating Fst in ${WINDOW_SIZE} bp windows..."
vcftools --bcf "${VCF_OUTPUT_BCF}" \
         --weir-fst-pop "${POP1_LIST}" \
         --weir-fst-pop "${POP2_LIST}" \
         --fst-window-size ${WINDOW_SIZE} \
         --fst-window-step ${WINDOW_SIZE} \
         --out "${RESULTS_DIR}/windowed_fst_${WINDOW_SIZE}bp"

# --- 4. CALCULATE PI (Nucleotide Diversity) in Windows ---
echo "4. Calculating Pi in ${WINDOW_SIZE} bp windows for each population..."

# POP1 Pi - Removed the unsupported --window-step flag.
vcftools --bcf "${VCF_OUTPUT_BCF}" \
         --keep "${POP1_LIST}" \
         --window-pi ${WINDOW_SIZE} \
         --out "${RESULTS_DIR}/windowed_pi_pop1_${WINDOW_SIZE}bp"

# POP2 Pi - Removed the unsupported --window-step flag.
vcftools --bcf "${VCF_OUTPUT_BCF}" \
         --keep "${POP2_LIST}" \
         --window-pi ${WINDOW_SIZE} \
         --out "${RESULTS_DIR}/windowed_pi_pop2_${WINDOW_SIZE}bp"

echo "--- Windowed pi/Fst Analysis Workflow Completed ---"
date +"%Y-%m-%d %H:%M:%S"
]0;aescudero@login3:~/wgg(mafft) [aescudero@nodo4143 wgg]$ cd fst_results/
]0;aescudero@login3:~/wgg/fst_results(mafft) [aescudero@nodo4143 fst_results]$ ls
admixture_final_ld_pruned_filtered.fst                                                   plot_fst.py
admixture_final_ld_pruned_filtered.fst.sorted                                            plot_fst_syri2.py
admixture_final_ld_pruned_filtered.log                                                   plot_fst_syri3.py
admixture_final_ld_pruned_filtered.nosex                                                 plot_fst_syri4.py
admixture_final_ld_pruned_filtered_pop1_vs_pop2.fst                                      plot_fst_syri4_test.py
admixture_final_ld_pruned_filtered_pop1_vs_pop2.fst.sorted                               plot_fst_syri5_test.py
admixture_final_ld_pruned_filtered_pop1_vs_pop2.log                                      plot_fst_syri.py
admixture_final_ld_pruned_filtered_pop1_vs_pop2.nosex                                    [0m[38;5;33mpop37[0m
[38;5;13mfst_manhattan_plot_high_Fst_10k_rearrangementsINVDUPTRANSINVDPINVTR_with_enrichment.png[0m  [38;5;33msv_chromosome_plots[0m
[38;5;13mfst_manhattan_plot_high_Fst_10k_rearrangementsINVDUPTRANS.png[0m                            sv_chromosome_plots_30.py
[38;5;13mfst_manhattan_plot_high_Fst_10k_rearrangementsINVDUPTRANS_with_enrichment.png[0m            sv_chromosome_plots.py
[38;5;13mfst_manhattan_plot_high_Fst_10k_rearrangements.png[0m                                       [38;5;33msv_density_plots[0m
[38;5;13mfst_manhattan_plot_high_Fst_filtered.png[0m                                                 SV_Enrichment_and_population_divergence_analysis_multiplots_30.py
[38;5;13mfst_manhattan_plot_high_values.png[0m                                                       SV_Enrichment_and_population_divergence_analysis_multiplots.py
[38;5;13mfst_manhattan_plot_high_values_significant_rearrangements.png[0m                            [38;5;33msv_local_density_plots[0m
[38;5;13mfst_manhattan_plot_high_values_with_all_rearrangements.png[0m                               sv_local_density_plots_30.py
[38;5;13mfst_manhattan_plot_pop1_vs_pop2.png[0m                                                      sv_local_density_plots.py
[38;5;13mfst_manhattan_plot_resized_filtered_pop1_vs_pop2.png[0m                                     threshold]
plot_fst09.py                                                                            [38;5;33mwindowed_stats[0m
]0;aescudero@login3:~/wgg/fst_results(mafft) [aescudero@nodo4143 fst_results]$ -[Kls -la -t
total 222364
-rw-rw-r--  1 aescudero aescudero 29040320 Nov 18 12:08 admixture_final_ld_pruned_filtered.fst.sorted
-rw-rw-r--  1 aescudero aescudero 29040342 Nov 18 12:08 admixture_final_ld_pruned_filtered.fst
-rw-rw-r--  1 aescudero aescudero     1342 Nov 18 12:08 admixture_final_ld_pruned_filtered.log
-rw-rw-r--  1 aescudero aescudero     3796 Nov 18 12:08 admixture_final_ld_pruned_filtered.nosex
drwxrwxr-x 12 aescudero aescudero   540672 Nov 18 12:08 [0m[38;5;33m..[0m
-rw-rw-r--  1 aescudero aescudero     8576 Nov 10 16:36 sv_chromosome_plots_30.py
-rw-rw-r--  1 aescudero aescudero    14002 Nov 10 16:34 sv_local_density_plots_30.py
-rw-rw-r--  1 aescudero aescudero    13984 Nov 10 16:32 sv_local_density_plots.py
drwxrwxr-x  2 aescudero aescudero     4096 Nov 10 15:07 [38;5;33msv_chromosome_plots[0m
drwxrwxr-x  7 aescudero aescudero     4096 Nov 10 15:06 [38;5;33m.[0m
drwxrwxr-x  2 aescudero aescudero     4096 Nov 10 13:57 [38;5;33msv_local_density_plots[0m
drwxrwxr-x  2 aescudero aescudero     4096 Nov 10 13:49 [38;5;33msv_density_plots[0m
-rw-rw-r--  1 aescudero aescudero    14079 Nov 10 13:40 SV_Enrichment_and_population_divergence_analysis_multiplots_30.py
-rw-rw-r--  1 aescudero aescudero  5306172 Nov  6 20:34 [38;5;13mfst_manhattan_plot_high_Fst_10k_rearrangementsINVDUPTRANSINVDPINVTR_with_enrichment.png[0m
-rw-rw-r--  1 aescudero aescudero     8903 Nov  6 12:02 plot_fst_syri5_test.py
-rw-rw-r--  1 aescudero aescudero     7947 Oct  6 19:45 sv_chromosome_plots.py
-rw-rw-r--  1 aescudero aescudero    14061 Oct  6 16:41 SV_Enrichment_and_population_divergence_analysis_multiplots.py
drwxrwxr-x  2 aescudero aescudero     4096 Oct  6 13:42 [38;5;33mwindowed_stats[0m
-rw-rw-r--  1 aescudero aescudero   561597 Sep  9 16:44 [38;5;13mfst_manhattan_plot_high_Fst_10k_rearrangementsINVDUPTRANS_with_enrichment.png[0m
-rw-rw-r--  1 aescudero aescudero     8868 Sep  9 14:59 plot_fst_syri4_test.py
-rw-rw-r--  1 aescudero aescudero  1164254 Aug  8 16:53 [38;5;13mfst_manhattan_plot_high_Fst_10k_rearrangementsINVDUPTRANS.png[0m
-rw-rw-r--  1 aescudero aescudero     4600 Aug  8 16:53 plot_fst_syri4.py
-rw-rw-r--  1 aescudero aescudero  1181312 Aug  8 14:25 [38;5;13mfst_manhattan_plot_high_Fst_10k_rearrangements.png[0m
-rw-rw-r--  1 aescudero aescudero     4640 Aug  8 14:25 plot_fst_syri3.py
-rw-rw-r--  1 aescudero aescudero     4442 Aug  8 14:12 plot_fst_syri2.py
-rw-rw-r--  1 aescudero aescudero    76144 Aug  8 14:07 [38;5;13mfst_manhattan_plot_high_Fst_filtered.png[0m
-rw-rw-r--  1 aescudero aescudero        0 Aug  8 11:29 threshold]
-rw-rw-r--  1 aescudero aescudero   147242 Aug  8 11:25 [38;5;13mfst_manhattan_plot_resized_filtered_pop1_vs_pop2.png[0m
-rw-rw-r--  1 aescudero aescudero   404121 Aug  7 15:40 [38;5;13mfst_manhattan_plot_high_values_significant_rearrangements.png[0m
-rw-rw-r--  1 aescudero aescudero     3983 Aug  7 15:38 plot_fst_syri.py
-rw-rw-r--  1 aescudero aescudero   457141 Aug  7 13:49 [38;5;13mfst_manhattan_plot_high_values_with_all_rearrangements.png[0m
-rw-rw-r--  1 aescudero aescudero   397646 Aug  7 13:18 [38;5;13mfst_manhattan_plot_high_values.png[0m
-rw-rw-r--  1 aescudero aescudero     2113 Aug  7 13:17 plot_fst09.py
-rw-rw-r--  1 aescudero aescudero    95877 Aug  7 13:09 [38;5;13mfst_manhattan_plot_pop1_vs_pop2.png[0m
-rw-rw-r--  1 aescudero aescudero     2017 Aug  7 13:08 plot_fst.py
-rw-rw-r--  1 aescudero aescudero 79534576 Aug  7 12:59 admixture_final_ld_pruned_filtered_pop1_vs_pop2.fst.sorted
-rw-rw-r--  1 aescudero aescudero 79534598 Aug  7 12:59 admixture_final_ld_pruned_filtered_pop1_vs_pop2.fst
-rw-rw-r--  1 aescudero aescudero     1383 Aug  7 12:59 admixture_final_ld_pruned_filtered_pop1_vs_pop2.log
-rw-rw-r--  1 aescudero aescudero     3796 Aug  7 12:59 admixture_final_ld_pruned_filtered_pop1_vs_pop2.nosex
drwxrwxr-x  2 aescudero aescudero     4096 Aug  7 12:55 [38;5;33mpop37[0m
]0;aescudero@login3:~/wgg/fst_results(mafft) [aescudero@nodo4143 fst_results]$ sv_local_density_plots_30.pycsv_local_density_plots_30.pyasv_local_density_plots_30.pytsv_local_density_plots_30.py sv_local_density_plots_30.py
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
]0;aescudero@login3:~/wgg/fst_results(mafft) [aescudero@nodo4143 fst_results]$ [K(mafft) [aescudero@nodo4143 fst_results]$ [K(mafft) [aescudero@nodo4143 fst_results]$ cond[Kda deactivate
]0;aescudero@login3:~/wgg/fst_results(base) [aescudero@nodo4143 fst_results]$ conda activate plot_env
]0;aescudero@login3:~/wgg/fst_results(plot_env) [aescudero@nodo4143 fst_results]$ conda activate plot_env[7Pdeactivateat sv_local_density_plots_30.py[C[1P sv_local_density_plots_30.py[1P sv_local_density_plots_30.py[1P sv_local_density_plots_30.pyp sv_local_density_plots_30.pyy sv_local_density_plots_30.pyt sv_local_density_plots_30.pyh sv_local_density_plots_30.pyo sv_local_density_plots_30.pyn sv_local_density_plots_30.py
Loading and processing windowed statistics...
Total 1000bp windows analyzed: 328966

Loading and processing SyRI data...
Analyzing enrichment within the top 30 largest SVs (>10000 bp):
SV_ID  TYPE  CHR_int    START      END    SIZE
  SV1   INV       14  3040103  5331878 2291775
  SV2   INV       18  1681715  2543617  861902
  SV3   INV       20  4207035  4474484  267449
  SV4   INV        3  6538064  6773584  235520
  SV5 INVTR       11   572229   807176  234947
  SV6 INVTR       11   423763   645432  221669
  SV7   DUP        2 15863192 16067646  204454
  SV8 INVTR       11   492162   686228  194066
  SV9 INVTR        2 16163408 16349879  186471
 SV10 INVDP        3   544317   723684  179367
 SV11 INVTR        4   959624  1122892  163268
 SV12 TRANS        3   235306   373115  137809
 SV13 INVDP        2 16098557 16233821  135264
 SV14 INVDP        2 16125649 16260866  135217
 SV15   DUP        2 15962711 16092230  129519
 SV16   INV       32   890484  1017805  127321
 SV17   INV       22  8850975  8974282  123307
 SV18   DUP        2 16085119 16205029  119910
 SV19   DUP        2 16303904 16410644  106740
 SV20   INV       25  8811273  8913117  101844
 SV21 INVDP        2 16027599 16111681   84082
 SV22 INVTR       27  8928753  9012287   83534
 SV23   INV       28  4452397  4532837   80440
 SV24   INV       14 10730438 10807332   76894
 SV25   INV       20 10448895 10519691   70796
 SV26   DUP        2 16412801 16476160   63359
 SV27 INVDP       18  5445457  5508555   63098
 SV28   INV       21  8212305  8272819   60514
 SV29   DUP        2 16413586 16473905   60319
 SV30 INVTR       31  7660902  7717996   57094

================================================================================
--- Individual Statistical Tests (Mann-Whitney U) for Top 30 SVs vs. Local Chromosome Background ---
================================================================================
Skipping SV13 (Chr2, INVDP) due to insufficient data (SV: 4, BG: 14134 windows).
Skipping SV18 (Chr2, DUP) due to insufficient data (SV: 2, BG: 14136 windows).
Skipping SV19 (Chr2, DUP) due to insufficient data (SV: 3, BG: 14135 windows).
Skipping SV21 (Chr2, INVDP) due to insufficient data (SV: 2, BG: 14135 windows).
SV_ID  TYPE  CHR   SIZE_kb  SV_Windows  BG_Windows  FST_Mean   FST_P         FST_Sig  PI_POP1_Mean PI_POP1_P     PI_POP1_Sig  PI_POP2_Mean PI_POP2_P     PI_POP2_Sig  PI_RATIO_Mean PI_RATIO_P    PI_RATIO_Sig
  SV1   INV   14 2291.7750        1956        7730    0.5597 < 0.001     Significant        0.0009   < 0.001     Significant        0.0018     0.359 Not Significant         0.9430    < 0.001     Significant
  SV2   INV   18  861.9020         723        8224    0.4194 < 0.001     Significant        0.0014   < 0.001     Significant        0.0018     0.066 Not Significant         1.2099    < 0.001     Significant
  SV3   INV   20  267.4490         225        9019    0.6299 < 0.001     Significant        0.0009   < 0.001     Significant        0.0017     0.007     Significant         1.0490    < 0.001     Significant
  SV4   INV    3  235.5200         202       12442    0.4673 < 0.001     Significant        0.0009   < 0.001     Significant        0.0019     0.035     Significant         1.0411    < 0.001     Significant
  SV5 INVTR   11  234.9470          37       10427    0.2021   1.000 Not Significant        0.0012     0.158 Not Significant        0.0008   < 0.001     Significant         2.2985    < 0.001     Significant
  SV6 INVTR   11  221.6690          26       10439    0.2084   1.000 Not Significant        0.0007     0.002     Significant        0.0007   < 0.001     Significant         3.8127      0.074 Not Significant
  SV7   DUP    2  204.4540          33       14104    0.1960   1.000 Not Significant        0.0005   < 0.001     Significant        0.0008   < 0.001     Significant         1.2756      0.571 Not Significant
  SV8 INVTR   11  194.0660          25       10440    0.2070   1.000 Not Significant        0.0009     0.033     Significant        0.0008   < 0.001     Significant         1.5665      0.045     Significant
  SV9 INVTR    2  186.4710          11       14127    0.2161   1.000 Not Significant        0.0008     0.144 Not Significant        0.0005   < 0.001     Significant         1.8413      0.040     Significant
 SV10 INVDP    3  179.3670          20       12625    0.2073   1.000 Not Significant        0.0009     0.011     Significant        0.0007   < 0.001     Significant         1.3185      0.444 Not Significant
 SV11 INVTR    4  163.2680         155       12082    0.2376   1.000 Not Significant        0.0027   < 0.001     Significant        0.0021   < 0.001     Significant         1.4302    < 0.001     Significant
 SV12 TRANS    3  137.8090          32       12614    0.2096   1.000 Not Significant        0.0010     0.017     Significant        0.0010   < 0.001     Significant         6.6754      0.003     Significant
 SV14 INVDP    2  135.2170           5       14132    0.2847   0.957 Not Significant        0.0010     0.499 Not Significant        0.0006     0.020     Significant         1.5893      0.257 Not Significant
 SV15   DUP    2  129.5190           5       14133    0.1832   0.997 Not Significant        0.0003     0.016     Significant        0.0005     0.012     Significant         1.8973      0.932 Not Significant
 SV16   INV   32  127.3210         117        6468    0.5865 < 0.001     Significant        0.0008   < 0.001     Significant        0.0017     0.315 Not Significant         1.6960    < 0.001     Significant
 SV17   INV   22  123.3070         120        8879    0.3312   0.986 Not Significant        0.0016     0.034     Significant        0.0024     0.004     Significant         0.7081      0.714 Not Significant
 SV20   INV   25  101.8440          97        8274    0.1386   1.000 Not Significant        0.0026   < 0.001     Significant        0.0025     0.037     Significant         1.2462      0.005     Significant
 SV22 INVTR   27   83.5340          55        8022    0.2512   1.000 Not Significant        0.0019     0.152 Not Significant        0.0015     0.003     Significant         4.5156    < 0.001     Significant
 SV23   INV   28   80.4400          75        7401    0.4592   0.011     Significant        0.0014     0.405 Not Significant        0.0018     0.374 Not Significant         0.9321      0.523 Not Significant
 SV24   INV   14   76.8940          60        9626    0.3275   1.000 Not Significant        0.0023   < 0.001     Significant        0.0014     0.029     Significant         1.9220    < 0.001     Significant
 SV25   INV   20   70.7960          68        9178    0.2636   1.000 Not Significant        0.0020   < 0.001     Significant        0.0016     0.452 Not Significant         2.4415    < 0.001     Significant
 SV26   DUP    2   63.3590           6       14132    0.2441   0.992 Not Significant        0.0010     0.767 Not Significant        0.0002   < 0.001     Significant        17.8856    < 0.001     Significant
 SV27 INVDP   18   63.0980          54        8894    0.1665   1.000 Not Significant        0.0021   < 0.001     Significant        0.0016     0.364 Not Significant         1.6029    < 0.001     Significant
 SV28   INV   21   60.5140          55        8927    0.2804   1.000 Not Significant        0.0014     0.274 Not Significant        0.0016     0.113 Not Significant         4.0277      0.012     Significant
 SV29   DUP    2   60.3190           6       14132    0.2441   0.992 Not Significant        0.0010     0.767 Not Significant        0.0002   < 0.001     Significant        17.8856    < 0.001     Significant
 SV30 INVTR   31   57.0940          22        6712    0.2100   0.914 Not Significant        0.0011   < 0.001     Significant        0.0008   < 0.001     Significant         0.9726      0.419 Not Significant

Note on Mann-Whitney U Test Alternatives:
- FST: SV > Local Background (Alternative: 'greater')
- PI metrics and Ratio: SV $\ne$ Local Background (Alternative: 'two-sided')

================================================================================
--- Generating 4-Panel Density Plots in 'sv_local_density_plots/' ---
================================================================================
Generated sv_local_density_plots/SV1_INV_Chr14_local_analysis.png
Generated sv_local_density_plots/SV2_INV_Chr18_local_analysis.png
Generated sv_local_density_plots/SV3_INV_Chr20_local_analysis.png
Generated sv_local_density_plots/SV4_INV_Chr3_local_analysis.png
Generated sv_local_density_plots/SV5_INVTR_Chr11_local_analysis.png
Generated sv_local_density_plots/SV6_INVTR_Chr11_local_analysis.png
Generated sv_local_density_plots/SV7_DUP_Chr2_local_analysis.png
Generated sv_local_density_plots/SV8_INVTR_Chr11_local_analysis.png
Generated sv_local_density_plots/SV9_INVTR_Chr2_local_analysis.png
Generated sv_local_density_plots/SV10_INVDP_Chr3_local_analysis.png
Generated sv_local_density_plots/SV11_INVTR_Chr4_local_analysis.png
Generated sv_local_density_plots/SV12_TRANS_Chr3_local_analysis.png
Generated sv_local_density_plots/SV14_INVDP_Chr2_local_analysis.png
Generated sv_local_density_plots/SV15_DUP_Chr2_local_analysis.png
Generated sv_local_density_plots/SV16_INV_Chr32_local_analysis.png
Generated sv_local_density_plots/SV17_INV_Chr22_local_analysis.png
Generated sv_local_density_plots/SV20_INV_Chr25_local_analysis.png
Generated sv_local_density_plots/SV22_INVTR_Chr27_local_analysis.png
Generated sv_local_density_plots/SV23_INV_Chr28_local_analysis.png
Generated sv_local_density_plots/SV24_INV_Chr14_local_analysis.png
Generated sv_local_density_plots/SV25_INV_Chr20_local_analysis.png
Generated sv_local_density_plots/SV26_DUP_Chr2_local_analysis.png
Generated sv_local_density_plots/SV27_INVDP_Chr18_local_analysis.png
Generated sv_local_density_plots/SV28_INV_Chr21_local_analysis.png
Generated sv_local_density_plots/SV29_DUP_Chr2_local_analysis.png
Generated sv_local_density_plots/SV30_INVTR_Chr31_local_analysis.png

--- Analysis Complete ---
]0;aescudero@login3:~/wgg/fst_results(plot_env) [aescudero@nodo4143 fst_results]$ python sv_local_density_plots_30.py[1Psv_local_density_plots_30.py[1Psv_local_density_plots_30.py[1Psv_local_density_plots_30.py[1Psv_local_density_plots_30.py[1Psv_local_density_plots_30.py[1Psv_local_density_plots_30.py[1Psv_local_density_plots_30.pycsv_local_density_plots_30.pyasv_local_density_plots_30.pytsv_local_density_plots_30.py sv_local_density_plots_30.py
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
]0;aescudero@login3:~/wgg/fst_results(plot_env) [aescudero@nodo4143 fst_results]$ cat sv_local_density_plots_30.py_.pya.pyl.pyl.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.pyn sv_local_density_plots_30_all.pya sv_local_density_plots_30_all.pyn sv_local_density_plots_30_all.pyo sv_local_density_plots_30_all.py
[?1049h[22;0;0t[1;45r(B[m[4l[?7h[39;49m[?1h=[?1h=[?1h=[?25l[39;49m(B[m[H[2J[43;87H(B[0;7m[ New File ](B[m[H(B[0;7m  GNU nano 2.9.8                                                               sv_local_density_plots_30_all.py                                                                         [1;183H(B[m[44d(B[0;7m^G(B[m Get Help     (B[0;7m^O(B[m Write Out    (B[0;7m^W(B[m Where Is     (B[0;7m^K(B[m Cut Text     (B[0;7m^J(B[m Justify	(B[0;7m^C(B[m Cur Pos	(B[0;7mM-U(B[m Undo[44;113H(B[0;7mM-A(B[m Mark Text   (B[0;7mM-](B[m To Bracket  (B[0;7mM-▲(B[m Previous    (B[0;7m^B(B[m Back[45d(B[0;7m^X(B[m Exit[45;17H(B[0;7m^R(B[m Read File    (B[0;7m^\(B[m Replace	(B[0;7m^U(B[m Uncut Text   (B[0;7m^T(B[m To Linter    (B[0;7m^_(B[m Go To Line   (B[0;7mM-E(B[m Redo[45;113H(B[0;7mM-6(B[m Copy Text   (B[0;7mM-W(B[m WhereIs Next(B[0;7mM-▼(B[m Next[45;161H(B[0;7m^F(B[m Forward[43d[3d[39;49m(B[m[?12l[?25h[?25l[1;175H(B[0;7mModified(B[m[3;12H[42m[1K[4d(B[0;1m[31m # Plot Local Background Distribution[5;13H[39m(B[msns.kdeplot(df_bg_local[stat_col].dropna(), ax=ax, color=(B[0;1m[32m'gray'[39m(B[m, fill=(B[0;1m[35mTrue[39m(B[m, alpha=0.5, label=(B[0;1m[32m'Local Background'[39m(B[m)[6;12H(B[0;1m[31m # Plot SV Distribution[7;13H[39m(B[msns.kdeplot(sv_loci[stat_col].dropna(), ax=ax, color=config[(B[0;1m[32m'color'[39m(B[m], fill=(B[0;1m[35mTrue[39m(B[m, alpha=0.7, label=sv_name)[8;12H[42m[1K[9d(B[0;1m[31m # Add means as vertical lines[10;13H[39m(B[max.axvline(df_bg_local[stat_col].mean(), color=(B[0;1m[32m'gray'[39m(B[m, linestyle=(B[0;1m[32m'--'[39m(B[m, linewidth=1, label=(B[0;1m[32m'BG Mean'[39m(B[m)[11;13Hax.axvline(sv_loci[stat_col].mean(), color=config[(B[0;1m[32m'color'[39m(B[m], linestyle=(B[0;1m[32m'-'[39m(B[m, linewidth=1, label=(B[0;1m[32m'SV Mean'[39m(B[m)[12;12H[42m[1K[13;13H[49m(B[max.set_title(f(B[0;1m[32m"{config['label']} Distribution: {sv_name}"[39m(B[m)[14;13Hax.set_xlabel(config[(B[0;1m[32m'label'[39m(B[m])[15;12H[42m[1K[16d(B[0;1m[31m # Special handling for PI Ratio plot[17;13H[36mif[39m(B[m stat_col == (B[0;1m[32m'PI_RATIO'[39m(B[m:[18;17Hax.axvline(1.0, color=(B[0;1m[32m'black'[39m(B[m, linestyle=(B[0;1m[32m':'[39m(B[m, linewidth=1, label=(B[0;1m[32m'Neutral Ratio (1.0)'[39m(B[m)[19;17Hax.set_xlim(0, local_standard_x_limit)[20;17Hax.set_title(f(B[0;1m[32m"{config['label']} Distribution (Cap at Local BG 95th: {local_standard_x_limit:.2f})"[39m(B[m)[21;12H[42m[1K[22;13H[49m(B[max.legend(loc=(B[0;1m[32m'upper right'[39m(B[m)[24;9Hplt.suptitle(f(B[0;1m[32m'Comparative Profile of {sv_name} ({sv_row["SIZE"]/1000:.1f} kb) vs. Local Chromosome Background'[39m(B[m, fontsize=16)[25;9Hplt.tight_layout(rect=[0, 0.03, 1, 0.95])[26;8H[42m[1K[27d	[49m(B[mfilename = os.path.join(PLOTS_OUTPUT_DIR, f(B[0;1m[32m"{sv_row['SV_ID']}_{sv_row['TYPE']}_Chr{chr_int}_local_analysis.png"[39m(B[m)[28;9Hplt.savefig(filename, dpi=300)[29;9Hplt.close(fig)[42m [30;9H(B[0;1m[36mprint[39m(B[m(f(B[0;1m[32m"Generated {filename}"[39m(B[m)[32;5H(B[0;1m[36mprint[39m(B[m((B[0;1m[32m"\n--- Analysis Complete ---"[39m(B[m)[33d[42m    [34d(B[0;1m[36mif[39m(B[m __name__ == (B[0;1m[32m"__main__"[39m(B[m:[35;5Hmain()[43d[K[35;11H[?12l[?25h[?25l[43d(B[0;7mSave modified buffer?  (Answering "No" will DISCARD changes.)                                                                                                                           [44;1H Y(B[m Yes[K[45d(B[0;7m N(B[m No  [45;18H(B[0;7mC(B[m Cancel[K[43;63H[?12l[?25h[?25l[44d(B[0;7m^G(B[m Get Help[44;47H(B[0;7mM-D(B[m DOS Format[44;93H(B[0;7mM-A(B[m Append[44;139H(B[0;7mM-B(B[m Backup File[45d(B[0;7m^C(B[m Cancel	         [45;47H(B[0;7mM-M(B[m Mac Format[45;93H(B[0;7mM-P(B[m Prepend[45;139H(B[0;7m^T(B[m To Files[43d(B[0;7mFile Name to Write: sv_local_density_plots_30_all.py         (B[m[43;53H[?12l[?25h[?25l[K[1;183H[1;175H(B[0;7m        (B[m[43;83H(B[0;7m[ Wrote 393 lines ](B[m[J[45;184H[?12l[?25h[45;1H[?1049l[23;0;0t[?1l>]0;aescudero@login3:~/wgg/fst_results(plot_env) [aescudero@nodo4143 fst_results]$ nano sv_local_density_plots_30_all.py[C[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.pyp sv_local_density_plots_30_all.pyy sv_local_density_plots_30_all.pyt sv_local_density_plots_30_all.pyh sv_local_density_plots_30_all.pyo sv_local_density_plots_30_all.pyn sv_local_density_plots_30_all.py
Loading and processing windowed statistics...
Total 1000bp windows analyzed: 328966

Loading and processing SyRI data...
Analyzing enrichment within the top 30 largest SVs (>10000 bp):
SV_ID  TYPE  CHR_int    START      END    SIZE
  SV1   INV       14  3040103  5331878 2291775
  SV2   INV       18  1681715  2543617  861902
  SV3   INV       20  4207035  4474484  267449
  SV4   INV        3  6538064  6773584  235520
  SV5 INVTR       11   572229   807176  234947
  SV6 INVTR       11   423763   645432  221669
  SV7   DUP        2 15863192 16067646  204454
  SV8 INVTR       11   492162   686228  194066
  SV9 INVTR        2 16163408 16349879  186471
 SV10 INVDP        3   544317   723684  179367
 SV11 INVTR        4   959624  1122892  163268
 SV12 TRANS        3   235306   373115  137809
 SV13 INVDP        2 16098557 16233821  135264
 SV14 INVDP        2 16125649 16260866  135217
 SV15   DUP        2 15962711 16092230  129519
 SV16   INV       32   890484  1017805  127321
 SV17   INV       22  8850975  8974282  123307
 SV18   DUP        2 16085119 16205029  119910
 SV19   DUP        2 16303904 16410644  106740
 SV20   INV       25  8811273  8913117  101844
 SV21 INVDP        2 16027599 16111681   84082
 SV22 INVTR       27  8928753  9012287   83534
 SV23   INV       28  4452397  4532837   80440
 SV24   INV       14 10730438 10807332   76894
 SV25   INV       20 10448895 10519691   70796
 SV26   DUP        2 16412801 16476160   63359
 SV27 INVDP       18  5445457  5508555   63098
 SV28   INV       21  8212305  8272819   60514
 SV29   DUP        2 16413586 16473905   60319
 SV30 INVTR       31  7660902  7717996   57094

================================================================================
--- Individual Statistical Tests (Mann-Whitney U) for Top 30 SVs vs. Local Chromosome Background ---
================================================================================
Skipping SV13 (Chr2, INVDP) due to insufficient data (SV: 4, BG: 14134 windows).
Skipping SV18 (Chr2, DUP) due to insufficient data (SV: 2, BG: 14136 windows).
Skipping SV19 (Chr2, DUP) due to insufficient data (SV: 3, BG: 14135 windows).
Skipping SV21 (Chr2, INVDP) due to insufficient data (SV: 2, BG: 14135 windows).
SV_ID  TYPE  CHR   SIZE_kb  SV_Windows  BG_Windows  FST_Mean   FST_P         FST_Sig  PI_POP1_Mean PI_POP1_P     PI_POP1_Sig  PI_POP2_Mean PI_POP2_P     PI_POP2_Sig  PI_RATIO_Mean PI_RATIO_P    PI_RATIO_Sig
  SV1   INV   14 2291.7750        1956        7730    0.5597 < 0.001     Significant        0.0009   < 0.001     Significant        0.0018     0.359 Not Significant         0.9430    < 0.001     Significant
  SV2   INV   18  861.9020         723        8224    0.4194 < 0.001     Significant        0.0014   < 0.001     Significant        0.0018     0.066 Not Significant         1.2099    < 0.001     Significant
  SV3   INV   20  267.4490         225        9019    0.6299 < 0.001     Significant        0.0009   < 0.001     Significant        0.0017     0.007     Significant         1.0490    < 0.001     Significant
  SV4   INV    3  235.5200         202       12442    0.4673 < 0.001     Significant        0.0009   < 0.001     Significant        0.0019     0.035     Significant         1.0411    < 0.001     Significant
  SV5 INVTR   11  234.9470          37       10427    0.2021   1.000 Not Significant        0.0012     0.158 Not Significant        0.0008   < 0.001     Significant         2.2985    < 0.001     Significant
  SV6 INVTR   11  221.6690          26       10439    0.2084   1.000 Not Significant        0.0007     0.002     Significant        0.0007   < 0.001     Significant         3.8127      0.074 Not Significant
  SV7   DUP    2  204.4540          33       14104    0.1960   1.000 Not Significant        0.0005   < 0.001     Significant        0.0008   < 0.001     Significant         1.2756      0.571 Not Significant
  SV8 INVTR   11  194.0660          25       10440    0.2070   1.000 Not Significant        0.0009     0.033     Significant        0.0008   < 0.001     Significant         1.5665      0.045     Significant
  SV9 INVTR    2  186.4710          11       14127    0.2161   1.000 Not Significant        0.0008     0.144 Not Significant        0.0005   < 0.001     Significant         1.8413      0.040     Significant
 SV10 INVDP    3  179.3670          20       12625    0.2073   1.000 Not Significant        0.0009     0.011     Significant        0.0007   < 0.001     Significant         1.3185      0.444 Not Significant
 SV11 INVTR    4  163.2680         155       12082    0.2376   1.000 Not Significant        0.0027   < 0.001     Significant        0.0021   < 0.001     Significant         1.4302    < 0.001     Significant
 SV12 TRANS    3  137.8090          32       12614    0.2096   1.000 Not Significant        0.0010     0.017     Significant        0.0010   < 0.001     Significant         6.6754      0.003     Significant
 SV14 INVDP    2  135.2170           5       14132    0.2847   0.957 Not Significant        0.0010     0.499 Not Significant        0.0006     0.020     Significant         1.5893      0.257 Not Significant
 SV15   DUP    2  129.5190           5       14133    0.1832   0.997 Not Significant        0.0003     0.016     Significant        0.0005     0.012     Significant         1.8973      0.932 Not Significant
 SV16   INV   32  127.3210         117        6468    0.5865 < 0.001     Significant        0.0008   < 0.001     Significant        0.0017     0.315 Not Significant         1.6960    < 0.001     Significant
 SV17   INV   22  123.3070         120        8879    0.3312   0.986 Not Significant        0.0016     0.034     Significant        0.0024     0.004     Significant         0.7081      0.714 Not Significant
 SV20   INV   25  101.8440          97        8274    0.1386   1.000 Not Significant        0.0026   < 0.001     Significant        0.0025     0.037     Significant         1.2462      0.005     Significant
 SV22 INVTR   27   83.5340          55        8022    0.2512   1.000 Not Significant        0.0019     0.152 Not Significant        0.0015     0.003     Significant         4.5156    < 0.001     Significant
 SV23   INV   28   80.4400          75        7401    0.4592   0.011     Significant        0.0014     0.405 Not Significant        0.0018     0.374 Not Significant         0.9321      0.523 Not Significant
 SV24   INV   14   76.8940          60        9626    0.3275   1.000 Not Significant        0.0023   < 0.001     Significant        0.0014     0.029     Significant         1.9220    < 0.001     Significant
 SV25   INV   20   70.7960          68        9178    0.2636   1.000 Not Significant        0.0020   < 0.001     Significant        0.0016     0.452 Not Significant         2.4415    < 0.001     Significant
 SV26   DUP    2   63.3590           6       14132    0.2441   0.992 Not Significant        0.0010     0.767 Not Significant        0.0002   < 0.001     Significant        17.8856    < 0.001     Significant
 SV27 INVDP   18   63.0980          54        8894    0.1665   1.000 Not Significant        0.0021   < 0.001     Significant        0.0016     0.364 Not Significant         1.6029    < 0.001     Significant
 SV28   INV   21   60.5140          55        8927    0.2804   1.000 Not Significant        0.0014     0.274 Not Significant        0.0016     0.113 Not Significant         4.0277      0.012     Significant
 SV29   DUP    2   60.3190           6       14132    0.2441   0.992 Not Significant        0.0010     0.767 Not Significant        0.0002   < 0.001     Significant        17.8856    < 0.001     Significant
 SV30 INVTR   31   57.0940          22        6712    0.2100   0.914 Not Significant        0.0011   < 0.001     Significant        0.0008   < 0.001     Significant         0.9726      0.419 Not Significant

Note on Mann-Whitney U Test Alternatives:
- FST: SV > Local Background (Alternative: 'greater')
- PI metrics and Ratio: SV $\ne$ Local Background (Alternative: 'two-sided')

================================================================================
--- 3B. Global Test: All Large SVs vs. All Non-SV Regions (>$10	ext{kb}$ SVs) ---
================================================================================
Traceback (most recent call last):
  File "/lustre/home/aescudero/wgg/fst_results/sv_local_density_plots_30_all.py", line 393, in <module>
    main()
  File "/lustre/home/aescudero/wgg/fst_results/sv_local_density_plots_30_all.py", line 282, in main
    print(f"Total SV Windows (>$10\text{kb}$ SVs): {len(df_all_sv):,}")
NameError: name 'kb' is not defined
]0;aescudero@login3:~/wgg/fst_results(plot_env) [aescudero@nodo4143 fst_results]$ python sv_local_density_plots_30_all.py[C[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.pyr sv_local_density_plots_30_all.pym sv_local_density_plots_30_all.py
]0;aescudero@login3:~/wgg/fst_results(plot_env) [aescudero@nodo4143 fst_results]$ rm sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.pyn sv_local_density_plots_30_all.pya sv_local_density_plots_30_all.pyn sv_local_density_plots_30_all.pyo sv_local_density_plots_30_all.py
[?1049h[22;0;0t[1;45r(B[m[4l[?7h[39;49m[?1h=[?1h=[?1h=[?25l[39;49m(B[m[H[2J[43;87H(B[0;7m[ New File ](B[m[H(B[0;7m  GNU nano 2.9.8                                                               sv_local_density_plots_30_all.py                                                                         [1;183H(B[m[44d(B[0;7m^G(B[m Get Help     (B[0;7m^O(B[m Write Out    (B[0;7m^W(B[m Where Is     (B[0;7m^K(B[m Cut Text     (B[0;7m^J(B[m Justify	(B[0;7m^C(B[m Cur Pos	(B[0;7mM-U(B[m Undo[44;113H(B[0;7mM-A(B[m Mark Text   (B[0;7mM-](B[m To Bracket  (B[0;7mM-▲(B[m Previous    (B[0;7m^B(B[m Back[45d(B[0;7m^X(B[m Exit[45;17H(B[0;7m^R(B[m Read File    (B[0;7m^\(B[m Replace	(B[0;7m^U(B[m Uncut Text   (B[0;7m^T(B[m To Linter    (B[0;7m^_(B[m Go To Line   (B[0;7mM-E(B[m Redo[45;113H(B[0;7mM-6(B[m Copy Text   (B[0;7mM-W(B[m WhereIs Next(B[0;7mM-▼(B[m Next[45;161H(B[0;7m^F(B[m Forward[43d[3d[39;49m(B[m[?12l[?25h[?25l[1;175H(B[0;7mModified(B[m[3;12H[42m[1K[4d(B[0;1m[31m # Plot Local Background Distribution[5;13H[39m(B[msns.kdeplot(df_bg_local[stat_col].dropna(), ax=ax, color=(B[0;1m[32m'gray'[39m(B[m, fill=(B[0;1m[35mTrue[39m(B[m, alpha=0.5, label=(B[0;1m[32m'Local Background'[39m(B[m)[6;12H(B[0;1m[31m # Plot SV Distribution[7;13H[39m(B[msns.kdeplot(sv_loci[stat_col].dropna(), ax=ax, color=config[(B[0;1m[32m'color'[39m(B[m], fill=(B[0;1m[35mTrue[39m(B[m, alpha=0.7, label=sv_name)[8;12H[42m[1K[9d(B[0;1m[31m # Add means as vertical lines[10;13H[39m(B[max.axvline(df_bg_local[stat_col].mean(), color=(B[0;1m[32m'gray'[39m(B[m, linestyle=(B[0;1m[32m'--'[39m(B[m, linewidth=1, label=(B[0;1m[32m'BG Mean'[39m(B[m)[11;13Hax.axvline(sv_loci[stat_col].mean(), color=config[(B[0;1m[32m'color'[39m(B[m], linestyle=(B[0;1m[32m'-'[39m(B[m, linewidth=1, label=(B[0;1m[32m'SV Mean'[39m(B[m)[12;12H[42m[1K[13;13H[49m(B[max.set_title(f(B[0;1m[32m"{config['label']} Distribution: {sv_name}"[39m(B[m)[14;13Hax.set_xlabel(config[(B[0;1m[32m'label'[39m(B[m])[15;12H[42m[1K[16d(B[0;1m[31m # Special handling for PI Ratio plot[17;13H[36mif[39m(B[m stat_col == (B[0;1m[32m'PI_RATIO'[39m(B[m:[18;17Hax.axvline(1.0, color=(B[0;1m[32m'black'[39m(B[m, linestyle=(B[0;1m[32m':'[39m(B[m, linewidth=1, label=(B[0;1m[32m'Neutral Ratio (1.0)'[39m(B[m)[19;17Hax.set_xlim(0, local_standard_x_limit)[20;17Hax.set_title(f(B[0;1m[32m"{config['label']} Distribution (Cap at Local BG 95th: {local_standard_x_limit:.2f})"[39m(B[m)[21;12H[42m[1K[22;13H[49m(B[max.legend(loc=(B[0;1m[32m'upper right'[39m(B[m)[24;9Hplt.suptitle(f(B[0;1m[32m'Comparative Profile of {sv_name} ({sv_row["SIZE"]/1000:.1f} kb) vs. Local Chromosome Background'[39m(B[m, fontsize=16)[25;9Hplt.tight_layout(rect=[0, 0.03, 1, 0.95])[26;8H[42m[1K[27d	[49m(B[mfilename = os.path.join(PLOTS_OUTPUT_DIR, f(B[0;1m[32m"{sv_row['SV_ID']}_{sv_row['TYPE']}_Chr{chr_int}_local_analysis.png"[39m(B[m)[28;9Hplt.savefig(filename, dpi=300)[29;9Hplt.close(fig)[42m [30;9H(B[0;1m[36mprint[39m(B[m(f(B[0;1m[32m"Generated {filename}"[39m(B[m)[32;5H(B[0;1m[36mprint[39m(B[m((B[0;1m[32m"\n--- Analysis Complete ---"[39m(B[m)[33d[42m    [34d(B[0;1m[36mif[39m(B[m __name__ == (B[0;1m[32m"__main__"[39m(B[m:[35;5Hmain()[43d[K[35;11H[?12l[?25h[?25l[43d(B[0;7mSave modified buffer?  (Answering "No" will DISCARD changes.)                                                                                                                           [44;1H Y(B[m Yes[K[45d(B[0;7m N(B[m No  [45;18H(B[0;7mC(B[m Cancel[K[43;63H[?12l[?25h[?25l[44d(B[0;7m^G(B[m Get Help[44;47H(B[0;7mM-D(B[m DOS Format[44;93H(B[0;7mM-A(B[m Append[44;139H(B[0;7mM-B(B[m Backup File[45d(B[0;7m^C(B[m Cancel	         [45;47H(B[0;7mM-M(B[m Mac Format[45;93H(B[0;7mM-P(B[m Prepend[45;139H(B[0;7m^T(B[m To Files[43d(B[0;7mFile Name to Write: sv_local_density_plots_30_all.py         (B[m[43;53H[?12l[?25h[?25l[K[1;183H[1;175H(B[0;7m        (B[m[43;83H(B[0;7m[ Wrote 393 lines ](B[m[J[45;184H[?12l[?25h[45;1H[?1049l[23;0;0t[?1l>]0;aescudero@login3:~/wgg/fst_results(plot_env) [aescudero@nodo4143 fst_results]$ nano sv_local_density_plots_30_all.py[2Prm[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[4@python[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
Loading and processing windowed statistics...
Total 1000bp windows analyzed: 328966

Loading and processing SyRI data...
Analyzing enrichment within the top 30 largest SVs (>10000 bp):
SV_ID  TYPE  CHR_int    START      END    SIZE
  SV1   INV       14  3040103  5331878 2291775
  SV2   INV       18  1681715  2543617  861902
  SV3   INV       20  4207035  4474484  267449
  SV4   INV        3  6538064  6773584  235520
  SV5 INVTR       11   572229   807176  234947
  SV6 INVTR       11   423763   645432  221669
  SV7   DUP        2 15863192 16067646  204454
  SV8 INVTR       11   492162   686228  194066
  SV9 INVTR        2 16163408 16349879  186471
 SV10 INVDP        3   544317   723684  179367
 SV11 INVTR        4   959624  1122892  163268
 SV12 TRANS        3   235306   373115  137809
 SV13 INVDP        2 16098557 16233821  135264
 SV14 INVDP        2 16125649 16260866  135217
 SV15   DUP        2 15962711 16092230  129519
 SV16   INV       32   890484  1017805  127321
 SV17   INV       22  8850975  8974282  123307
 SV18   DUP        2 16085119 16205029  119910
 SV19   DUP        2 16303904 16410644  106740
 SV20   INV       25  8811273  8913117  101844
 SV21 INVDP        2 16027599 16111681   84082
 SV22 INVTR       27  8928753  9012287   83534
 SV23   INV       28  4452397  4532837   80440
 SV24   INV       14 10730438 10807332   76894
 SV25   INV       20 10448895 10519691   70796
 SV26   DUP        2 16412801 16476160   63359
 SV27 INVDP       18  5445457  5508555   63098
 SV28   INV       21  8212305  8272819   60514
 SV29   DUP        2 16413586 16473905   60319
 SV30 INVTR       31  7660902  7717996   57094

================================================================================
--- Individual Statistical Tests (Mann-Whitney U) for Top 30 SVs vs. Local Chromosome Background ---
================================================================================
Skipping SV13 (Chr2, INVDP) due to insufficient data (SV: 4, BG: 14134 windows).
Skipping SV18 (Chr2, DUP) due to insufficient data (SV: 2, BG: 14136 windows).
Skipping SV19 (Chr2, DUP) due to insufficient data (SV: 3, BG: 14135 windows).
Skipping SV21 (Chr2, INVDP) due to insufficient data (SV: 2, BG: 14135 windows).
SV_ID  TYPE  CHR   SIZE_kb  SV_Windows  BG_Windows  FST_Mean   FST_P         FST_Sig  PI_POP1_Mean PI_POP1_P     PI_POP1_Sig  PI_POP2_Mean PI_POP2_P     PI_POP2_Sig  PI_RATIO_Mean PI_RATIO_P    PI_RATIO_Sig
  SV1   INV   14 2291.7750        1956        7730    0.5597 < 0.001     Significant        0.0009   < 0.001     Significant        0.0018     0.359 Not Significant         0.9430    < 0.001     Significant
  SV2   INV   18  861.9020         723        8224    0.4194 < 0.001     Significant        0.0014   < 0.001     Significant        0.0018     0.066 Not Significant         1.2099    < 0.001     Significant
  SV3   INV   20  267.4490         225        9019    0.6299 < 0.001     Significant        0.0009   < 0.001     Significant        0.0017     0.007     Significant         1.0490    < 0.001     Significant
  SV4   INV    3  235.5200         202       12442    0.4673 < 0.001     Significant        0.0009   < 0.001     Significant        0.0019     0.035     Significant         1.0411    < 0.001     Significant
  SV5 INVTR   11  234.9470          37       10427    0.2021   1.000 Not Significant        0.0012     0.158 Not Significant        0.0008   < 0.001     Significant         2.2985    < 0.001     Significant
  SV6 INVTR   11  221.6690          26       10439    0.2084   1.000 Not Significant        0.0007     0.002     Significant        0.0007   < 0.001     Significant         3.8127      0.074 Not Significant
  SV7   DUP    2  204.4540          33       14104    0.1960   1.000 Not Significant        0.0005   < 0.001     Significant        0.0008   < 0.001     Significant         1.2756      0.571 Not Significant
  SV8 INVTR   11  194.0660          25       10440    0.2070   1.000 Not Significant        0.0009     0.033     Significant        0.0008   < 0.001     Significant         1.5665      0.045     Significant
  SV9 INVTR    2  186.4710          11       14127    0.2161   1.000 Not Significant        0.0008     0.144 Not Significant        0.0005   < 0.001     Significant         1.8413      0.040     Significant
 SV10 INVDP    3  179.3670          20       12625    0.2073   1.000 Not Significant        0.0009     0.011     Significant        0.0007   < 0.001     Significant         1.3185      0.444 Not Significant
 SV11 INVTR    4  163.2680         155       12082    0.2376   1.000 Not Significant        0.0027   < 0.001     Significant        0.0021   < 0.001     Significant         1.4302    < 0.001     Significant
 SV12 TRANS    3  137.8090          32       12614    0.2096   1.000 Not Significant        0.0010     0.017     Significant        0.0010   < 0.001     Significant         6.6754      0.003     Significant
 SV14 INVDP    2  135.2170           5       14132    0.2847   0.957 Not Significant        0.0010     0.499 Not Significant        0.0006     0.020     Significant         1.5893      0.257 Not Significant
 SV15   DUP    2  129.5190           5       14133    0.1832   0.997 Not Significant        0.0003     0.016     Significant        0.0005     0.012     Significant         1.8973      0.932 Not Significant
 SV16   INV   32  127.3210         117        6468    0.5865 < 0.001     Significant        0.0008   < 0.001     Significant        0.0017     0.315 Not Significant         1.6960    < 0.001     Significant
 SV17   INV   22  123.3070         120        8879    0.3312   0.986 Not Significant        0.0016     0.034     Significant        0.0024     0.004     Significant         0.7081      0.714 Not Significant
 SV20   INV   25  101.8440          97        8274    0.1386   1.000 Not Significant        0.0026   < 0.001     Significant        0.0025     0.037     Significant         1.2462      0.005     Significant
 SV22 INVTR   27   83.5340          55        8022    0.2512   1.000 Not Significant        0.0019     0.152 Not Significant        0.0015     0.003     Significant         4.5156    < 0.001     Significant
 SV23   INV   28   80.4400          75        7401    0.4592   0.011     Significant        0.0014     0.405 Not Significant        0.0018     0.374 Not Significant         0.9321      0.523 Not Significant
 SV24   INV   14   76.8940          60        9626    0.3275   1.000 Not Significant        0.0023   < 0.001     Significant        0.0014     0.029     Significant         1.9220    < 0.001     Significant
 SV25   INV   20   70.7960          68        9178    0.2636   1.000 Not Significant        0.0020   < 0.001     Significant        0.0016     0.452 Not Significant         2.4415    < 0.001     Significant
 SV26   DUP    2   63.3590           6       14132    0.2441   0.992 Not Significant        0.0010     0.767 Not Significant        0.0002   < 0.001     Significant        17.8856    < 0.001     Significant
 SV27 INVDP   18   63.0980          54        8894    0.1665   1.000 Not Significant        0.0021   < 0.001     Significant        0.0016     0.364 Not Significant         1.6029    < 0.001     Significant
 SV28   INV   21   60.5140          55        8927    0.2804   1.000 Not Significant        0.0014     0.274 Not Significant        0.0016     0.113 Not Significant         4.0277      0.012     Significant
 SV29   DUP    2   60.3190           6       14132    0.2441   0.992 Not Significant        0.0010     0.767 Not Significant        0.0002   < 0.001     Significant        17.8856    < 0.001     Significant
 SV30 INVTR   31   57.0940          22        6712    0.2100   0.914 Not Significant        0.0011   < 0.001     Significant        0.0008   < 0.001     Significant         0.9726      0.419 Not Significant

Note on Mann-Whitney U Test Alternatives:
- FST: SV > Local Background (Alternative: 'greater')
- PI metrics and Ratio: SV $\ne$ Local Background (Alternative: 'two-sided')

================================================================================
--- 3B. Global Test: All Large SVs vs. All Non-SV Regions (>10 kbp SVs) ---
================================================================================
Total SV Windows (>10 kbp SVs): 4,878
Total Non-SV Windows: 324,088

Global Statistical Comparison (Mann-Whitney U Test):
| Metric                                      | Mean SV | Mean Non-SV | P-value | Significance |
|---------------------------------------------|---------|-------------|---------|--------------|
| FST                                         | 0.4505  | 0.4057      | < 0.001 | Significant  |
| $\pi_{\text{Pop1}}$                         | 0.0013  | 0.0015      | < 0.001 | Significant  |
| $\pi_{\text{Pop2}}$                         | 0.0019  | 0.0019      | 0.036   | Significant  |
| $\pi_{\text{Pop1}}/\pi_{\text{Pop2}}$ Ratio | 1.2875  | 1.4587      | < 0.001 | Significant  |

Note: FST test alternative hypothesis is SV > Non-SV.

================================================================================
--- Generating 4-Panel Density Plots in 'sv_local_density_plots/' ---
================================================================================
Generated sv_local_density_plots/SV1_INV_Chr14_local_analysis.png
Generated sv_local_density_plots/SV2_INV_Chr18_local_analysis.png
Generated sv_local_density_plots/SV3_INV_Chr20_local_analysis.png
Generated sv_local_density_plots/SV4_INV_Chr3_local_analysis.png
Generated sv_local_density_plots/SV5_INVTR_Chr11_local_analysis.png
Generated sv_local_density_plots/SV6_INVTR_Chr11_local_analysis.png
Generated sv_local_density_plots/SV7_DUP_Chr2_local_analysis.png
Generated sv_local_density_plots/SV8_INVTR_Chr11_local_analysis.png
Generated sv_local_density_plots/SV9_INVTR_Chr2_local_analysis.png
Generated sv_local_density_plots/SV10_INVDP_Chr3_local_analysis.png
Generated sv_local_density_plots/SV11_INVTR_Chr4_local_analysis.png
Generated sv_local_density_plots/SV12_TRANS_Chr3_local_analysis.png
Generated sv_local_density_plots/SV14_INVDP_Chr2_local_analysis.png
Generated sv_local_density_plots/SV15_DUP_Chr2_local_analysis.png
Generated sv_local_density_plots/SV16_INV_Chr32_local_analysis.png
Generated sv_local_density_plots/SV17_INV_Chr22_local_analysis.png
Generated sv_local_density_plots/SV20_INV_Chr25_local_analysis.png
Generated sv_local_density_plots/SV22_INVTR_Chr27_local_analysis.png
Generated sv_local_density_plots/SV23_INV_Chr28_local_analysis.png
Generated sv_local_density_plots/SV24_INV_Chr14_local_analysis.png
Generated sv_local_density_plots/SV25_INV_Chr20_local_analysis.png
Generated sv_local_density_plots/SV26_DUP_Chr2_local_analysis.png
Generated sv_local_density_plots/SV27_INVDP_Chr18_local_analysis.png
Generated sv_local_density_plots/SV28_INV_Chr21_local_analysis.png
Generated sv_local_density_plots/SV29_DUP_Chr2_local_analysis.png
Generated sv_local_density_plots/SV30_INVTR_Chr31_local_analysis.png

--- Analysis Complete ---
]0;aescudero@login3:~/wgg/fst_results(plot_env) [aescudero@nodo4143 fst_results]$ python sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.py[1P sv_local_density_plots_30_all.pyc sv_local_density_plots_30_all.pya sv_local_density_plots_30_all.pyt sv_local_density_plots_30_all.py
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
]0;aescudero@login3:~/wgg/fst_results(plot_env) [aescudero@nodo4143 fst_results]$ cd ..
]0;aescudero@login3:~/wgg(plot_env) [aescudero@nodo4143 wgg]$ cd RDA/
]0;aescudero@login3:~/wgg/RDA(plot_env) [aescudero@nodo4143 RDA]$ ls
admixture_final_ld_pruned_cleaned.bed                       inversion_genotypes.csv                     RDAanalysis_2poplabeling_biopca_final.R
admixture_final_ld_pruned_cleaned.bim                       inversion_heterozygosity.csv                RDAanalysis_2poplabeling_biopca_prueba.R
admixture_final_ld_pruned_cleaned.fam                       inversion_heterozygosity_per_inversion.csv  RDAanalysis_2poplabeling_biopca.R
admixture_final_ld_pruned_cleaned.log                       inversion_hybridity_and_genotype.csv        RDAanalysis_chr14.R
admixture_final_ld_pruned_cleaned.nosex                     inverted_genotype_analysis.R                RDAanalysis_chr18.R
admixture_final_ld_pruned_filtered.bcf                      locations.txt                               RDAanalysis_chr20.R
admixture_final_ld_pruned_filtered.bed                      multipanel_genomic_plot_chr14.pdf           RDAanalysis_chr28.R
admixture_final_ld_pruned_filtered.bim                      [0m[38;5;13mmultipanel_genomic_plot_chr14.png[0m           RDAanalysis_chr32.R
admixture_final_ld_pruned_filtered_clean.bcf                multipanel_genomic_plot_chr18.pdf           RDAanalysis_chr3.R
admixture_final_ld_pruned_filtered.fam                      [38;5;13mmultipanel_genomic_plot_chr18.png[0m           RDA_plot.pdf
admixture_final_ld_pruned_filtered_for_RDA.log              multipanel_genomic_plot_chr20.pdf           RDA_plot_pop.pdf
admixture_final_ld_pruned_filtered_for_RDA.nosex            [38;5;13mmultipanel_genomic_plot_chr20.png[0m           rda_snp_scores_chr14.csv
admixture_final_ld_pruned_filtered_for_RDA.raw              multipanel_genomic_plot_chr28.pdf           rda_snp_scores_chr18.csv
admixture_final_ld_pruned_filtered.fst                      [38;5;13mmultipanel_genomic_plot_chr28.png[0m           rda_snp_scores_chr20.csv
admixture_final_ld_pruned_filtered.log                      multipanel_genomic_plot_chr32.pdf           rda_snp_scores_chr28.csv
admixture_final_ld_pruned_filtered.nosex                    [38;5;13mmultipanel_genomic_plot_chr32.png[0m           rda_snp_scores_chr32.csv
admixture_final_ld_pruned_filtered_pop1_vs_pop2.fst.sorted  multipanel_genomic_plot_chr3.pdf            rda_snp_scores_chr3.csv
admixture_final_ld_pruned_filtered_snps_to_keep.txt         [38;5;13mmultipanel_genomic_plot_chr3.png[0m            rda_snp_scores.csv
admixture_final_ld_pruned_filtered.vcf                      multipanel_genomic_plot.pdf                 Rplots.pdf
bioclim_for_rda.csv                                         [38;5;13mmultipanel_genomic_plot.png[0m                 scriptdowloadingworldclim.R
[38;5;13mcomprehensive_genomic_plot.png[0m                              PCA_plot_bioclim.pdf                        selected_snps_for_plotting_chr14.csv
filtered_genetic_data.tsv                                   pdf_merger_for_genomics_plots.py            selected_snps_for_plotting_chr18.csv
filtered_inversion_genotypes.csv                            plot_fst_RDA_chr14.py                       selected_snps_for_plotting_chr20.csv
filter_genetic_data_before_plotting_RAD.R                   plot_fst_RDA_chr18.py                       selected_snps_for_plotting_chr28.csv
final_combined_genomic_figure.pdf                           plot_fst_RDA_chr20.py                       selected_snps_for_plotting_chr32.csv
final_df_for_analysis.csv                                   plot_fst_RDA_chr28.py                       selected_snps_for_plotting_chr3.csv
genetic_data_final.csv                                      plot_fst_RDA_chr32.py                       selected_snps_for_plotting.csv
genetic_data_for_rda.csv                                    plot_fst_RDA_chr3.py                        selectingindependentvariables_prueba.R
genotype_differences.csv                                    plot_fst_RDA.py                             selectingindependentvariables.R
individual_bioclim_data.csv                                 population_list.txt                         [38;5;33mwc2.1_country[0m
]0;aescudero@login3:~/wgg/RDA(plot_env) [aescudero@nodo4143 RDA]$ ls -la -t
total 4111016
drwxrwxr-x  3 aescudero aescudero      12288 Nov 19 16:30 [0m[38;5;33m.[0m
-rw-rw-r--  1 aescudero aescudero     922000 Nov 19 16:30 final_combined_genomic_figure.pdf
-rw-rw-r--  1 aescudero aescudero       1891 Nov 19 16:25 pdf_merger_for_genomics_plots.py
-rw-rw-r--  1 aescudero aescudero     155244 Nov 19 14:49 multipanel_genomic_plot_chr3.pdf
-rw-rw-r--  1 aescudero aescudero    1016889 Nov 19 14:49 [38;5;13mmultipanel_genomic_plot_chr3.png[0m
-rw-rw-r--  1 aescudero aescudero      12541 Nov 19 14:49 plot_fst_RDA_chr3.py
-rw-rw-r--  1 aescudero aescudero     153575 Nov 19 14:45 multipanel_genomic_plot_chr28.pdf
-rw-rw-r--  1 aescudero aescudero    1026803 Nov 19 14:45 [38;5;13mmultipanel_genomic_plot_chr28.png[0m
-rw-rw-r--  1 aescudero aescudero     152340 Nov 19 14:44 multipanel_genomic_plot_chr32.pdf
-rw-rw-r--  1 aescudero aescudero    1024520 Nov 19 14:44 [38;5;13mmultipanel_genomic_plot_chr32.png[0m
-rw-rw-r--  1 aescudero aescudero     154920 Nov 19 14:42 multipanel_genomic_plot_chr20.pdf
-rw-rw-r--  1 aescudero aescudero    1055841 Nov 19 14:42 [38;5;13mmultipanel_genomic_plot_chr20.png[0m
-rw-rw-r--  1 aescudero aescudero     153900 Nov 19 14:42 multipanel_genomic_plot_chr18.pdf
-rw-rw-r--  1 aescudero aescudero    1093090 Nov 19 14:42 [38;5;13mmultipanel_genomic_plot_chr18.png[0m
-rw-rw-r--  1 aescudero aescudero     153952 Nov 19 14:42 multipanel_genomic_plot_chr14.pdf
-rw-rw-r--  1 aescudero aescudero    1146467 Nov 19 14:42 [38;5;13mmultipanel_genomic_plot_chr14.png[0m
-rw-rw-r--  1 aescudero aescudero      12546 Nov 19 14:40 plot_fst_RDA_chr28.py
-rw-rw-r--  1 aescudero aescudero      12546 Nov 19 14:39 plot_fst_RDA_chr32.py
-rw-rw-r--  1 aescudero aescudero      12546 Nov 19 12:41 plot_fst_RDA_chr20.py
-rw-rw-r--  1 aescudero aescudero      12546 Nov 19 12:32 plot_fst_RDA_chr18.py
-rw-rw-r--  1 aescudero aescudero      12546 Nov 19 12:25 plot_fst_RDA_chr14.py
-rw-rw-r--  1 aescudero aescudero      89756 Nov 19 11:09 multipanel_genomic_plot.pdf
-rw-rw-r--  1 aescudero aescudero     696002 Nov 19 11:09 [38;5;13mmultipanel_genomic_plot.png[0m
-rw-rw-r--  1 aescudero aescudero      13482 Nov 19 11:09 plot_fst_RDA.py
-rw-rw-r--  1 aescudero aescudero       6744 Nov 19 11:00 PCA_plot_bioclim.pdf
-rw-rw-r--  1 aescudero aescudero      16019 Nov 19 11:00 RDA_plot_pop.pdf
-rw-rw-r--  1 aescudero aescudero      20898 Nov 19 10:59 final_df_for_analysis.csv
-rw-rw-r--  1 aescudero aescudero      20509 Nov 19 10:59 Rplots.pdf
-rw-rw-r--  1 aescudero aescudero       2984 Nov 19 10:56 selectingindependentvariables_prueba.R
-rw-rw-r--  1 aescudero aescudero       9881 Nov 19 10:19 RDAanalysis_2poplabeling_biopca_prueba.R
-rw-rw-r--  1 aescudero aescudero     256261 Nov 18 19:39 rda_snp_scores_chr28.csv
-rw-rw-r--  1 aescudero aescudero      64452 Nov 18 19:39 selected_snps_for_plotting_chr28.csv
-rw-rw-r--  1 aescudero aescudero       6020 Nov 18 19:38 RDAanalysis_chr28.R
-rw-rw-r--  1 aescudero aescudero     256366 Nov 18 19:37 rda_snp_scores_chr32.csv
-rw-rw-r--  1 aescudero aescudero      64380 Nov 18 19:37 selected_snps_for_plotting_chr32.csv
-rw-rw-r--  1 aescudero aescudero       6020 Nov 18 19:36 RDAanalysis_chr32.R
-rw-rw-r--  1 aescudero aescudero     254393 Nov 18 19:34 rda_snp_scores_chr3.csv
-rw-rw-r--  1 aescudero aescudero      62287 Nov 18 19:34 selected_snps_for_plotting_chr3.csv
-rw-rw-r--  1 aescudero aescudero       6007 Nov 18 19:34 RDAanalysis_chr3.R
-rw-rw-r--  1 aescudero aescudero     256438 Nov 18 19:30 rda_snp_scores_chr20.csv
-rw-rw-r--  1 aescudero aescudero      64730 Nov 18 19:30 selected_snps_for_plotting_chr20.csv
-rw-rw-r--  1 aescudero aescudero       6020 Nov 18 19:29 RDAanalysis_chr20.R
-rw-rw-r--  1 aescudero aescudero     256699 Nov 18 19:26 rda_snp_scores_chr18.csv
-rw-rw-r--  1 aescudero aescudero      64875 Nov 18 19:26 selected_snps_for_plotting_chr18.csv
-rw-rw-r--  1 aescudero aescudero       6020 Nov 18 19:25 RDAanalysis_chr18.R
-rw-rw-r--  1 aescudero aescudero     257046 Nov 18 19:23 rda_snp_scores_chr14.csv
-rw-rw-r--  1 aescudero aescudero      64947 Nov 18 19:23 selected_snps_for_plotting_chr14.csv
-rw-rw-r--  1 aescudero aescudero       6020 Nov 18 19:22 RDAanalysis_chr14.R
-rw-rw-r--  1 aescudero aescudero       8367 Nov 18 19:15 RDAanalysis_2poplabeling_biopca_final.R
-rw-rw-r--  1 aescudero aescudero 2329462017 Nov 18 18:51 admixture_final_ld_pruned_filtered.vcf
-rw-rw-r--  1 aescudero aescudero   74665675 Nov 18 18:51 admixture_final_ld_pruned_filtered.fst
-rw-rw-r--  1 aescudero aescudero       1331 Nov 18 18:51 admixture_final_ld_pruned_filtered.log
-rw-rw-r--  1 aescudero aescudero       3796 Nov 18 18:51 admixture_final_ld_pruned_filtered.nosex
-rw-rw-r--  1 aescudero aescudero  112909701 Nov 18 18:51 admixture_final_ld_pruned_filtered.bcf
-rw-rw-r--  1 aescudero aescudero   38079217 Nov 18 18:51 admixture_final_ld_pruned_filtered.bed
-rw-rw-r--  1 aescudero aescudero   22727786 Nov 18 18:51 admixture_final_ld_pruned_filtered.bim
-rw-rw-r--  1 aescudero aescudero       5452 Nov 18 18:51 admixture_final_ld_pruned_filtered.fam
-rw-rw-r--  1 aescudero aescudero   38079217 Nov 18 18:50 admixture_final_ld_pruned_cleaned.bed
-rw-rw-r--  1 aescudero aescudero   22727786 Nov 18 18:50 admixture_final_ld_pruned_cleaned.bim
-rw-rw-r--  1 aescudero aescudero       5452 Nov 18 18:50 admixture_final_ld_pruned_cleaned.fam
-rw-rw-r--  1 aescudero aescudero       1315 Nov 18 18:50 admixture_final_ld_pruned_cleaned.log
-rw-rw-r--  1 aescudero aescudero      14468 Nov 18 18:50 admixture_final_ld_pruned_cleaned.nosex
-rw-rw-r--  1 aescudero aescudero  326660217 Nov 18 18:50 admixture_final_ld_pruned_filtered_for_RDA.raw
-rw-rw-r--  1 aescudero aescudero       1099 Nov 18 18:50 admixture_final_ld_pruned_filtered_for_RDA.log
-rw-rw-r--  1 aescudero aescudero       3796 Nov 18 18:50 admixture_final_ld_pruned_filtered_for_RDA.nosex
-rw-rw-r--  1 aescudero aescudero       8013 Nov 18 15:46 RDAanalysis_2poplabeling_biopca.R
-rw-rw-r--  1 aescudero aescudero      16183 Nov 18 14:37 RDA_plot.pdf
-rw-rw-r--  1 aescudero aescudero    1839475 Nov 18 12:13 filtered_genetic_data.tsv
-rw-rw-r--  1 aescudero aescudero       1825 Nov 18 12:12 filter_genetic_data_before_plotting_RAD.R
-rw-rw-r--  1 aescudero aescudero      53569 Nov 18 12:11 admixture_final_ld_pruned_filtered_snps_to_keep.txt
-rw-rw-r--  1 aescudero aescudero  112909294 Nov 18 12:11 admixture_final_ld_pruned_filtered_clean.bcf
drwxrwxr-x 12 aescudero aescudero     540672 Nov 18 12:08 [38;5;33m..[0m
-rw-rw-r--  1 aescudero aescudero      66082 Nov 18 11:59 rda_snp_scores.csv
-rw-rw-r--  1 aescudero aescudero      16544 Nov 18 11:59 selected_snps_for_plotting.csv
-rw-rw-r--  1 aescudero aescudero       2985 Nov 18 11:28 selectingindependentvariables.R
-rw-rw-r--  1 aescudero aescudero      59408 Nov 18 11:14 individual_bioclim_data.csv
-rw-rw-r--  1 aescudero aescudero       2959 Nov 18 11:14 scriptdowloadingworldclim.R
-rw-rw-r--  1 aescudero aescudero      18272 Sep  8 13:08 inversion_hybridity_and_genotype.csv
-rw-rw-r--  1 aescudero aescudero       9014 Sep  8 13:04 inversion_heterozygosity_per_inversion.csv
-rw-rw-r--  1 aescudero aescudero      26688 Sep  8 13:03 genotype_differences.csv
-rw-rw-r--  1 aescudero aescudero       5490 Sep  8 13:03 inversion_heterozygosity.csv
-rw-rw-r--  1 aescudero aescudero      11406 Sep  8 12:43 inversion_genotypes.csv
-rw-rw-r--  1 aescudero aescudero    7459548 Sep  8 11:57 filtered_inversion_genotypes.csv
-rw-rw-r--  1 aescudero aescudero       2715 Sep  8 11:55 inverted_genotype_analysis.R
-rw-rw-r--  1 aescudero aescudero       2826 Sep  4 15:56 population_list.txt
-rw-rw-r--  1 aescudero aescudero    1189158 Sep  4 15:38 [38;5;13mcomprehensive_genomic_plot.png[0m
-rw-rw-r--  1 aescudero aescudero   79534576 Sep  4 15:19 admixture_final_ld_pruned_filtered_pop1_vs_pop2.fst.sorted
-rw-rw-r--  1 aescudero aescudero      25995 Sep  3 11:42 bioclim_for_rda.csv
-rw-rw-r--  1 aescudero aescudero    2289359 Sep  3 11:42 genetic_data_for_rda.csv
-rw-rw-r--  1 aescudero aescudero 1026883584 Sep  3 11:39 genetic_data_final.csv
drwxrwxr-x  2 aescudero aescudero       4096 Sep  1 13:25 [38;5;33mwc2.1_country[0m
-rw-rw-r--  1 aescudero aescudero      10921 Sep  1 12:54 locations.txt
]0;aescudero@login3:~/wgg/RDA(plot_env) [aescudero@nodo4143 RDA]$ cat RDAanalysis_2poplabeling_biopca_final.R
#---------------------------------------------------#
# Step 1: Install and Load Required Packages
#---------------------------------------------------#
# 'vegan' for the RDA, 'ggplot2' for advanced plotting,
# and 'data.table' for efficient data reading.
if (!requireNamespace("vegan", quietly = TRUE)) {
  install.packages("vegan", dependencies = TRUE)
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", dependencies = TRUE)
}

if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table", dependencies = TRUE)
}

library(vegan)
library(ggplot2)
library(data.table)

#---------------------------------------------------#
# Step 2: Load and Clean Data
#---------------------------------------------------#

# Load the bioclimatic data.
bioclim_data <- read.csv("final_df_for_analysis.csv")

# Load the genetic data using fread() for efficiency.
raw_data <- fread("admixture_final_ld_pruned_filtered_for_RDA.raw")

# Load the population data.
tryCatch({
  # Changed from read.csv with sep="\t" to read.table with sep=" "
  pop_data <- read.table("population_list.txt", sep=" ", header = TRUE)
  if (is.null(pop_data) || nrow(pop_data) == 0) {
    stop("Population data file is empty or invalid.")
  }
}, error = function(e) {
  stop("Error loading population data: ", e$message)
})

# Format the genetic data: remove non-genotype columns and set row names.
genetic_data_final <- as.data.frame(raw_data[, 7:ncol(raw_data)])
rownames(genetic_data_final) <- raw_data$IID

# --- Subsample and clean the genetic data ---
# We will check for and remove any SNPs (columns) with NA/NaN/Inf values.
invalid_snps <- which(apply(genetic_data_final, 2, function(x) any(!is.finite(x))))
if(length(invalid_snps) > 0) {
    warning("Removed ", length(invalid_snps), " SNPs with invalid data.")
    genetic_data_final <- genetic_data_final[, -invalid_snps]
}

# Subsample SNPs to a manageable number (e.g., 5000) AFTER cleaning
set.seed(42) # For reproducibility
if (ncol(genetic_data_final) > 5000) {
    selected_snps <- sample(1:ncol(genetic_data_final), 5000)
    genetic_data_final <- genetic_data_final[, selected_snps]
}
cat("Final genetic data dimensions:", dim(genetic_data_final), "\n")


#---------------------------------------------------#
# Step 3: Align and Filter Data by Individuals
#---------------------------------------------------#
# This step ensures both datasets are perfectly aligned.

# Find individuals with complete bioclimatic data.
valid_bioclim_individuals <- complete.cases(bioclim_data)

# Find individuals with complete genetic data.
valid_genetic_individuals <- complete.cases(genetic_data_final)

# Find common individuals with valid data in all three sets.
common_valid_individuals <- intersect(bioclim_data$IND[valid_bioclim_individuals], rownames(genetic_data_final)[valid_genetic_individuals])
common_valid_individuals <- intersect(common_valid_individuals, pop_data$IID)

# --- NEW CHECK: Stop if no common individuals are found ---
if (length(common_valid_individuals) == 0) {
  stop("Error: No common individuals found across all three datasets (bioclimatic, genetic, and population data). Please check your input files.")
}

# Filter all data frames to keep only these individuals.
bioclim_data_filtered <- bioclim_data[bioclim_data$IND %in% common_valid_individuals, ]
genetic_data_final <- genetic_data_final[common_valid_individuals, ]
# The key change is adding drop = FALSE here to prevent simplification to a vector
pop_data_filtered <- pop_data[pop_data$IID %in% common_valid_individuals, , drop = FALSE]

# Order all data frames to ensure perfect alignment.
bioclim_data_filtered <- bioclim_data_filtered[order(bioclim_data_filtered$IND), ]
genetic_data_final <- genetic_data_final[order(rownames(genetic_data_final)), ]
# The key change is adding drop = FALSE here to prevent simplification to a vector
pop_data_filtered <- pop_data_filtered[order(pop_data_filtered$IID), , drop = FALSE]

# Final check of alignment
stopifnot(all(rownames(genetic_data_final) == bioclim_data_filtered$IND))
stopifnot(all(rownames(genetic_data_final) == pop_data_filtered$IID))
cat("Final number of individuals for RDA:", nrow(genetic_data_final), "\n")


#---------------------------------------------------#
# Step 4: Run the RDA Analysis
#---------------------------------------------------#
# Separate the predictor variables (bioclimatic) from the full data frame.
bioclim_predictors <- bioclim_data_filtered[, c(
  "wc2.1_30s_bio_1", "wc2.1_30s_bio_4", "wc2.1_30s_bio_12",
  "wc2.1_30s_bio_14", "wc2.1_30s_bio_16"
)]

# Scale and center the predictor variables.
bioclim_scaled <- scale(bioclim_predictors)

# Run the RDA.
rda_result <- rda(genetic_data_final ~ ., data = as.data.frame(bioclim_scaled))

#---------------------------------------------------#
# Step 5: Interpret and Visualize the RDA Results with ggplot2
#---------------------------------------------------#
summary(rda_result)
anova.cca(rda_result, permutations = 999)
anova.cca(rda_result, by = "margin", permutations = 999)

# Get the scores for plotting
site_scores <- scores(rda_result, display = "sites", scaling = 2)
env_scores <- scores(rda_result, display = "bp", scaling = 2)

# Convert scores to data frames for use with ggplot2
site_scores_df <- as.data.frame(site_scores)
site_scores_df$IID <- rownames(site_scores_df)
env_scores_df <- as.data.frame(env_scores)
env_scores_df$variables <- rownames(env_scores_df)

# Merge site scores with population data
plot_data <- merge(site_scores_df, pop_data_filtered, by = "IID")

# Create the RDA plot
rda_plot <- ggplot() +
  # Add points colored by population
  geom_point(data = plot_data, aes(x = RDA1, y = RDA2, color = POP), size = 3) +
  # Add arrows for environmental variables
  geom_segment(data = env_scores_df, aes(x = 0, y = 0, xend = RDA1, yend = RDA2), 
               arrow = arrow(length = unit(0.2, "cm")), color = "red") +
  # Add labels for environmental variables
  geom_text(data = env_scores_df, aes(x = RDA1, y = RDA2, label = variables), 
            hjust = 0, nudge_x = 0.05, color = "red") +
  
  # --- CHANGE START: Custom Colors and Labels ---
  # Maps POP1 to C. borbonica (cyan) and POP2 to C. boryana (orange)
  scale_color_manual(
    values = c("POP1" = "cyan", "POP2" = "orange"),
    labels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")
  ) +
  # --- CHANGE END ---

  # Set plot title and labels
  labs(title = "",
       subtitle = "",
       x = "RDA1", y = "RDA2", color = "Species") +
  # Add a theme for a clean look
  theme_minimal() +
  # Center the title and customize legend
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

# Save the RDA plot to a PDF file
pdf("RDA_plot_pop.pdf", width = 8, height = 8)
print(rda_plot)
dev.off()

print("RDA plot saved as RDA_plot_pop.pdf")

#---------------------------------------------------#
# Step 6: Perform and Visualize PCA on Bioclimatic Data
#---------------------------------------------------#
print("Performing PCA on bioclimatic variables...")

# Fix: Set the row names of the bioclimatic data to the individual IDs
# to ensure the merge with population data works correctly.
rownames(bioclim_predictors) <- bioclim_data_filtered$IND

# Perform PCA on the scaled bioclimatic data.
pca_bioclim <- prcomp(bioclim_predictors, center = TRUE, scale. = TRUE)

# Get the individual scores for PC1 and PC2.
pca_scores <- as.data.frame(pca_bioclim$x)
pca_scores$IID <- rownames(pca_scores)

# Merge PCA scores with population data.
pca_plot_data <- merge(pca_scores, pop_data_filtered, by = "IID")

# Create the PCA plot.
pca_plot <- ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = POP)) +
  geom_point(size = 3) +
  
  # --- CHANGE START: Custom Colors and Labels ---
  scale_color_manual(
    values = c("POP1" = "cyan", "POP2" = "orange"),
    labels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")
  ) +
  # --- CHANGE END ---

  labs(title = "",
       subtitle = "",
       x = "PC1", y = "PC2", color = "Populations") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

# Save the PCA plot to a PDF file.
pdf("PCA_plot_bioclim.pdf", width = 8, height = 8)
print(pca_plot)
dev.off()

print("PCA plot saved as PCA_plot_bioclim.pdf")
]0;aescudero@login3:~/wgg/RDA(plot_env) [aescudero@nodo4143 RDA]$ cat RDAanalysis_2poplabeling_biopca_final.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cls -la -t[K[Kcd RDA/[2P..at sv_local_density_plots_30_all.pyd ..[KRDA/ls[K -la -tcat RDAanalysis_2poplabeling_biopca_final.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Kcat /home/aescudero/wgg/RDAanalysis_2poplabeling_biopca_final.R[1P RDAanalysis_2poplabeling_biopca_final.R[1P RDAanalysis_2poplabeling_biopca_final.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cn RDAanalysis_2poplabeling_biopca_final.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Ca RDAanalysis_2poplabeling_biopca_final.Rn RDAanalysis_2poplabeling_biopca_final.Ro RDAanalysis_2poplabeling_biopca_final.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C_.R1.R0.R0.R0.R
[?1049h[22;0;0t[1;45r(B[m[4l[?7h[39;49m[?1h=[?1h=[?1h=[?25l[39;49m(B[m[H[2J[43;87H(B[0;7m[ New File ](B[m[H(B[0;7m  GNU nano 2.9.8                                                         RDAanalysis_2poplabeling_biopca_final_1000.R                                                                   [1;183H(B[m[44d(B[0;7m^G(B[m Get Help     (B[0;7m^O(B[m Write Out    (B[0;7m^W(B[m Where Is     (B[0;7m^K(B[m Cut Text     (B[0;7m^J(B[m Justify	(B[0;7m^C(B[m Cur Pos	(B[0;7mM-U(B[m Undo[44;113H(B[0;7mM-A(B[m Mark Text   (B[0;7mM-](B[m To Bracket  (B[0;7mM-▲(B[m Previous    (B[0;7m^B(B[m Back[45d(B[0;7m^X(B[m Exit[45;17H(B[0;7m^R(B[m Read File    (B[0;7m^\(B[m Replace	(B[0;7m^U(B[m Uncut Text   (B[0;7m^T(B[m To Spell     (B[0;7m^_(B[m Go To Line   (B[0;7mM-E(B[m Redo[45;113H(B[0;7mM-6(B[m Copy Text   (B[0;7mM-W(B[m WhereIs Next(B[0;7mM-▼(B[m Next[45;161H(B[0;7m^F(B[m Forward[43d[3d[39;49m(B[m[?12l[?25h[?25l[1;175H(B[0;7mModified[43d(B[m[K[1;183H[3;3H) +[4;3Hlabs(title = paste0("RDA on FST Outlier SNPs (Top ", (1-FST_QUANTILE)*100, "% windows)"),[5;8Hx = "RDA1", y = "RDA2", color = "Species") +[6;3Htheme_minimal() +[7;3Htheme(plot.title = element_text(hjust = 0.5))[9dpdf("RDA_plot_pop_FST_filtered.pdf", width = 8, height = 8)[10dprint(rda_plot)[11ddev.off()[12dprint("RDA plot saved as RDA_plot_pop_FST_filtered.pdf")[14d[36m#---------------------------------------------------#[15d# Step 6: PCA on Bioclim[16d#---------------------------------------------------#[17d[39m(B[mrownames(bioclim_predictors) <- bioclim_data_filtered$IND[18dpca_bioclim <- prcomp(bioclim_predictors, center = TRUE, scale. = TRUE)[19dpca_scores <- as.data.frame(pca_bioclim$x)[20dpca_scores$IID <- rownames(pca_scores)[21dpca_plot_data <- merge(pca_scores, pop_data_filtered, by = "IID")[23dpca_plot <- ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = POP)) +[24;3Hgeom_point(size = 3) +[25;3Hscale_color_manual([26;5Hvalues = c("POP1" = "cyan", "POP2" = "orange"),[27;5Hlabels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")[28;3H) +[29;3Hlabs(title = "PCA on Bioclimatic Variables", x = "PC1", y = "PC2", color = "Populations") +[30;3Htheme_minimal() +[31;3Htheme(plot.title = element_text(hjust = 0.5))[33dpdf("PCA_plot_bioclim.pdf", width = 8, height = 8)[34dprint(pca_plot)[35ddev.off()[36dprint("PCA plot saved as PCA_plot_bioclim_1000.pdf")[?12l[?25h[?25l[43d(B[0;7mSave modified buffer?  (Answering "No" will DISCARD changes.)                                                                                                                           [44;1H Y(B[m Yes[K[45d(B[0;7m N(B[m No  [45;18H(B[0;7mC(B[m Cancel[K[43;63H[?12l[?25h[?25l[44d(B[0;7m^G(B[m Get Help[44;47H(B[0;7mM-D(B[m DOS Format[44;93H(B[0;7mM-A(B[m Append[44;139H(B[0;7mM-B(B[m Backup File[45d(B[0;7m^C(B[m Cancel	         [45;47H(B[0;7mM-M(B[m Mac Format[45;93H(B[0;7mM-P(B[m Prepend[45;139H(B[0;7m^T(B[m To Files[43d(B[0;7mFile Name to Write: RDAanalysis_2poplabeling_biopca_final_1000.R(B[m[43;65H[?12l[?25h[?25l[K[1;183H[1;175H(B[0;7m        (B[m[43;83H(B[0;7m[ Wrote 194 lines ](B[m[J[45;184H[?12l[?25h[45;1H[?1049l[23;0;0t[?1l>]0;aescudero@login3:~/wgg/RDA(plot_env) [aescudero@nodo4143 RDA]$ conda deactivate
]0;aescudero@login3:~/wgg/RDA(base) [aescudero@nodo4143 RDA]$ conda deactivate[K[K[K[K[K[K[K[K[K[Kactivate R_env

EnvironmentNameNotFound: Could not find conda environment: R_env
You can list all discoverable environments with `conda info --envs`.


]0;aescudero@login3:~/wgg/RDA(base) [aescudero@nodo4143 RDA]$ conda activate R_env[1Penv\env[1Penv
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ conda activate Renv_env[4Pdeactivatenano RDAanalysis_2poplabeling_biopca_final_1000.R[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[CR RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cs RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cc RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cr RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Ci RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cp RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Ct RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
[?25h[?25hLoading required package: permute
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hWarning message:
Removed 606380 SNPs with invalid data. 
[?25h
--- STARTING FST WINDOW FILTERING ---
[?25h[?25h[?25h[?25h[?25h[?25hFST Threshold (Top 5%): 0.7961
[?25hNumber of outlier windows identified: 17473
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hError in foverlaps(snp_info, outlier_windows, type = "within", nomatch = 0L) : 
  Duplicate columns are not allowed in overlap joins. This may change in the future.
Calls: foverlaps -> stopf -> raise_condition -> signal
Execution halted
[?25h]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ Rscript RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cr RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cm RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ rm RDAanalysis_2poplabeling_biopca_final_1000.R[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cn RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Ca RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cn RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Co RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
[?1049h[22;0;0t[1;45r(B[m[4l[?7h[39;49m[?1h=[?1h=[?1h=[?25l[39;49m(B[m[H[2J[43;87H(B[0;7m[ New File ](B[m[H(B[0;7m  GNU nano 2.9.8                                                         RDAanalysis_2poplabeling_biopca_final_1000.R                                                                   [1;183H(B[m[44d(B[0;7m^G(B[m Get Help     (B[0;7m^O(B[m Write Out    (B[0;7m^W(B[m Where Is     (B[0;7m^K(B[m Cut Text     (B[0;7m^J(B[m Justify	(B[0;7m^C(B[m Cur Pos	(B[0;7mM-U(B[m Undo[44;113H(B[0;7mM-A(B[m Mark Text   (B[0;7mM-](B[m To Bracket  (B[0;7mM-▲(B[m Previous    (B[0;7m^B(B[m Back[45d(B[0;7m^X(B[m Exit[45;17H(B[0;7m^R(B[m Read File    (B[0;7m^\(B[m Replace	(B[0;7m^U(B[m Uncut Text   (B[0;7m^T(B[m To Spell     (B[0;7m^_(B[m Go To Line   (B[0;7mM-E(B[m Redo[45;113H(B[0;7mM-6(B[m Copy Text   (B[0;7mM-W(B[m WhereIs Next(B[0;7mM-▼(B[m Next[45;161H(B[0;7m^F(B[m Forward[43d[3d[39;49m(B[m[?12l[?25h[?25l[1;175H(B[0;7mModified[43d(B[m[K[1;183H[3;13Hhjust = 0, nudge_x = 0.05, color = "red") +[4;3Hscale_color_manual([5;5Hvalues = c("POP1" = "cyan", "POP2" = "orange"),[6;5Hlabels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")[7;3H) +[8;3Hlabs(title = paste0("RDA on FST Outlier SNPs (Top ", (1-FST_QUANTILE)*100, "% windows)"),[9;8Hx = "RDA1", y = "RDA2", color = "Species") +[10;3Htheme_minimal() +[11;3Htheme(plot.title = element_text(hjust = 0.5))[13dpdf("RDA_plot_pop_FST_filtered_1000.pdf", width = 8, height = 8)[14dprint(rda_plot)[15ddev.off()[16dprint("RDA plot saved as RDA_plot_pop_FST_filtered_1000.pdf")[18d[36m#---------------------------------------------------#[19d# Step 6: PCA on Bioclim[20d#---------------------------------------------------#[21d[39m(B[mrownames(bioclim_predictors) <- bioclim_data_filtered$IND[22dpca_bioclim <- prcomp(bioclim_predictors, center = TRUE, scale. = TRUE)[23dpca_scores <- as.data.frame(pca_bioclim$x)[24dpca_scores$IID <- rownames(pca_scores)[25dpca_plot_data <- merge(pca_scores, pop_data_filtered, by = "IID")[27dpca_plot <- ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = POP)) +[28;3Hgeom_point(size = 3) +[29;3Hscale_color_manual([30;5Hvalues = c("POP1" = "cyan", "POP2" = "orange"),[31;5Hlabels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")[32;3H) +[33;3Hlabs(title = "PCA on Bioclimatic Variables", x = "PC1", y = "PC2", color = "Populations") +[34;3Htheme_minimal() +[35;3Htheme(plot.title = element_text(hjust = 0.5))[37dpdf("PCA_plot_bioclim.pdf", width = 8, height = 8)[38dprint(pca_plot)[39ddev.off()[40dprint("PCA plot saved as PCA_plot_bioclim.pdf")[?12l[?25h[?25l[43d(B[0;7mSave modified buffer?  (Answering "No" will DISCARD changes.)                                                                                                                           [44;1H Y(B[m Yes[K[45d(B[0;7m N(B[m No  [45;18H(B[0;7mC(B[m Cancel[K[43;63H[?12l[?25h[?25l[44d(B[0;7m^G(B[m Get Help[44;47H(B[0;7mM-D(B[m DOS Format[44;93H(B[0;7mM-A(B[m Append[44;139H(B[0;7mM-B(B[m Backup File[45d(B[0;7m^C(B[m Cancel	         [45;47H(B[0;7mM-M(B[m Mac Format[45;93H(B[0;7mM-P(B[m Prepend[45;139H(B[0;7m^T(B[m To Files[43d(B[0;7mFile Name to Write: RDAanalysis_2poplabeling_biopca_final_1000.R(B[m[43;65H[?12l[?25h[?25l[K[1;183H[1;175H(B[0;7m        (B[m[43;83H(B[0;7m[ Wrote 198 lines ](B[m[J[45;184H[?12l[?25h[45;1H[?1049l[23;0;0t[?1l>]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ nano RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[2Prm[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[5@Rscript[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
[?25h[?25hLoading required package: permute
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hWarning message:
Removed 606380 SNPs with invalid data. 
[?25h
--- STARTING FST WINDOW FILTERING ---
[?25h[?25h[?25h[?25h[?25h[?25hFST Threshold (Top 1%): 0.9128
[?25hNumber of outlier windows identified: 3495
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hNumber of SNPs located in FST outlier windows: 199
[?25h[?25h[?25hFinal genetic data dimensions for RDA: 184 199 
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hnull device 
          1 
[?25h[1] "RDA plot saved as RDA_plot_pop_FST_filtered_1000.pdf"
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hnull device 
          1 
[?25h[1] "PCA plot saved as PCA_plot_bioclim.pdf"
[?25h[?25h]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ Rscript RDAanalysis_2poplabeling_biopca_final_1000.R[1P.R[1P.R[1P.R[1P.R[1P.R
[?25h[?25hLoading required package: permute
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hWarning message:
Removed 606380 SNPs with invalid data. 
[?25h[?25h[?25hFinal genetic data dimensions: 184 5000 
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hFinal number of individuals for RDA: 184 
[?25h[?25h[?25h[?25h
Call:
rda(formula = genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 +      wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + wc2.1_30s_bio_16, data = as.data.frame(bioclim_scaled)) 

Partitioning of variance:
              Inertia Proportion
Total          1728.3     1.0000
Constrained     474.5     0.2746
Unconstrained  1253.7     0.7254

Eigenvalues, and their contribution to the variance 

Importance of components:
                          RDA1     RDA2     RDA3     RDA4      RDA5       PC1
Eigenvalue            287.3834 72.99167 57.96457 40.19117 16.011282 132.01727
Proportion Explained    0.1663  0.04223  0.03354  0.02326  0.009264   0.07639
Cumulative Proportion   0.1663  0.20852  0.24206  0.26531  0.274577   0.35096
                           PC2      PC3      PC4      PC5      PC6     PC7
Eigenvalue            68.63702 34.74744 33.22067 33.08984 26.11051 23.5047
Proportion Explained   0.03971  0.02011  0.01922  0.01915  0.01511  0.0136
Cumulative Proportion  0.39068  0.41078  0.43001  0.44915  0.46426  0.4779
                           PC8      PC9      PC10     PC11      PC12      PC13
Eigenvalue            22.40181 19.76991 17.223788 15.46855 14.357883 13.210410
Proportion Explained   0.01296  0.01144  0.009966  0.00895  0.008308  0.007644
Cumulative Proportion  0.49082  0.50226  0.512227  0.52118  0.529485  0.537129
                           PC14      PC15      PC16      PC17     PC18     PC19
Eigenvalue            11.387081 11.049344 10.914450 10.566626 10.45658 10.04038
Proportion Explained   0.006589  0.006393  0.006315  0.006114  0.00605  0.00581
Cumulative Proportion  0.543718  0.550111  0.556426  0.562540  0.56859  0.57440
                         PC20     PC21     PC22     PC23     PC24     PC25
Eigenvalue            9.97230 9.881718 9.721613 9.511237 9.464861 9.365120
Proportion Explained  0.00577 0.005718 0.005625 0.005503 0.005477 0.005419
Cumulative Proportion 0.58017 0.585888 0.591513 0.597016 0.602493 0.607912
                          PC26     PC27     PC28     PC29     PC30     PC31
Eigenvalue            9.267460 9.157299 9.058698 9.024429 8.782894 8.769539
Proportion Explained  0.005362 0.005299 0.005241 0.005222 0.005082 0.005074
Cumulative Proportion 0.613274 0.618572 0.623814 0.629036 0.634117 0.639192
                          PC32     PC33     PC34     PC35     PC36     PC37
Eigenvalue            8.614089 8.568338 8.461162 8.415006 8.309412 8.263158
Proportion Explained  0.004984 0.004958 0.004896 0.004869 0.004808 0.004781
Cumulative Proportion 0.644176 0.649134 0.654029 0.658898 0.663706 0.668488
                          PC38     PC39     PC40     PC41     PC42     PC43
Eigenvalue            8.150939 7.979568 7.806575 7.707043 7.633472 7.513000
Proportion Explained  0.004716 0.004617 0.004517 0.004459 0.004417 0.004347
Cumulative Proportion 0.673204 0.677821 0.682338 0.686797 0.691214 0.695561
                          PC44    PC45     PC46     PC47     PC48     PC49
Eigenvalue            7.438580 7.34439 7.285335 7.242679 7.142780 7.100154
Proportion Explained  0.004304 0.00425 0.004215 0.004191 0.004133 0.004108
Cumulative Proportion 0.699865 0.70411 0.708330 0.712521 0.716654 0.720762
                         PC50     PC51    PC52     PC53     PC54     PC55
Eigenvalue            6.98259 6.920141 6.82587 6.685417 6.592252 6.556190
Proportion Explained  0.00404 0.004004 0.00395 0.003868 0.003814 0.003794
Cumulative Proportion 0.72480 0.728807 0.73276 0.736624 0.740439 0.744232
                          PC56     PC57    PC58     PC59     PC60     PC61
Eigenvalue            6.503461 6.428041 6.36026 6.298743 6.275754 6.176311
Proportion Explained  0.003763 0.003719 0.00368 0.003645 0.003631 0.003574
Cumulative Proportion 0.747995 0.751715 0.75539 0.759039 0.762671 0.766244
                          PC62     PC63     PC64     PC65     PC66     PC67
Eigenvalue            6.079672 6.061023 6.015511 5.960295 5.884521 5.867716
Proportion Explained  0.003518 0.003507 0.003481 0.003449 0.003405 0.003395
Cumulative Proportion 0.769762 0.773269 0.776750 0.780198 0.783603 0.786998
                          PC68     PC69     PC70     PC71     PC72     PC73
Eigenvalue            5.827247 5.768638 5.728687 5.641931 5.604376 5.589061
Proportion Explained  0.003372 0.003338 0.003315 0.003265 0.003243 0.003234
Cumulative Proportion 0.790370 0.793708 0.797023 0.800287 0.803530 0.806764
                          PC74     PC75     PC76     PC77     PC78     PC79
Eigenvalue            5.515221 5.499430 5.435899 5.382788 5.348028 5.332334
Proportion Explained  0.003191 0.003182 0.003145 0.003115 0.003094 0.003085
Cumulative Proportion 0.809955 0.813137 0.816282 0.819397 0.822491 0.825577
                          PC80     PC81     PC82     PC83     PC84     PC85
Eigenvalue            5.246660 5.242910 5.209561 5.143179 5.106499 5.093558
Proportion Explained  0.003036 0.003034 0.003014 0.002976 0.002955 0.002947
Cumulative Proportion 0.828613 0.831646 0.834660 0.837636 0.840591 0.843538
                         PC86     PC87     PC88    PC89     PC90     PC91
Eigenvalue            5.04641 4.995962 4.940506 4.92605 4.865434 4.800991
Proportion Explained  0.00292 0.002891 0.002859 0.00285 0.002815 0.002778
Cumulative Proportion 0.84646 0.849349 0.852208 0.85506 0.857873 0.860651
                          PC92     PC93     PC94     PC95     PC96     PC97
Eigenvalue            4.767736 4.693996 4.682146 4.650554 4.628814 4.520155
Proportion Explained  0.002759 0.002716 0.002709 0.002691 0.002678 0.002615
Cumulative Proportion 0.863410 0.866126 0.868835 0.871526 0.874204 0.876819
                          PC98     PC99    PC100    PC101    PC102    PC103
Eigenvalue            4.494495 4.461972 4.435585 4.340237 4.281749 4.271527
Proportion Explained  0.002601 0.002582 0.002566 0.002511 0.002477 0.002472
Cumulative Proportion 0.879420 0.882002 0.884568 0.887080 0.889557 0.892029
                         PC104    PC105    PC106    PC107    PC108    PC109
Eigenvalue            4.236600 4.179379 4.153017 4.107962 4.041052 3.997794
Proportion Explained  0.002451 0.002418 0.002403 0.002377 0.002338 0.002313
Cumulative Proportion 0.894480 0.896898 0.899301 0.901678 0.904016 0.906330
                        PC110    PC111    PC112    PC113    PC114    PC115
Eigenvalue            3.95853 3.918236 3.893411 3.824744 3.811098 3.757017
Proportion Explained  0.00229 0.002267 0.002253 0.002213 0.002205 0.002174
Cumulative Proportion 0.90862 0.910887 0.913140 0.915353 0.917558 0.919732
                         PC116    PC117    PC118    PC119    PC120    PC121
Eigenvalue            3.705231 3.677223 3.651698 3.603525 3.585996 3.575912
Proportion Explained  0.002144 0.002128 0.002113 0.002085 0.002075 0.002069
Cumulative Proportion 0.921876 0.924004 0.926117 0.928202 0.930277 0.932346
                         PC122    PC123    PC124    PC125    PC126    PC127
Eigenvalue            3.502372 3.420447 3.399918 3.356229 3.341255 3.279264
Proportion Explained  0.002027 0.001979 0.001967 0.001942 0.001933 0.001897
Cumulative Proportion 0.934372 0.936351 0.938319 0.940260 0.942194 0.944091
                         PC128    PC129    PC130    PC131    PC132    PC133
Eigenvalue            3.265030 3.217738 3.200059 3.168349 3.112672 3.065762
Proportion Explained  0.001889 0.001862 0.001852 0.001833 0.001801 0.001774
Cumulative Proportion 0.945980 0.947842 0.949694 0.951527 0.953328 0.955102
                         PC134    PC135    PC136    PC137    PC138    PC139
Eigenvalue            3.035911 2.988688 2.974292 2.959113 2.931322 2.900797
Proportion Explained  0.001757 0.001729 0.001721 0.001712 0.001696 0.001678
Cumulative Proportion 0.956859 0.958588 0.960309 0.962021 0.963717 0.965396
                        PC140    PC141    PC142    PC143    PC144    PC145
Eigenvalue            2.85137 2.812277 2.753910 2.729558 2.684295 2.675035
Proportion Explained  0.00165 0.001627 0.001593 0.001579 0.001553 0.001548
Cumulative Proportion 0.96705 0.968673 0.970266 0.971846 0.973399 0.974947
                         PC146    PC147    PC148    PC149    PC150    PC151
Eigenvalue            2.642740 2.554225 2.531984 2.475917 2.266223 2.196927
Proportion Explained  0.001529 0.001478 0.001465 0.001433 0.001311 0.001271
Cumulative Proportion 0.976476 0.977954 0.979419 0.980851 0.982162 0.983434
                         PC152    PC153    PC154     PC155     PC156     PC157
Eigenvalue            2.154027 1.930098 1.739004 1.5966554 1.5121203 1.3572363
Proportion Explained  0.001246 0.001117 0.001006 0.0009238 0.0008749 0.0007853
Cumulative Proportion 0.984680 0.985797 0.986803 0.9877268 0.9886018 0.9893871
                          PC158     PC159     PC160     PC161     PC162
Eigenvalue            1.2594564 1.1289167 1.0555100 1.0460310 1.0040296
Proportion Explained  0.0007287 0.0006532 0.0006107 0.0006052 0.0005809
Cumulative Proportion 0.9901158 0.9907690 0.9913798 0.9919850 0.9925660
                          PC163     PC164    PC165     PC166     PC167
Eigenvalue            1.0029709 0.9569521 0.922972 0.9188095 0.9012801
Proportion Explained  0.0005803 0.0005537 0.000534 0.0005316 0.0005215
Cumulative Proportion 0.9931463 0.9937000 0.994234 0.9947657 0.9952872
                          PC168     PC169     PC170     PC171     PC172
Eigenvalue            0.8730976 0.8375901 0.7965876 0.7697028 0.7590206
Proportion Explained  0.0005052 0.0004846 0.0004609 0.0004454 0.0004392
Cumulative Proportion 0.9957924 0.9962770 0.9967379 0.9971833 0.9976225
                          PC173     PC174     PC175     PC176     PC177
Eigenvalue            0.7385377 0.7257166 0.6976674 0.6694382 0.6466926
Proportion Explained  0.0004273 0.0004199 0.0004037 0.0003873 0.0003742
Cumulative Proportion 0.9980498 0.9984697 0.9988734 0.9992607 0.9996349
                          PC178
Eigenvalue            0.6309561
Proportion Explained  0.0003651
Cumulative Proportion 1.0000000

Accumulated constrained eigenvalues
Importance of components:
                          RDA1    RDA2    RDA3     RDA4     RDA5
Eigenvalue            287.3834 72.9917 57.9646 40.19117 16.01128
Proportion Explained    0.6056  0.1538  0.1221  0.08469  0.03374
Cumulative Proportion   0.6056  0.7594  0.8816  0.96626  1.00000

[?25hPermutation test for rda under reduced model
Permutation: free
Number of permutations: 999

Model: rda(formula = genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 + wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + wc2.1_30s_bio_16, data = as.data.frame(bioclim_scaled))
          Df Variance      F Pr(>F)    
Model      5   474.54 13.475  0.001 ***
Residual 178  1253.72                  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
[?25hPermutation test for rda under reduced model
Marginal effects of terms
Permutation: free
Number of permutations: 999

Model: rda(formula = genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 + wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + wc2.1_30s_bio_16, data = as.data.frame(bioclim_scaled))
                  Df Variance      F Pr(>F)    
wc2.1_30s_bio_1    1    36.35 5.1609  0.001 ***
wc2.1_30s_bio_4    1    21.27 3.0200  0.001 ***
wc2.1_30s_bio_12   1    44.69 6.3451  0.001 ***
wc2.1_30s_bio_14   1    58.20 8.2636  0.001 ***
wc2.1_30s_bio_16   1    36.71 5.2121  0.001 ***
Residual         178  1253.72                  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hnull device 
          1 
[?25h[1] "RDA plot saved as RDA_plot_pop.pdf"
[?25h[1] "Performing PCA on bioclimatic variables..."
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hnull device 
          1 
[?25h[1] "PCA plot saved as PCA_plot_bioclim.pdf"
[?25h[?25h]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ Rscript RDAanalysis_2poplabeling_biopca_final.R_1000.R[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cr RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cm RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ rm RDAanalysis_2poplabeling_biopca_final_1000.R[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[1P RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cn RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Ca RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cn RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Co RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
[?1049h[22;0;0t[1;45r(B[m[4l[?7h[39;49m[?1h=[?1h=[?1h=[?25l[39;49m(B[m[H[2J[43;87H(B[0;7m[ New File ](B[m[H(B[0;7m  GNU nano 2.9.8                                                         RDAanalysis_2poplabeling_biopca_final_1000.R                                                                   [1;183H(B[m[44d(B[0;7m^G(B[m Get Help     (B[0;7m^O(B[m Write Out    (B[0;7m^W(B[m Where Is     (B[0;7m^K(B[m Cut Text     (B[0;7m^J(B[m Justify	(B[0;7m^C(B[m Cur Pos	(B[0;7mM-U(B[m Undo[44;113H(B[0;7mM-A(B[m Mark Text   (B[0;7mM-](B[m To Bracket  (B[0;7mM-▲(B[m Previous    (B[0;7m^B(B[m Back[45d(B[0;7m^X(B[m Exit[45;17H(B[0;7m^R(B[m Read File    (B[0;7m^\(B[m Replace	(B[0;7m^U(B[m Uncut Text   (B[0;7m^T(B[m To Spell     (B[0;7m^_(B[m Go To Line   (B[0;7mM-E(B[m Redo[45;113H(B[0;7mM-6(B[m Copy Text   (B[0;7mM-W(B[m WhereIs Next(B[0;7mM-▼(B[m Next[45;161H(B[0;7m^F(B[m Forward[43d[3d[39;49m(B[m[?12l[?25h[?25l[1;175H(B[0;7mModified[43d(B[m[K[1;183H[3dprint("RDA plot saved as RDA_plot_pop_FST_filtered_1000.pdf")[5d[36m#---------------------------------------------------#[6d# Step 6: PCA on Bioclim[7d#---------------------------------------------------#[8d[39m(B[mrownames(bioclim_predictors) <- bioclim_data_filtered$IND[9dpca_bioclim <- prcomp(bioclim_predictors, center = TRUE, scale. = TRUE)[10dpca_scores <- as.data.frame(pca_bioclim$x)[11dpca_scores$IID <- rownames(pca_scores)[12dpca_plot_data <- merge(pca_scores, pop_data_filtered, by = "IID")[14dpca_plot <- ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = POP)) +[15;3Hgeom_point(size = 3) +[16;3Hscale_color_manual([17;5Hvalues = c("POP1" = "cyan", "POP2" = "orange"),[18;5Hlabels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")[19;3H) +[20;3Hlabs(title = "PCA on Bioclimatic Variables", x = "PC1", y = "PC2", color = "Populations") +[21;3Htheme_minimal() +[22;3Htheme(plot.title = element_text(hjust = 0.5))[24dpdf("PCA_plot_bioclim.pdf", width = 8, height = 8)[25dprint(pca_plot)[26ddev.off()[27dprint("PCA plot saved as PCA_plot_bioclim.pdf")[?12l[?25h[?25l[43d(B[0;7mSave modified buffer?  (Answering "No" will DISCARD changes.)                                                                                                                           [44;1H Y(B[m Yes[K[45d(B[0;7m N(B[m No  [45;18H(B[0;7mC(B[m Cancel[K[43;63H[?12l[?25h[?25l[44d(B[0;7m^G(B[m Get Help[44;47H(B[0;7mM-D(B[m DOS Format[44;93H(B[0;7mM-A(B[m Append[44;139H(B[0;7mM-B(B[m Backup File[45d(B[0;7m^C(B[m Cancel	         [45;47H(B[0;7mM-M(B[m Mac Format[45;93H(B[0;7mM-P(B[m Prepend[45;139H(B[0;7m^T(B[m To Files[43d(B[0;7mFile Name to Write: RDAanalysis_2poplabeling_biopca_final_1000.R(B[m[43;65H[?12l[?25h[?25l[K[1;183H[1;175H(B[0;7m        (B[m[43;83H(B[0;7m[ Wrote 205 lines ](B[m[J[45;184H[?12l[?25h[45;1H[?1049l[23;0;0t[?1l>]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ nano RDAanalysis_2poplabeling_biopca_final_1000.R[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[2Prm[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[CRscript RDAanalysis_2poplabeling_biopca_final.R
[?25h[?25hLoading required package: permute
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hWarning message:
Removed 606380 SNPs with invalid data. 
[?25h[?25h[?25hFinal genetic data dimensions: 184 5000 
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hFinal number of individuals for RDA: 184 
[?25h[?25h[?25h[?25h
Call:
rda(formula = genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 +      wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + wc2.1_30s_bio_16, data = as.data.frame(bioclim_scaled)) 

Partitioning of variance:
              Inertia Proportion
Total          1728.3     1.0000
Constrained     474.5     0.2746
Unconstrained  1253.7     0.7254

Eigenvalues, and their contribution to the variance 

Importance of components:
                          RDA1     RDA2     RDA3     RDA4      RDA5       PC1
Eigenvalue            287.3834 72.99167 57.96457 40.19117 16.011282 132.01727
Proportion Explained    0.1663  0.04223  0.03354  0.02326  0.009264   0.07639
Cumulative Proportion   0.1663  0.20852  0.24206  0.26531  0.274577   0.35096
                           PC2      PC3      PC4      PC5      PC6     PC7
Eigenvalue            68.63702 34.74744 33.22067 33.08984 26.11051 23.5047
Proportion Explained   0.03971  0.02011  0.01922  0.01915  0.01511  0.0136
Cumulative Proportion  0.39068  0.41078  0.43001  0.44915  0.46426  0.4779
                           PC8      PC9      PC10     PC11      PC12      PC13
Eigenvalue            22.40181 19.76991 17.223788 15.46855 14.357883 13.210410
Proportion Explained   0.01296  0.01144  0.009966  0.00895  0.008308  0.007644
Cumulative Proportion  0.49082  0.50226  0.512227  0.52118  0.529485  0.537129
                           PC14      PC15      PC16      PC17     PC18     PC19
Eigenvalue            11.387081 11.049344 10.914450 10.566626 10.45658 10.04038
Proportion Explained   0.006589  0.006393  0.006315  0.006114  0.00605  0.00581
Cumulative Proportion  0.543718  0.550111  0.556426  0.562540  0.56859  0.57440
                         PC20     PC21     PC22     PC23     PC24     PC25
Eigenvalue            9.97230 9.881718 9.721613 9.511237 9.464861 9.365120
Proportion Explained  0.00577 0.005718 0.005625 0.005503 0.005477 0.005419
Cumulative Proportion 0.58017 0.585888 0.591513 0.597016 0.602493 0.607912
                          PC26     PC27     PC28     PC29     PC30     PC31
Eigenvalue            9.267460 9.157299 9.058698 9.024429 8.782894 8.769539
Proportion Explained  0.005362 0.005299 0.005241 0.005222 0.005082 0.005074
Cumulative Proportion 0.613274 0.618572 0.623814 0.629036 0.634117 0.639192
                          PC32     PC33     PC34     PC35     PC36     PC37
Eigenvalue            8.614089 8.568338 8.461162 8.415006 8.309412 8.263158
Proportion Explained  0.004984 0.004958 0.004896 0.004869 0.004808 0.004781
Cumulative Proportion 0.644176 0.649134 0.654029 0.658898 0.663706 0.668488
                          PC38     PC39     PC40     PC41     PC42     PC43
Eigenvalue            8.150939 7.979568 7.806575 7.707043 7.633472 7.513000
Proportion Explained  0.004716 0.004617 0.004517 0.004459 0.004417 0.004347
Cumulative Proportion 0.673204 0.677821 0.682338 0.686797 0.691214 0.695561
                          PC44    PC45     PC46     PC47     PC48     PC49
Eigenvalue            7.438580 7.34439 7.285335 7.242679 7.142780 7.100154
Proportion Explained  0.004304 0.00425 0.004215 0.004191 0.004133 0.004108
Cumulative Proportion 0.699865 0.70411 0.708330 0.712521 0.716654 0.720762
                         PC50     PC51    PC52     PC53     PC54     PC55
Eigenvalue            6.98259 6.920141 6.82587 6.685417 6.592252 6.556190
Proportion Explained  0.00404 0.004004 0.00395 0.003868 0.003814 0.003794
Cumulative Proportion 0.72480 0.728807 0.73276 0.736624 0.740439 0.744232
                          PC56     PC57    PC58     PC59     PC60     PC61
Eigenvalue            6.503461 6.428041 6.36026 6.298743 6.275754 6.176311
Proportion Explained  0.003763 0.003719 0.00368 0.003645 0.003631 0.003574
Cumulative Proportion 0.747995 0.751715 0.75539 0.759039 0.762671 0.766244
                          PC62     PC63     PC64     PC65     PC66     PC67
Eigenvalue            6.079672 6.061023 6.015511 5.960295 5.884521 5.867716
Proportion Explained  0.003518 0.003507 0.003481 0.003449 0.003405 0.003395
Cumulative Proportion 0.769762 0.773269 0.776750 0.780198 0.783603 0.786998
                          PC68     PC69     PC70     PC71     PC72     PC73
Eigenvalue            5.827247 5.768638 5.728687 5.641931 5.604376 5.589061
Proportion Explained  0.003372 0.003338 0.003315 0.003265 0.003243 0.003234
Cumulative Proportion 0.790370 0.793708 0.797023 0.800287 0.803530 0.806764
                          PC74     PC75     PC76     PC77     PC78     PC79
Eigenvalue            5.515221 5.499430 5.435899 5.382788 5.348028 5.332334
Proportion Explained  0.003191 0.003182 0.003145 0.003115 0.003094 0.003085
Cumulative Proportion 0.809955 0.813137 0.816282 0.819397 0.822491 0.825577
                          PC80     PC81     PC82     PC83     PC84     PC85
Eigenvalue            5.246660 5.242910 5.209561 5.143179 5.106499 5.093558
Proportion Explained  0.003036 0.003034 0.003014 0.002976 0.002955 0.002947
Cumulative Proportion 0.828613 0.831646 0.834660 0.837636 0.840591 0.843538
                         PC86     PC87     PC88    PC89     PC90     PC91
Eigenvalue            5.04641 4.995962 4.940506 4.92605 4.865434 4.800991
Proportion Explained  0.00292 0.002891 0.002859 0.00285 0.002815 0.002778
Cumulative Proportion 0.84646 0.849349 0.852208 0.85506 0.857873 0.860651
                          PC92     PC93     PC94     PC95     PC96     PC97
Eigenvalue            4.767736 4.693996 4.682146 4.650554 4.628814 4.520155
Proportion Explained  0.002759 0.002716 0.002709 0.002691 0.002678 0.002615
Cumulative Proportion 0.863410 0.866126 0.868835 0.871526 0.874204 0.876819
                          PC98     PC99    PC100    PC101    PC102    PC103
Eigenvalue            4.494495 4.461972 4.435585 4.340237 4.281749 4.271527
Proportion Explained  0.002601 0.002582 0.002566 0.002511 0.002477 0.002472
Cumulative Proportion 0.879420 0.882002 0.884568 0.887080 0.889557 0.892029
                         PC104    PC105    PC106    PC107    PC108    PC109
Eigenvalue            4.236600 4.179379 4.153017 4.107962 4.041052 3.997794
Proportion Explained  0.002451 0.002418 0.002403 0.002377 0.002338 0.002313
Cumulative Proportion 0.894480 0.896898 0.899301 0.901678 0.904016 0.906330
                        PC110    PC111    PC112    PC113    PC114    PC115
Eigenvalue            3.95853 3.918236 3.893411 3.824744 3.811098 3.757017
Proportion Explained  0.00229 0.002267 0.002253 0.002213 0.002205 0.002174
Cumulative Proportion 0.90862 0.910887 0.913140 0.915353 0.917558 0.919732
                         PC116    PC117    PC118    PC119    PC120    PC121
Eigenvalue            3.705231 3.677223 3.651698 3.603525 3.585996 3.575912
Proportion Explained  0.002144 0.002128 0.002113 0.002085 0.002075 0.002069
Cumulative Proportion 0.921876 0.924004 0.926117 0.928202 0.930277 0.932346
                         PC122    PC123    PC124    PC125    PC126    PC127
Eigenvalue            3.502372 3.420447 3.399918 3.356229 3.341255 3.279264
Proportion Explained  0.002027 0.001979 0.001967 0.001942 0.001933 0.001897
Cumulative Proportion 0.934372 0.936351 0.938319 0.940260 0.942194 0.944091
                         PC128    PC129    PC130    PC131    PC132    PC133
Eigenvalue            3.265030 3.217738 3.200059 3.168349 3.112672 3.065762
Proportion Explained  0.001889 0.001862 0.001852 0.001833 0.001801 0.001774
Cumulative Proportion 0.945980 0.947842 0.949694 0.951527 0.953328 0.955102
                         PC134    PC135    PC136    PC137    PC138    PC139
Eigenvalue            3.035911 2.988688 2.974292 2.959113 2.931322 2.900797
Proportion Explained  0.001757 0.001729 0.001721 0.001712 0.001696 0.001678
Cumulative Proportion 0.956859 0.958588 0.960309 0.962021 0.963717 0.965396
                        PC140    PC141    PC142    PC143    PC144    PC145
Eigenvalue            2.85137 2.812277 2.753910 2.729558 2.684295 2.675035
Proportion Explained  0.00165 0.001627 0.001593 0.001579 0.001553 0.001548
Cumulative Proportion 0.96705 0.968673 0.970266 0.971846 0.973399 0.974947
                         PC146    PC147    PC148    PC149    PC150    PC151
Eigenvalue            2.642740 2.554225 2.531984 2.475917 2.266223 2.196927
Proportion Explained  0.001529 0.001478 0.001465 0.001433 0.001311 0.001271
Cumulative Proportion 0.976476 0.977954 0.979419 0.980851 0.982162 0.983434
                         PC152    PC153    PC154     PC155     PC156     PC157
Eigenvalue            2.154027 1.930098 1.739004 1.5966554 1.5121203 1.3572363
Proportion Explained  0.001246 0.001117 0.001006 0.0009238 0.0008749 0.0007853
Cumulative Proportion 0.984680 0.985797 0.986803 0.9877268 0.9886018 0.9893871
                          PC158     PC159     PC160     PC161     PC162
Eigenvalue            1.2594564 1.1289167 1.0555100 1.0460310 1.0040296
Proportion Explained  0.0007287 0.0006532 0.0006107 0.0006052 0.0005809
Cumulative Proportion 0.9901158 0.9907690 0.9913798 0.9919850 0.9925660
                          PC163     PC164    PC165     PC166     PC167
Eigenvalue            1.0029709 0.9569521 0.922972 0.9188095 0.9012801
Proportion Explained  0.0005803 0.0005537 0.000534 0.0005316 0.0005215
Cumulative Proportion 0.9931463 0.9937000 0.994234 0.9947657 0.9952872
                          PC168     PC169     PC170     PC171     PC172
Eigenvalue            0.8730976 0.8375901 0.7965876 0.7697028 0.7590206
Proportion Explained  0.0005052 0.0004846 0.0004609 0.0004454 0.0004392
Cumulative Proportion 0.9957924 0.9962770 0.9967379 0.9971833 0.9976225
                          PC173     PC174     PC175     PC176     PC177
Eigenvalue            0.7385377 0.7257166 0.6976674 0.6694382 0.6466926
Proportion Explained  0.0004273 0.0004199 0.0004037 0.0003873 0.0003742
Cumulative Proportion 0.9980498 0.9984697 0.9988734 0.9992607 0.9996349
                          PC178
Eigenvalue            0.6309561
Proportion Explained  0.0003651
Cumulative Proportion 1.0000000

Accumulated constrained eigenvalues
Importance of components:
                          RDA1    RDA2    RDA3     RDA4     RDA5
Eigenvalue            287.3834 72.9917 57.9646 40.19117 16.01128
Proportion Explained    0.6056  0.1538  0.1221  0.08469  0.03374
Cumulative Proportion   0.6056  0.7594  0.8816  0.96626  1.00000

[?25hPermutation test for rda under reduced model
Permutation: free
Number of permutations: 999

Model: rda(formula = genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 + wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + wc2.1_30s_bio_16, data = as.data.frame(bioclim_scaled))
          Df Variance      F Pr(>F)    
Model      5   474.54 13.475  0.001 ***
Residual 178  1253.72                  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
[?25h^Z
[1]+  Stopped                 Rscript RDAanalysis_2poplabeling_biopca_final.R
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ Rscript RDAanalysis_2poplabeling_biopca_final.R[C[K[K_1000.R 
[?25h[?25hLoading required package: permute
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hWarning message:
Removed 606380 SNPs with invalid data (NAs or Inf). 
[?25h
--- STARTING FST WINDOW FILTERING ---
[?25h[?25h[?25h[?25h[?25h[?25hFST Threshold (Top 100%): -0.8078
[?25hNumber of windows included: 349456
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hNumber of SNPs located in FST windows: 221429
[?25h[?25h[?25hFinal genetic data dimensions for RDA: 184 221429 
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hnull device 
          1 
[?25h[1] "RDA plot saved as RDA_plot_pop_FST_filtered_1000.pdf"
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hnull device 
          1 
[?25h[1] "PCA plot saved as PCA_plot_bioclim.pdf"
[?25h[?25h]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ [K(Renv) [aescudero@nodo4143 RDA]$ [K(Renv) [aescudero@nodo4143 RDA]$ [K(Renv) [aescudero@nodo4143 RDA]$ Rscript RDAanalysis_2poplabeling_biopca_final_1000.R [1P[1P[1P[1P[1P[1P[1P[1@r[1@m
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ rm RDAanalysis_2poplabeling_biopca_final_1000.R [1P[1P[1@n[1@a[1@n[1@o
[?1049h[22;0;0t[1;45r(B[m[4l[?7h[39;49m[?1h=[?1h=[?1h=[?25l[39;49m(B[m[H[2J[43;87H(B[0;7m[ New File ](B[m[H(B[0;7m  GNU nano 2.9.8                                                         RDAanalysis_2poplabeling_biopca_final_1000.R                                                                   [1;183H(B[m[44d(B[0;7m^G(B[m Get Help     (B[0;7m^O(B[m Write Out    (B[0;7m^W(B[m Where Is     (B[0;7m^K(B[m Cut Text     (B[0;7m^J(B[m Justify	(B[0;7m^C(B[m Cur Pos	(B[0;7mM-U(B[m Undo[44;113H(B[0;7mM-A(B[m Mark Text   (B[0;7mM-](B[m To Bracket  (B[0;7mM-▲(B[m Previous    (B[0;7m^B(B[m Back[45d(B[0;7m^X(B[m Exit[45;17H(B[0;7m^R(B[m Read File    (B[0;7m^\(B[m Replace	(B[0;7m^U(B[m Uncut Text   (B[0;7m^T(B[m To Spell     (B[0;7m^_(B[m Go To Line   (B[0;7mM-E(B[m Redo[45;113H(B[0;7mM-6(B[m Copy Text   (B[0;7mM-W(B[m WhereIs Next(B[0;7mM-▼(B[m Next[45;161H(B[0;7m^F(B[m Forward[43d[3d[39;49m(B[m[?12l[?25h[?25l[1;175H(B[0;7mModified[43d(B[m[K[1;183H[4d[36m#---------------------------------------------------#[5d# Step 6: PCA on Bioclim[6d#---------------------------------------------------#[7d[39m(B[mrownames(bioclim_predictors) <- bioclim_data_filtered$IND[8dpca_bioclim <- prcomp(bioclim_predictors, center = TRUE, scale. = TRUE)[9dpca_scores <- as.data.frame(pca_bioclim$x)[10dpca_scores$IID <- rownames(pca_scores)[11dpca_plot_data <- merge(pca_scores, pop_data_filtered, by = "IID")[13dpca_plot <- ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = POP)) +[14;3Hgeom_point(size = 3) +[15;3Hscale_color_manual([16;5Hvalues = c("POP1" = "cyan", "POP2" = "orange"),[17;5Hlabels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")[18;3H) +[19;3Hlabs(title = "PCA on Bioclimatic Variables", x = "PC1", y = "PC2", color = "Populations") +[20;3Htheme_minimal() +[21;3Htheme(plot.title = element_text(hjust = 0.5))[23dpdf("PCA_plot_bioclim.pdf", width = 8, height = 8)[24dprint(pca_plot)[25ddev.off()[26dprint("PCA plot saved as PCA_plot_bioclim.pdf")[?12l[?25h[?25l[43d(B[0;7mSave modified buffer?  (Answering "No" will DISCARD changes.)                                                                                                                           [44;1H Y(B[m Yes[K[45d(B[0;7m N(B[m No  [45;18H(B[0;7mC(B[m Cancel[K[43;63H[?12l[?25h[?25l[44d(B[0;7m^G(B[m Get Help[44;47H(B[0;7mM-D(B[m DOS Format[44;93H(B[0;7mM-A(B[m Append[44;139H(B[0;7mM-B(B[m Backup File[45d(B[0;7m^C(B[m Cancel	         [45;47H(B[0;7mM-M(B[m Mac Format[45;93H(B[0;7mM-P(B[m Prepend[45;139H(B[0;7m^T(B[m To Files[43d(B[0;7mFile Name to Write: RDAanalysis_2poplabeling_biopca_final_1000.R(B[m[43;65H[?12l[?25h[?25l[K[1;183H[1;175H(B[0;7m        (B[m[43;83H(B[0;7m[ Wrote 224 lines ](B[m[J[45;184H[?12l[?25h[45;1H[?1049l[23;0;0t[?1l>]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ nano RDAanalysis_2poplabeling_biopca_final_1000.R [C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[2Prm[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[5@Rscript[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
[?25h[?25hLoading required package: permute
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hTotal SNPs before cleaning: 827809 
[?25h[?25h[?25hWarning message:
Removed 234 SNPs with >50% missing data. 
[?25hImputing remaining missing genotypes with the mean...
[?25h[?25h[?25h[?25hTotal SNPs available after imputation: 827575 
[?25h
--- STARTING FST WINDOW FILTERING ---
[?25h[?25h[?25h[?25h[?25h[?25hFST Threshold (Top 100%): -0.8078
[?25hNumber of windows included: 349456
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hNumber of SNPs mapped to FST windows: 827050
[?25h[?25h[?25h
Subsampling from 827050 down to 5000 SNPs for robust RDA...
[?25hFinal genetic data dimensions for RDA: 184 5000 
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hnull device 
          1 
[?25h[1] "RDA plot saved as RDA_plot_pop_FST_filtered_1000.pdf"
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hnull device 
          1 
[?25h[1] "PCA plot saved as PCA_plot_bioclim.pdf"
[?25h[?25h]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ Rscript RDAanalysis_2poplabeling_biopca_final_1000.R [C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[3Pnano[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[2Prm[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ rm RDAanalysis_2poplabeling_biopca_final_1000.R [C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[5@Rscript[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[3Pnano[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
[?1049h[22;0;0t[1;45r(B[m[4l[?7h[39;49m[?1h=[?1h=[?1h=[?25l[39;49m(B[m[H[2J[43;87H(B[0;7m[ New File ](B[m[H(B[0;7m  GNU nano 2.9.8                                                         RDAanalysis_2poplabeling_biopca_final_1000.R                                                                   [1;183H(B[m[44d(B[0;7m^G(B[m Get Help     (B[0;7m^O(B[m Write Out    (B[0;7m^W(B[m Where Is     (B[0;7m^K(B[m Cut Text     (B[0;7m^J(B[m Justify	(B[0;7m^C(B[m Cur Pos	(B[0;7mM-U(B[m Undo[44;113H(B[0;7mM-A(B[m Mark Text   (B[0;7mM-](B[m To Bracket  (B[0;7mM-▲(B[m Previous    (B[0;7m^B(B[m Back[45d(B[0;7m^X(B[m Exit[45;17H(B[0;7m^R(B[m Read File    (B[0;7m^\(B[m Replace	(B[0;7m^U(B[m Uncut Text   (B[0;7m^T(B[m To Spell     (B[0;7m^_(B[m Go To Line   (B[0;7mM-E(B[m Redo[45;113H(B[0;7mM-6(B[m Copy Text   (B[0;7mM-W(B[m WhereIs Next(B[0;7mM-▼(B[m Next[45;161H(B[0;7m^F(B[m Forward[43d[3d[39;49m(B[m[?12l[?25h[?25l[1;175H(B[0;7mModified[43d(B[m[K[1;183H[3;3Htheme_minimal() +[4;3Htheme(plot.title = element_text(hjust = 0.5))[6dpdf("RDA_plot_pop_FST_filtered_1000.pdf", width = 8, height = 8)[7dprint(rda_plot)[8ddev.off()[9dprint("RDA plot saved as RDA_plot_pop_FST_filtered_1000.pdf")[11d[36m#---------------------------------------------------#[12d# Step 6: PCA on Bioclim[13d#---------------------------------------------------#[14d[39m(B[mrownames(bioclim_predictors) <- bioclim_data_filtered$IND[15dpca_bioclim <- prcomp(bioclim_predictors, center = TRUE, scale. = TRUE)[16dpca_scores <- as.data.frame(pca_bioclim$x)[17dpca_scores$IID <- rownames(pca_scores)[18dpca_plot_data <- merge(pca_scores, pop_data_filtered, by = "IID")[20dpca_plot <- ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = POP)) +[21;3Hgeom_point(size = 3) +[22;3Hscale_color_manual([23;5Hvalues = c("POP1" = "cyan", "POP2" = "orange"),[24;5Hlabels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")[25;3H) +[26;3Hlabs(title = "PCA on Bioclimatic Variables", x = "PC1", y = "PC2", color = "Populations") +[27;3Htheme_minimal() +[28;3Htheme(plot.title = element_text(hjust = 0.5))[30dpdf("PCA_plot_bioclim.pdf", width = 8, height = 8)[31dprint(pca_plot)[32ddev.off()[33dprint("PCA plot saved as PCA_plot_bioclim.pdf")[?12l[?25h[?25l[43d(B[0;7mSave modified buffer?  (Answering "No" will DISCARD changes.)                                                                                                                           [44;1H Y(B[m Yes[K[45d(B[0;7m N(B[m No  [45;18H(B[0;7mC(B[m Cancel[K[43;63H[?12l[?25h[?25l[44d(B[0;7m^G(B[m Get Help[44;47H(B[0;7mM-D(B[m DOS Format[44;93H(B[0;7mM-A(B[m Append[44;139H(B[0;7mM-B(B[m Backup File[45d(B[0;7m^C(B[m Cancel	         [45;47H(B[0;7mM-M(B[m Mac Format[45;93H(B[0;7mM-P(B[m Prepend[45;139H(B[0;7m^T(B[m To Files[43d(B[0;7mFile Name to Write: RDAanalysis_2poplabeling_biopca_final_1000.R(B[m[43;65H[?12l[?25h[?25l[K[1;183H[1;175H(B[0;7m        (B[m[43;83H(B[0;7m[ Wrote 251 lines ](B[m[J[45;184H[?12l[?25h[45;1H[?1049l[23;0;0t[?1l>]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ nano RDAanalysis_2poplabeling_biopca_final_1000.R [C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[2Prm[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[5@Rscript[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
[?25h[?25hLoading required package: permute
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hTotal SNPs before cleaning: 827809 
[?25h[?25h[?25hWarning message:
Removed 234 SNPs with >50% missing data. 
[?25hImputing remaining missing genotypes with the mean...
[?25h[?25h[?25h[?25hTotal SNPs available after imputation: 827575 
[?25h
--- STARTING FST WINDOW FILTERING ---
[?25h[?25h[?25h[?25h[?25h[?25hFST Threshold (Top 100%): -0.8078
[?25hNumber of windows included: 349456
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hNumber of SNPs mapped to FST windows: 827050
[?25h[?25h[?25h
Subsampling from 827050 down to 5000 SNPs for robust RDA...
[?25hFinal genetic data dimensions for RDA: 184 5000 
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h
=======================================================
[?25h                RDA STATISTICAL SUMMARY                
[?25h=======================================================
[?25h[?25h
--- R-squared and Adjusted R-squared ---
[?25h$r.squared
[1] 0.2606158

$adj.r.squared
[1] 0.2398466

[?25h
--- Global ANOVA (Significance of the Model) ---
[?25h[?25hPermutation test for rda under reduced model
Permutation: free
Number of permutations: 999

Model: rda(formula = genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 + wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + wc2.1_30s_bio_16, data = as.data.frame(bioclim_scaled))
          Df Variance      F Pr(>F)    
Model      5   419.64 12.548  0.001 ***
Residual 178  1190.56                  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
[?25h
--- ANOVA by Terms (Marginal Effects of Variables) ---
[?25h[?25hPermutation test for rda under reduced model
Marginal effects of terms
Permutation: free
Number of permutations: 999

Model: rda(formula = genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 + wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + wc2.1_30s_bio_16, data = as.data.frame(bioclim_scaled))
                  Df Variance      F Pr(>F)    
wc2.1_30s_bio_1    1    32.51 4.8603  0.001 ***
wc2.1_30s_bio_4    1    19.38 2.8970  0.001 ***
wc2.1_30s_bio_12   1    41.39 6.1882  0.001 ***
wc2.1_30s_bio_14   1    53.07 7.9347  0.001 ***
wc2.1_30s_bio_16   1    32.88 4.9158  0.001 ***
Residual         178  1190.56                  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
[?25h
--- ANOVA by Axis (Significance of RDA Axes) ---
[?25h[?25hPermutation test for rda under reduced model
Forward tests for axes
Permutation: free
Number of permutations: 999

Model: rda(formula = genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 + wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + wc2.1_30s_bio_16, data = as.data.frame(bioclim_scaled))
          Df Variance       F Pr(>F)    
RDA1       1   250.17 37.4030  0.001 ***
RDA2       1    67.43 10.0816  0.001 ***
RDA3       1    50.72  7.5824  0.001 ***
RDA4       1    36.09  5.3952  0.001 ***
RDA5       1    15.24  2.2786  0.001 ***
Residual 178  1190.56                   
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
[?25h
=======================================================
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hnull device 
          1 
[?25h[1] "RDA plot saved as RDA_plot_pop_FST_filtered_1000.pdf"
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hnull device 
          1 
[?25h[1] "PCA plot saved as PCA_plot_bioclim.pdf"
[?25h[?25h]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ cat [49@Rscript RDAanalysis_2poplabeling_biopca_final_1000.R[C[1P[1P[1P[1P[1P[1P[1P[1@c[1@a[1@t
#---------------------------------------------------#
# Step 1: Install and Load Required Packages
#---------------------------------------------------#
if (!requireNamespace("vegan", quietly = TRUE)) install.packages("vegan", dependencies = TRUE)
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2", dependencies = TRUE)
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table", dependencies = TRUE)

library(vegan)
library(ggplot2)
library(data.table)

# --- CONFIGURATION ---
# Path verified from your Python script
FST_WINDOW_FILE <- "/home/aescudero/wgg/fst_results/windowed_stats/windowed_fst_1000bp.windowed.weir.fst"
# Top % of FST windows to keep (0.00 = Keep ALL windows provided in the file)
FST_QUANTILE <- 0.00
# Target number of SNPs for RDA
TARGET_SNP_COUNT <- 5000
# ---------------------

#---------------------------------------------------#
# Step 2: Load and Clean Data
#---------------------------------------------------#

# Load the bioclimatic data.
bioclim_data <- read.csv("final_df_for_analysis.csv")

# Load the genetic data using fread() for efficiency.
raw_data <- fread("admixture_final_ld_pruned_filtered_for_RDA.raw")

# Load the population data.
tryCatch({
  pop_data <- read.table("population_list.txt", sep=" ", header = TRUE)
  if (is.null(pop_data) || nrow(pop_data) == 0) stop("Population data file is empty.")
}, error = function(e) stop("Error loading population data: ", e$message))

# Format the genetic data: remove non-genotype columns and set row names.
# Columns 1-6 are metadata (FID, IID, PAT, MAT, SEX, PHENOTYPE)
genetic_data_final <- as.data.frame(raw_data[, 7:ncol(raw_data)])
rownames(genetic_data_final) <- raw_data$IID

# --- Clean Genetic Data (Impute instead of Remove) ---
cat("Total SNPs before cleaning:", ncol(genetic_data_final), "\n")

# 1. Calculate missingness
missing_rate <- colMeans(is.na(genetic_data_final))

# 2. Remove SNPs with excessive missing data (> 50%)
excessive_missing_indices <- which(missing_rate > 0.5)
if(length(excessive_missing_indices) > 0) {
    warning("Removed ", length(excessive_missing_indices), " SNPs with >50% missing data.")
    genetic_data_final <- genetic_data_final[, -excessive_missing_indices]
}

# 3. Impute remaining NAs with the mean genotype
cat("Imputing remaining missing genotypes with the mean...\n")
# Using a faster matrix approach for imputation
genetic_mat <- as.matrix(genetic_data_final)
k <- which(is.na(genetic_mat), arr.ind=TRUE)
if(length(k) > 0) {
    genetic_mat[k] <- colMeans(genetic_mat, na.rm=TRUE)[k[,2]]
    genetic_data_final <- as.data.frame(genetic_mat)
}
cat("Total SNPs available after imputation:", ncol(genetic_data_final), "\n")

#---------------------------------------------------#
# Step 2b: Filter SNPs based on 1000bp FST Windows
#---------------------------------------------------#
cat("\n--- STARTING FST WINDOW FILTERING ---\n")

# 1. Load FST Window Data
if (!file.exists(FST_WINDOW_FILE)) stop("FST window file not found.")
fst_windows <- fread(FST_WINDOW_FILE)

# 2. Identify Outlier Windows
# Filter out NAs first
fst_windows <- fst_windows[!is.na(WEIGHTED_FST)]
threshold <- quantile(fst_windows$WEIGHTED_FST, FST_QUANTILE)
outlier_windows <- fst_windows[WEIGHTED_FST >= threshold]

cat(sprintf("FST Threshold (Top %.0f%%): %.4f\n", (1-FST_QUANTILE)*100, threshold))
cat(sprintf("Number of windows included: %d\n", nrow(outlier_windows)))

# 3. Map SNPs to Windows
# Extract SNP names from columns
snp_names <- colnames(genetic_data_final)

# Parse SNP names to get CHR and POS.
snp_info <- data.table(SNP_ID = snp_names)
snp_info[, c("CHROM", "POS_RAW") := tstrsplit(SNP_ID, ":", fixed=TRUE, keep=1:2)]
snp_info[, POS := as.integer(gsub("_.*", "", POS_RAW))]

# Ensure Chromosome names match
outlier_windows[, CHROM := as.character(CHROM)]
snp_info[, CHROM := as.character(CHROM)]

# Prepare for range join (1000bp window)
outlier_windows[, BIN_END := BIN_START + 1000]

# Create explicit End column for SNPs to avoid duplicate column error in foverlaps
snp_info[, POS_END := POS]

# Set keys for foverlaps
setkey(snp_info, CHROM, POS, POS_END)
setkey(outlier_windows, CHROM, BIN_START, BIN_END)

# Perform Overlap Join
overlaps <- foverlaps(snp_info, outlier_windows, type="within", nomatch=0L)
keep_snps <- unique(overlaps$SNP_ID)

cat(sprintf("Number of SNPs mapped to FST windows: %d\n", length(keep_snps)))

if (length(keep_snps) < 50) {
  warning("Very few SNPs retained. Check if chromosome names match.")
}

# 4. Subset the Genetic Data
genetic_data_final <- genetic_data_final[, keep_snps, drop=FALSE]

#---------------------------------------------------#
# Step 2c: Subsample to 5000 SNPs
#---------------------------------------------------#
if (ncol(genetic_data_final) > TARGET_SNP_COUNT) {
    cat(sprintf("\nSubsampling from %d down to %d SNPs for robust RDA...\n", ncol(genetic_data_final), TARGET_SNP_COUNT))
    set.seed(42) # Ensure reproducibility
    keep_indices <- sample(1:ncol(genetic_data_final), TARGET_SNP_COUNT)
    genetic_data_final <- genetic_data_final[, keep_indices, drop=FALSE]
}

cat("Final genetic data dimensions for RDA:", dim(genetic_data_final), "\n")


#---------------------------------------------------#
# Step 3: Align and Filter Data by Individuals
#---------------------------------------------------#
valid_bioclim <- complete.cases(bioclim_data)
valid_genetic <- complete.cases(genetic_data_final)

common_inds <- intersect(bioclim_data$IND[valid_bioclim], rownames(genetic_data_final)[valid_genetic])
common_inds <- intersect(common_inds, pop_data$IID)

if (length(common_inds) == 0) stop("Error: No common individuals found.")

bioclim_data_filtered <- bioclim_data[bioclim_data$IND %in% common_inds, ]
genetic_data_final <- genetic_data_final[common_inds, ]
pop_data_filtered <- pop_data[pop_data$IID %in% common_inds, , drop = FALSE]

# Order data
bioclim_data_filtered <- bioclim_data_filtered[order(bioclim_data_filtered$IND), ]
genetic_data_final <- genetic_data_final[order(rownames(genetic_data_final)), ]
pop_data_filtered <- pop_data_filtered[order(pop_data_filtered$IID), , drop = FALSE]

stopifnot(all(rownames(genetic_data_final) == bioclim_data_filtered$IND))

#---------------------------------------------------#
# Step 4: Run the RDA Analysis
#---------------------------------------------------#
bioclim_predictors <- bioclim_data_filtered[, c(
  "wc2.1_30s_bio_1", "wc2.1_30s_bio_4", "wc2.1_30s_bio_12",
  "wc2.1_30s_bio_14", "wc2.1_30s_bio_16"
)]

bioclim_scaled <- scale(bioclim_predictors)
rda_result <- rda(genetic_data_final ~ ., data = as.data.frame(bioclim_scaled))

#---------------------------------------------------#
# Step 5: Visualize and Print RDA Results
#---------------------------------------------------#
cat("\n=======================================================\n")
cat("                RDA STATISTICAL SUMMARY                \n")
cat("=======================================================\n")

# 1. R-squared (variance explained)
r2 <- RsquareAdj(rda_result)
cat("\n--- R-squared and Adjusted R-squared ---\n")
print(r2)

# 2. Global Significance Test
cat("\n--- Global ANOVA (Significance of the Model) ---\n")
global_anova <- anova.cca(rda_result, permutations = 999)
print(global_anova)

# 3. Variable Significance (Marginal effects)
cat("\n--- ANOVA by Terms (Marginal Effects of Variables) ---\n")
term_anova <- anova.cca(rda_result, by = "margin", permutations = 999)
print(term_anova)

# 4. Axis Significance
cat("\n--- ANOVA by Axis (Significance of RDA Axes) ---\n")
axis_anova <- anova.cca(rda_result, by = "axis", permutations = 999)
print(axis_anova)

cat("\n=======================================================\n")

# Prepare plotting data
site_scores_df <- as.data.frame(scores(rda_result, display = "sites", scaling = 2))
site_scores_df$IID <- rownames(site_scores_df)
env_scores_df <- as.data.frame(scores(rda_result, display = "bp", scaling = 2))
env_scores_df$variables <- rownames(env_scores_df)

plot_data <- merge(site_scores_df, pop_data_filtered, by = "IID")

plot_title <- if(FST_QUANTILE > 0) {
  paste0("RDA on SNPs in FST Windows (Top ", (1-FST_QUANTILE)*100, "%, ", TARGET_SNP_COUNT, " SNPs)")
} else {
  paste0("RDA on All SNPs mapped to FST Windows (Subsampled to ", TARGET_SNP_COUNT, ")")
}

rda_plot <- ggplot() +
  geom_point(data = plot_data, aes(x = RDA1, y = RDA2, color = POP), size = 3) +
  geom_segment(data = env_scores_df, aes(x = 0, y = 0, xend = RDA1, yend = RDA2), 
               arrow = arrow(length = unit(0.2, "cm")), color = "red") +
  geom_text(data = env_scores_df, aes(x = RDA1, y = RDA2, label = variables), 
            hjust = 0, nudge_x = 0.05, color = "red") +
  scale_color_manual(
    values = c("POP1" = "cyan", "POP2" = "orange"),
    labels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")
  ) +
  labs(title = plot_title,
       x = "RDA1", y = "RDA2", color = "Species") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

pdf("RDA_plot_pop_FST_filtered_1000.pdf", width = 8, height = 8)
print(rda_plot)
dev.off()
print("RDA plot saved as RDA_plot_pop_FST_filtered_1000.pdf")

#---------------------------------------------------#
# Step 6: PCA on Bioclim
#---------------------------------------------------#
rownames(bioclim_predictors) <- bioclim_data_filtered$IND
pca_bioclim <- prcomp(bioclim_predictors, center = TRUE, scale. = TRUE)
pca_scores <- as.data.frame(pca_bioclim$x)
pca_scores$IID <- rownames(pca_scores)
pca_plot_data <- merge(pca_scores, pop_data_filtered, by = "IID")

pca_plot <- ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = POP)) +
  geom_point(size = 3) +
  scale_color_manual(
    values = c("POP1" = "cyan", "POP2" = "orange"),
    labels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")
  ) +
  labs(title = "PCA on Bioclimatic Variables", x = "PC1", y = "PC2", color = "Populations") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

pdf("PCA_plot_bioclim.pdf", width = 8, height = 8)
print(pca_plot)
dev.off()
print("PCA plot saved as PCA_plot_bioclim.pdf")
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ cat RDAanalysis_2poplabeling_biopca_final_1000.R [C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[4@Rscrip[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[3Pnano[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
[?1049h[22;0;0t[1;45r(B[m[4l[?7h[39;49m[?1h=[?1h=[?1h=[?25l[39;49m(B[m[H[2J[43;85H(B[0;7m[ Reading File ](B[m[43;84H(B[0;7m[ Read 251 lines ](B[m[H(B[0;7m  GNU nano 2.9.8                                                         RDAanalysis_2poplabeling_biopca_final_1000.R                                                                   [1;183H(B[m[44d(B[0;7m^G(B[m Get Help     (B[0;7m^O(B[m Write Out    (B[0;7m^W(B[m Where Is     (B[0;7m^K(B[m Cut Text     (B[0;7m^J(B[m Justify	(B[0;7m^C(B[m Cur Pos	(B[0;7mM-U(B[m Undo[44;113H(B[0;7mM-A(B[m Mark Text   (B[0;7mM-](B[m To Bracket  (B[0;7mM-▲(B[m Previous    (B[0;7m^B(B[m Back[45d(B[0;7m^X(B[m Exit[45;17H(B[0;7m^R(B[m Read File    (B[0;7m^\(B[m Replace	(B[0;7m^U(B[m Uncut Text   (B[0;7m^T(B[m To Spell     (B[0;7m^_(B[m Go To Line   (B[0;7mM-E(B[m Redo[45;113H(B[0;7mM-6(B[m Copy Text   (B[0;7mM-W(B[m WhereIs Next(B[0;7mM-▼(B[m Next[45;161H(B[0;7m^F(B[m Forward[43d[3d[39;49m[36m#---------------------------------------------------#[4d# Step 1: Install and Load Required Packages[5d#---------------------------------------------------#[6d[39m(B[mif (!requireNamespace("vegan", quietly = TRUE)) install.packages("vegan", dependencies = TRUE)[7dif (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2", dependencies = TRUE)[8dif (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table", dependencies = TRUE)[10dlibrary(vegan)[11dlibrary(ggplot2)[12dlibrary(data.table)[14d[36m# --- CONFIGURATION ---[15d# Path verified from your Python script[16d[39m(B[mFST_WINDOW_FILE <- "/home/aescudero/wgg/fst_results/windowed_stats/windowed_fst_1000bp.windowed.weir.fst"[17d[36m# Top % of FST windows to keep (0.00 = Keep ALL windows provided in the file)[18d[39m(B[mFST_QUANTILE <- 0.00[19d[36m# Target number of SNPs for RDA[20d[39m(B[mTARGET_SNP_COUNT <- 5000[21d[36m# ---------------------[23d#---------------------------------------------------#[24d# Step 2: Load and Clean Data[25d#---------------------------------------------------#[27d# Load the bioclimatic data.[28d[39m(B[mbioclim_data <- read.csv("final_df_for_analysis.csv")[30d[36m# Load the genetic data using fread() for efficiency.[31d[39m(B[mraw_data <- fread("admixture_final_ld_pruned_filtered_for_RDA.raw")[33d[36m# Load the population data.[34d[39m(B[mtryCatch({[35;3Hpop_data <- read.table("population_list.txt", sep=" ", header = TRUE)[36;3Hif (is.null(pop_data) || nrow(pop_data) == 0) stop("Population data file is empty.")[37d}, error = function(e) stop("Error loading population data: ", e$message))[39d[36m# Format the genetic data: remove non-genotype columns and set row names.[40d# Columns 1-6 are metadata (FID, IID, PAT, MAT, SEX, PHENOTYPE)[41d[39m(B[mgenetic_data_final <- as.data.frame(raw_data[, 7:ncol(raw_data)])[42drownames(genetic_data_final) <- raw_data$IID[3d[?12l[?25h[?25l[43d[J[45;184H[?12l[?25h[45;1H[?1049l[23;0;0t[?1l>]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ nano RDAanalysis_2poplabeling_biopca_final_1000.R [1@_[1@p[1@r[1@u[1@e[1@b[1@a
[?1049h[22;0;0t[1;45r(B[m[4l[?7h[39;49m[?1h=[?1h=[?1h=[?25l[39;49m(B[m[H[2J[43;87H(B[0;7m[ New File ](B[m[H(B[0;7m  GNU nano 2.9.8                                                     RDAanalysis_2poplabeling_biopca_final_1000_prueba.R                                                                [1;183H(B[m[44d(B[0;7m^G(B[m Get Help     (B[0;7m^O(B[m Write Out    (B[0;7m^W(B[m Where Is     (B[0;7m^K(B[m Cut Text     (B[0;7m^J(B[m Justify	(B[0;7m^C(B[m Cur Pos	(B[0;7mM-U(B[m Undo[44;113H(B[0;7mM-A(B[m Mark Text   (B[0;7mM-](B[m To Bracket  (B[0;7mM-▲(B[m Previous    (B[0;7m^B(B[m Back[45d(B[0;7m^X(B[m Exit[45;17H(B[0;7m^R(B[m Read File    (B[0;7m^\(B[m Replace	(B[0;7m^U(B[m Uncut Text   (B[0;7m^T(B[m To Spell     (B[0;7m^_(B[m Go To Line   (B[0;7mM-E(B[m Redo[45;113H(B[0;7mM-6(B[m Copy Text   (B[0;7mM-W(B[m WhereIs Next(B[0;7mM-▼(B[m Next[45;161H(B[0;7m^F(B[m Forward[43d[3d[39;49m(B[m[?12l[?25h[?25l[1;175H(B[0;7mModified[43d(B[m[K[1;183H[3dsite_scores_df$IID <- rownames(site_scores_df)[4denv_scores_df <- as.data.frame(scores(rda_result, display = "bp", scaling = 2))[5denv_scores_df$variables <- rownames(env_scores_df)[7dplot_data <- merge(site_scores_df, pop_data_filtered, by = "IID")[9dplot_title <- if(FST_QUANTILE > 0) {[10;3Hpaste0("RDA on SNPs in FST Windows (Top ", (1-FST_QUANTILE)*100, "%, ", TARGET_SNP_COUNT, " SNPs)")[11d} else {[12;3Hpaste0("RDA on All SNPs mapped to FST Windows (Subsampled to ", TARGET_SNP_COUNT, ")")[13d}[15drda_plot <- ggplot() +[16;3Hgeom_point(data = plot_data, aes(x = RDA1, y = RDA2, color = POP), size = 3) +[17;3Hgeom_segment(data = env_scores_df, aes(x = 0, y = 0, xend = RDA1, yend = RDA2),[18;16Harrow = arrow(length = unit(0.2, "cm")), color = "red") +[19;3Hgeom_text(data = env_scores_df, aes(x = RDA1, y = RDA2, label = variables),[20;13Hhjust = 0, nudge_x = 0.05, color = "red") +[21;3Hscale_color_manual([22;5Hvalues = c("POP1" = "cyan", "POP2" = "orange"),[23;5Hlabels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")[24;3H) +[25;3Hlabs(title = plot_title,[26;8Hx = "RDA1", y = "RDA2", color = "Species") +[27;3Htheme_minimal() +[28;3Htheme(plot.title = element_text(hjust = 0.5))[30dpdf("RDA_plot_pop_FST_filtered_1000.pdf", width = [?12l[?25h[?25l[21;30r[21;1H[5T[1;45r[3;1H    labels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")[4d  ) +[K[5d  labs(title = plot_title,[K[6;8Hx = "RDA1", y = "RDA2", color = "Species") +[7d  theme_minimal() +[K[8;3Htheme(plot.title = element_text(hjust = 0.5))[9d[K[10dpdf("RDA_plot_pop_FST_filtered_1000.pdf", width = 8, height = 8)[K[11dprint(rda_plot)[12ddev.off()[K[13dprint("RDA plot saved as RDA_plot_pop_FST_filtered_1000.pdf")[15d[36m#---------------------------------------------------#[16d# Step 6: PCA on Bioclim[39m(B[m[K[17d[36m#---------------------------------------------------#[39m(B[m[K[18drownames(bioclim_predictors) <- bioclim_data_filtered$IND[K[19dpca_bioclim <- prcomp(bioclim_predictors, center = TRUE, scale. = TRUE)[K[20dpca_scores <- as.data.frame(pca_bioclim$x)[K[21dpca_scores$IID <- rownames(pca_scores)[22dpca_plot_data <- merge(pca_scores, pop_data_filtered, by = "IID")[24dpca_plot <- ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = POP)) +[25;3Hgeom_point(size = 3) +[30;16H"PCA on Bioclimatic Variables", x = "PC1", y = "PC2", color = "Populations") +[31;3Htheme_minimal() +[32;3Htheme(plot.title = element_text(hjust = 0.5))[34dpdf("PCA_plot_bioclim.pdf", width = 8, height = 8)[35dprint(pca_plot)[36ddev.off()[37dprint("PCA plot saved as PCA_plot_bioclim.pdf")[?12l[?25h[?25l[43d(B[0;7mSave modified buffer?  (Answering "No" will DISCARD changes.)                                                                                                                           [44;1H Y(B[m Yes[K[45d(B[0;7m N(B[m No  [45;18H(B[0;7mC(B[m Cancel[K[43;63H[?12l[?25h[?25l[44d(B[0;7m^G(B[m Get Help[44;47H(B[0;7mM-D(B[m DOS Format[44;93H(B[0;7mM-A(B[m Append[44;139H(B[0;7mM-B(B[m Backup File[45d(B[0;7m^C(B[m Cancel	         [45;47H(B[0;7mM-M(B[m Mac Format[45;93H(B[0;7mM-P(B[m Prepend[45;139H(B[0;7m^T(B[m To Files[43d(B[0;7mFile Name to Write: RDAanalysis_2poplabeling_biopca_final_1000_prueba.R(B[m[43;72H[?12l[?25h[?25l[K[1;183H[1;175H(B[0;7m        (B[m[43;83H(B[0;7m[ Wrote 255 lines ](B[m[J[45;184H[?12l[?25h[45;1H[?1049l[23;0;0t[?1l>]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ nano RDAanalysis_2poplabeling_biopca_final_1000_prueba.R [1P[1P[1P[1P[1@R[1@s[1@c[1@r[1@i[1@p[1@t
[?25h[?25hLoading required package: permute
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hTotal SNPs before cleaning: 827809 
[?25h[?25h[?25hWarning message:
Removed 234 SNPs with >50% missing data. 
[?25hImputing remaining missing genotypes with the mean...
[?25h[?25h[?25h[?25hTotal SNPs available after imputation: 827575 
[?25h
--- STARTING FST WINDOW FILTERING ---
[?25h[?25h[?25h[?25h[?25h[?25hFST Threshold (Top 100%): -0.8078
[?25hNumber of windows included: 349456
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hNumber of SNPs mapped to FST windows: 827050
[?25h[?25h[?25h
Subsampling from 827050 down to 5000 SNPs for robust RDA...
[?25hFinal genetic data dimensions for RDA: 184 5000 
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h
=======================================================
[?25h                RDA STATISTICAL SUMMARY                
[?25h=======================================================
[?25h
--- Full RDA Summary (Variance & Eigenvalues) ---
[?25h
Call:
rda(formula = genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 +      wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + wc2.1_30s_bio_16, data = as.data.frame(bioclim_scaled)) 

Partitioning of variance:
              Inertia Proportion
Total          1610.2     1.0000
Constrained     419.6     0.2606
Unconstrained  1190.6     0.7394

Eigenvalues, and their contribution to the variance 

Importance of components:
                          RDA1     RDA2    RDA3     RDA4      RDA5       PC1
Eigenvalue            250.1708 67.43088 50.7154 36.08624 15.240440 110.06971
Proportion Explained    0.1554  0.04188  0.0315  0.02241  0.009465   0.06836
Cumulative Proportion   0.1554  0.19724  0.2287  0.25115  0.260616   0.32897
                           PC2      PC3      PC4      PC5      PC6      PC7
Eigenvalue            60.50533 32.39542 30.47298 28.26827 23.05517 21.31022
Proportion Explained   0.03758  0.02012  0.01892  0.01756  0.01432  0.01323
Cumulative Proportion  0.36655  0.38667  0.40559  0.42315  0.43747  0.45070
                           PC8      PC9    PC10      PC11      PC12      PC13
Eigenvalue            20.23089 17.21149 16.2652 14.624000 13.937523 11.797433
Proportion Explained   0.01256  0.01069  0.0101  0.009082  0.008656  0.007327
Cumulative Proportion  0.46327  0.47396  0.4841  0.493139  0.501795  0.509121
                           PC14      PC15      PC16      PC17     PC18     PC19
Eigenvalue            11.139994 10.600975 10.404805 10.060370 9.856439 9.751179
Proportion Explained   0.006918  0.006584  0.006462  0.006248 0.006121 0.006056
Cumulative Proportion  0.516040  0.522623  0.529085  0.535333 0.541454 0.547510
                          PC20     PC21    PC22     PC23     PC24    PC25
Eigenvalue            9.619748 9.531348 9.43543 9.297619 9.141480 8.93643
Proportion Explained  0.005974 0.005919 0.00586 0.005774 0.005677 0.00555
Cumulative Proportion 0.553484 0.559404 0.56526 0.571038 0.576715 0.58226
                         PC26     PC27    PC28     PC29     PC30     PC31
Eigenvalue            8.87254 8.772482 8.61421 8.537763 8.433307 8.335507
Proportion Explained  0.00551 0.005448 0.00535 0.005302 0.005237 0.005177
Cumulative Proportion 0.58778 0.593223 0.59857 0.603875 0.609113 0.614289
                          PC32     PC33    PC34     PC35     PC36     PC37
Eigenvalue            8.183546 8.064192 8.00308 7.927435 7.916957 7.783452
Proportion Explained  0.005082 0.005008 0.00497 0.004923 0.004917 0.004834
Cumulative Proportion 0.619372 0.624380 0.62935 0.634273 0.639190 0.644024
                          PC38     PC39     PC40     PC41     PC42     PC43
Eigenvalue            7.697981 7.634837 7.546954 7.478300 7.401088 7.353076
Proportion Explained  0.004781 0.004742 0.004687 0.004644 0.004596 0.004567
Cumulative Proportion 0.648805 0.653546 0.658233 0.662877 0.667474 0.672040
                          PC44     PC45     PC46     PC47     PC48     PC49
Eigenvalue            7.177124 7.092761 7.023404 6.952631 6.867997 6.841025
Proportion Explained  0.004457 0.004405 0.004362 0.004318 0.004265 0.004249
Cumulative Proportion 0.676498 0.680903 0.685264 0.689582 0.693848 0.698096
                          PC50     PC51     PC52     PC53     PC54     PC55
Eigenvalue            6.732654 6.679061 6.631280 6.566573 6.550172 6.481922
Proportion Explained  0.004181 0.004148 0.004118 0.004078 0.004068 0.004026
Cumulative Proportion 0.702277 0.706425 0.710544 0.714622 0.718690 0.722715
                          PC56     PC57     PC58     PC59     PC60     PC61
Eigenvalue            6.398931 6.361734 6.267764 6.260165 6.178040 6.104363
Proportion Explained  0.003974 0.003951 0.003893 0.003888 0.003837 0.003791
Cumulative Proportion 0.726689 0.730640 0.734533 0.738420 0.742257 0.746048
                          PC62     PC63     PC64     PC65     PC66     PC67
Eigenvalue            6.068597 5.951473 5.873373 5.805017 5.781500 5.755572
Proportion Explained  0.003769 0.003696 0.003648 0.003605 0.003591 0.003574
Cumulative Proportion 0.749817 0.753513 0.757161 0.760766 0.764357 0.767931
                          PC68     PC69     PC70     PC71     PC72     PC73
Eigenvalue            5.740036 5.642693 5.600205 5.550116 5.499125 5.476890
Proportion Explained  0.003565 0.003504 0.003478 0.003447 0.003415 0.003401
Cumulative Proportion 0.771496 0.775000 0.778478 0.781925 0.785340 0.788742
                        PC74     PC75     PC76    PC77    PC78     PC79
Eigenvalue            5.4743 5.446741 5.395596 5.34573 5.29806 5.214649
Proportion Explained  0.0034 0.003383 0.003351 0.00332 0.00329 0.003239
Cumulative Proportion 0.7921 0.795524 0.798875 0.80219 0.80549 0.808724
                          PC80     PC81     PC82     PC83     PC84    PC85
Eigenvalue            5.181532 5.064167 5.051668 4.985584 4.966871 4.95865
Proportion Explained  0.003218 0.003145 0.003137 0.003096 0.003085 0.00308
Cumulative Proportion 0.811941 0.815087 0.818224 0.821320 0.824405 0.82748
                          PC86     PC87     PC88     PC89     PC90    PC91
Eigenvalue            4.901852 4.867111 4.854686 4.777491 4.761461 4.71866
Proportion Explained  0.003044 0.003023 0.003015 0.002967 0.002957 0.00293
Cumulative Proportion 0.830528 0.833551 0.836566 0.839533 0.842490 0.84542
                          PC92     PC93    PC94     PC95     PC96     PC97
Eigenvalue            4.675636 4.647306 4.58937 4.582254 4.537837 4.522944
Proportion Explained  0.002904 0.002886 0.00285 0.002846 0.002818 0.002809
Cumulative Proportion 0.848324 0.851211 0.85406 0.856907 0.859725 0.862534
                          PC98     PC99   PC100    PC101    PC102    PC103
Eigenvalue            4.484079 4.453589 4.37934 4.320634 4.310629 4.260718
Proportion Explained  0.002785 0.002766 0.00272 0.002683 0.002677 0.002646
Cumulative Proportion 0.865318 0.868084 0.87080 0.873487 0.876164 0.878810
                         PC104    PC105    PC106    PC107    PC108    PC109
Eigenvalue            4.203775 4.162152 4.112489 4.082856 4.032828 4.006761
Proportion Explained  0.002611 0.002585 0.002554 0.002536 0.002505 0.002488
Cumulative Proportion 0.881421 0.884006 0.886560 0.889096 0.891600 0.894089
                         PC110    PC111    PC112   PC113    PC114    PC115
Eigenvalue            3.934961 3.901611 3.834992 3.81670 3.787083 3.769899
Proportion Explained  0.002444 0.002423 0.002382 0.00237 0.002352 0.002341
Cumulative Proportion 0.896532 0.898955 0.901337 0.90371 0.906059 0.908401
                         PC116    PC117    PC118   PC119    PC120    PC121
Eigenvalue            3.725993 3.705098 3.644613 3.59022 3.562471 3.548989
Proportion Explained  0.002314 0.002301 0.002263 0.00223 0.002212 0.002204
Cumulative Proportion 0.910715 0.913016 0.915279 0.91751 0.919721 0.921925
                         PC122    PC123   PC124    PC125    PC126    PC127
Eigenvalue            3.506211 3.476173 3.43040 3.400228 3.370002 3.340813
Proportion Explained  0.002177 0.002159 0.00213 0.002112 0.002093 0.002075
Cumulative Proportion 0.924103 0.926262 0.92839 0.930504 0.932597 0.934671
                         PC128    PC129   PC130    PC131    PC132    PC133
Eigenvalue            3.240909 3.228681 3.18806 3.147743 3.134908 3.117615
Proportion Explained  0.002013 0.002005 0.00198 0.001955 0.001947 0.001936
Cumulative Proportion 0.936684 0.938689 0.94067 0.942624 0.944571 0.946507
                         PC134    PC135    PC136    PC137    PC138    PC139
Eigenvalue            3.081647 3.033140 2.991546 2.973204 2.940634 2.916822
Proportion Explained  0.001914 0.001884 0.001858 0.001846 0.001826 0.001811
Cumulative Proportion 0.948421 0.950305 0.952163 0.954009 0.955835 0.957647
                         PC140    PC141    PC142    PC143    PC144   PC145
Eigenvalue            2.906776 2.868613 2.824654 2.788119 2.760921 2.68938
Proportion Explained  0.001805 0.001782 0.001754 0.001732 0.001715 0.00167
Cumulative Proportion 0.959452 0.961234 0.962988 0.964719 0.966434 0.96810
                         PC146    PC147    PC148    PC149   PC150    PC151
Eigenvalue            2.679192 2.621829 2.595383 2.477930 2.33447 2.246768
Proportion Explained  0.001664 0.001628 0.001612 0.001539 0.00145 0.001395
Cumulative Proportion 0.969768 0.971396 0.973008 0.974547 0.97600 0.977392
                         PC152    PC153    PC154    PC155    PC156    PC157
Eigenvalue            2.165917 2.138448 2.070658 1.889978 1.805589 1.635597
Proportion Explained  0.001345 0.001328 0.001286 0.001174 0.001121 0.001016
Cumulative Proportion 0.978737 0.980065 0.981351 0.982525 0.983646 0.984662
                          PC158     PC159     PC160     PC161     PC162
Eigenvalue            1.5646522 1.4771786 1.4230382 1.4141022 1.3921217
Proportion Explained  0.0009717 0.0009174 0.0008838 0.0008782 0.0008646
Cumulative Proportion 0.9856339 0.9865513 0.9874350 0.9883132 0.9891778
                          PC163     PC164     PC165     PC166     PC167
Eigenvalue            1.3546720 1.3174424 1.2710753 1.2343385 1.2135518
Proportion Explained  0.0008413 0.0008182 0.0007894 0.0007666 0.0007537
Cumulative Proportion 0.9900191 0.9908373 0.9916267 0.9923933 0.9931469
                          PC168     PC169     PC170     PC171     PC172
Eigenvalue            1.1831413 1.1599219 1.1358243 1.0808949 1.0284723
Proportion Explained  0.0007348 0.0007204 0.0007054 0.0006713 0.0006387
Cumulative Proportion 0.9938817 0.9946021 0.9953075 0.9959787 0.9966175
                          PC173     PC174     PC175     PC176     PC177
Eigenvalue            0.9908012 0.9514622 0.9347594 0.9087811 0.8497425
Proportion Explained  0.0006153 0.0005909 0.0005805 0.0005644 0.0005277
Cumulative Proportion 0.9972328 0.9978237 0.9984042 0.9989686 0.9994963
                          PC178
Eigenvalue            0.8110290
Proportion Explained  0.0005037
Cumulative Proportion 1.0000000

Accumulated constrained eigenvalues
Importance of components:
                          RDA1    RDA2    RDA3     RDA4     RDA5
Eigenvalue            250.1708 67.4309 50.7154 36.08624 15.24044
Proportion Explained    0.5962  0.1607  0.1209  0.08599  0.03632
Cumulative Proportion   0.5962  0.7568  0.8777  0.96368  1.00000

[?25h[?25h
--- R-squared and Adjusted R-squared ---
[?25h$r.squared
[1] 0.2606158

$adj.r.squared
[1] 0.2398466

[?25h
--- Global ANOVA (Significance of the Model) ---
[?25h[?25hPermutation test for rda under reduced model
Permutation: free
Number of permutations: 999

Model: rda(formula = genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 + wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + wc2.1_30s_bio_16, data = as.data.frame(bioclim_scaled))
          Df Variance      F Pr(>F)    
Model      5   419.64 12.548  0.001 ***
Residual 178  1190.56                  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
[?25h
--- ANOVA by Terms (Marginal Effects of Variables) ---
[?25h[?25hPermutation test for rda under reduced model
Marginal effects of terms
Permutation: free
Number of permutations: 999

Model: rda(formula = genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 + wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + wc2.1_30s_bio_16, data = as.data.frame(bioclim_scaled))
                  Df Variance      F Pr(>F)    
wc2.1_30s_bio_1    1    32.51 4.8603  0.001 ***
wc2.1_30s_bio_4    1    19.38 2.8970  0.001 ***
wc2.1_30s_bio_12   1    41.39 6.1882  0.001 ***
wc2.1_30s_bio_14   1    53.07 7.9347  0.001 ***
wc2.1_30s_bio_16   1    32.88 4.9158  0.001 ***
Residual         178  1190.56                  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
[?25h
--- ANOVA by Axis (Significance of RDA Axes) ---
[?25h^Z
[2]+  Stopped                 Rscript RDAanalysis_2poplabeling_biopca_final_1000_prueba.R
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ Rscript RDAanalysis_2poplabeling_biopca_final_1000_prueba.R [1P[1P[1P[1P[1P[1P[1P[1@,[1@v[1P[1P[1@m[1@v[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C R A D [1P [1P D A a nalysis_ 2 poplabeling_biopca _ F [1P f i nal _ 1000 . R 
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ mv RDAanalysis_2poplabeling_biopca_final_1000_prueba.R RDAanalysis_2poplabeling_biopca_final_1000.R [C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[40PRscript RDAanalysis_2poplabeling_biopca_final_1000_prueba[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[3Pnano[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C
[?1049h[22;0;0t[1;45r(B[m[4l[?7h[39;49m[?1h=[?1h=[?1h=[?25l[39;49m(B[m[H[2J[43;87H(B[0;7m[ New File ](B[m[H(B[0;7m  GNU nano 2.9.8                                                     RDAanalysis_2poplabeling_biopca_final_1000_prueba.R                                                                [1;183H(B[m[44d(B[0;7m^G(B[m Get Help     (B[0;7m^O(B[m Write Out    (B[0;7m^W(B[m Where Is     (B[0;7m^K(B[m Cut Text     (B[0;7m^J(B[m Justify	(B[0;7m^C(B[m Cur Pos	(B[0;7mM-U(B[m Undo[44;113H(B[0;7mM-A(B[m Mark Text   (B[0;7mM-](B[m To Bracket  (B[0;7mM-▲(B[m Previous    (B[0;7m^B(B[m Back[45d(B[0;7m^X(B[m Exit[45;17H(B[0;7m^R(B[m Read File    (B[0;7m^\(B[m Replace	(B[0;7m^U(B[m Uncut Text   (B[0;7m^T(B[m To Spell     (B[0;7m^_(B[m Go To Line   (B[0;7mM-E(B[m Redo[45;113H(B[0;7mM-6(B[m Copy Text   (B[0;7mM-W(B[m WhereIs Next(B[0;7mM-▼(B[m Next[45;161H(B[0;7m^F(B[m Forward[43d[3d[39;49m(B[m[?12l[?25h[?25l[1;175H(B[0;7mModified[43d(B[m[K[1;183H[4d[36m#---------------------------------------------------#[5d# Step 6: PCA on Bioclim[6d#---------------------------------------------------#[7d[39m(B[mrownames(bioclim_predictors) <- bioclim_data_filtered$IND[8dpca_bioclim <- prcomp(bioclim_predictors, center = TRUE, scale. = TRUE)[9dpca_scores <- as.data.frame(pca_bioclim$x)[10dpca_scores$IID <- rownames(pca_scores)[11dpca_plot_data <- merge(pca_scores, pop_data_filtered, by = "IID")[13dpca_plot <- ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = POP)) +[14;3Hgeom_point(size = 3) +[15;3Hscale_color_manual([16;5Hvalues = c("POP1" = "cyan", "POP2" = "orange"),[17;5Hlabels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")[18;3H) +[19;3Hlabs(title = "PCA on Bioclimatic Variables", x = "PC1", y = "PC2", color = "Populations") +[20;3Htheme_minimal() +[21;3Htheme(plot.title = element_text(hjust = 0.5))[23dpdf("PCA_plot_bioclim.pdf", width = 8, height = 8)[24dprint(pca_plot)[25ddev.off()[26dprint("PCA plot saved as PCA_plot_bioclim.pdf")[?12l[?25h[?25l[43;62H(B[0;7m[ line 244/245 (99%), col 48/48 (100%), char 9810/9811 (99%) ](B[m[26;48H[?12l[?25h[?25l[43d(B[0;7mSave modified buffer?  (Answering "No" will DISCARD changes.)                                                                                                                           [44;1H Y(B[m Yes[K[45d(B[0;7m N(B[m No  [45;18H(B[0;7mC(B[m Cancel[K[43;63H[?12l[?25h[?25l[44d(B[0;7m^G(B[m Get Help[44;47H(B[0;7mM-D(B[m DOS Format[44;93H(B[0;7mM-A(B[m Append[44;139H(B[0;7mM-B(B[m Backup File[45d(B[0;7m^C(B[m Cancel	         [45;47H(B[0;7mM-M(B[m Mac Format[45;93H(B[0;7mM-P(B[m Prepend[45;139H(B[0;7m^T(B[m To Files[43d(B[0;7mFile Name to Write: RDAanalysis_2poplabeling_biopca_final_1000_prueba.R(B[m[43;72H[?12l[?25h[?25l[K[1;183H[1;175H(B[0;7m        (B[m[43;83H(B[0;7m[ Wrote 244 lines ](B[m[J[45;184H[?12l[?25h[45;1H[?1049l[23;0;0t[?1l>]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ nano RDAanalysis_2poplabeling_biopca_final_1000_prueba.R [1P[1P[1P[1P[1@R[1@s[1@c[1@r[1@i[1@p[1@t
[?25h[?25hLoading required package: permute
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hTotal SNPs before cleaning: 827809 
[?25h[?25hWarning message:
Removed 606380 SNPs with invalid data (NAs or Inf). 
[?25hTotal SNPs available after strict filtering: 221429 
[?25h
--- STARTING FST WINDOW FILTERING ---
[?25h[?25h[?25h[?25h[?25h[?25hFST Threshold (Top 100%): -0.8078
[?25hNumber of windows included: 349456
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25hNumber of SNPs mapped to FST windows: 221429
[?25h[?25h[?25h
Subsampling from 221429 down to 5000 SNPs for robust RDA...
[?25hFinal genetic data dimensions for RDA: 184 5000 
[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h[?25h
=======================================================
[?25h                RDA STATISTICAL SUMMARY                
[?25h=======================================================
[?25h
--- Full RDA Summary (Variance & Eigenvalues) ---
[?25h
Call:
rda(formula = genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 +      wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + wc2.1_30s_bio_16, data = as.data.frame(bioclim_scaled)) 

Partitioning of variance:
              Inertia Proportion
Total          1711.0     1.0000
Constrained     476.7     0.2786
Unconstrained  1234.3     0.7214

Eigenvalues, and their contribution to the variance 

Importance of components:
                          RDA1     RDA2     RDA3     RDA4      RDA5       PC1
Eigenvalue            289.8859 72.96403 56.40118 41.64097 15.814397 127.91144
Proportion Explained    0.1694  0.04264  0.03296  0.02434  0.009243   0.07476
Cumulative Proportion   0.1694  0.21207  0.24503  0.26937  0.278615   0.35337
                           PC2      PC3      PC4      PC5      PC6      PC7
Eigenvalue            65.28356 35.86614 34.62473 32.97728 24.73518 24.11112
Proportion Explained   0.03816  0.02096  0.02024  0.01927  0.01446  0.01409
Cumulative Proportion  0.39153  0.41249  0.43273  0.45200  0.46646  0.48055
                           PC8      PC9      PC10      PC11      PC12      PC13
Eigenvalue            21.43736 19.37365 16.722655 15.584531 14.335744 12.805966
Proportion Explained   0.01253  0.01132  0.009774  0.009109  0.008379  0.007485
Cumulative Proportion  0.49308  0.50440  0.514177  0.523286  0.531664  0.539149
                           PC14      PC15      PC16      PC17      PC18
Eigenvalue            11.716679 11.213263 10.969022 10.343820 10.236294
Proportion Explained   0.006848  0.006554  0.006411  0.006046  0.005983
Cumulative Proportion  0.545997  0.552550  0.558961  0.565007  0.570990
                           PC19      PC20     PC21     PC22     PC23     PC24
Eigenvalue            10.142850 10.069116 9.833736 9.771575 9.687779 9.606260
Proportion Explained   0.005928  0.005885 0.005747 0.005711 0.005662 0.005614
Cumulative Proportion  0.576918  0.582803 0.588550 0.594261 0.599923 0.605538
                          PC25    PC26     PC27     PC28     PC29     PC30
Eigenvalue            9.465073 9.25601 9.132904 9.099788 8.957430 8.936403
Proportion Explained  0.005532 0.00541 0.005338 0.005318 0.005235 0.005223
Cumulative Proportion 0.611070 0.61648 0.621817 0.627136 0.632371 0.637594
                          PC31     PC32     PC33     PC34     PC35     PC36
Eigenvalue            8.807540 8.618630 8.430851 8.374262 8.158255 8.114151
Proportion Explained  0.005148 0.005037 0.004927 0.004894 0.004768 0.004742
Cumulative Proportion 0.642741 0.647779 0.652706 0.657601 0.662369 0.667111
                          PC37     PC38     PC39     PC40     PC41     PC42
Eigenvalue            8.032637 7.962224 7.878602 7.771506 7.704442 7.605191
Proportion Explained  0.004695 0.004654 0.004605 0.004542 0.004503 0.004445
Cumulative Proportion 0.671806 0.676459 0.681064 0.685606 0.690109 0.694554
                          PC43     PC44     PC45     PC46     PC47     PC48
Eigenvalue            7.583066 7.426912 7.275652 7.266816 7.112512 7.050153
Proportion Explained  0.004432 0.004341 0.004252 0.004247 0.004157 0.004121
Cumulative Proportion 0.698986 0.703327 0.707579 0.711826 0.715983 0.720104
                          PC49     PC50     PC51     PC52     PC53     PC54
Eigenvalue            6.823421 6.785431 6.761496 6.656755 6.613702 6.525133
Proportion Explained  0.003988 0.003966 0.003952 0.003891 0.003865 0.003814
Cumulative Proportion 0.724092 0.728058 0.732009 0.735900 0.739765 0.743579
                          PC55     PC56     PC57     PC58     PC59     PC60
Eigenvalue            6.444863 6.429149 6.369260 6.280514 6.251555 6.137187
Proportion Explained  0.003767 0.003758 0.003723 0.003671 0.003654 0.003587
Cumulative Proportion 0.747346 0.751103 0.754826 0.758497 0.762150 0.765737
                          PC61    PC62     PC63     PC64     PC65     PC66
Eigenvalue            6.084618 6.07468 6.009430 5.945211 5.887692 5.854936
Proportion Explained  0.003556 0.00355 0.003512 0.003475 0.003441 0.003422
Cumulative Proportion 0.769294 0.77284 0.776356 0.779831 0.783272 0.786694
                          PC67     PC68    PC69     PC70     PC71     PC72
Eigenvalue            5.838647 5.739394 5.69698 5.682045 5.610206 5.575421
Proportion Explained  0.003412 0.003354 0.00333 0.003321 0.003279 0.003259
Cumulative Proportion 0.790106 0.793461 0.79679 0.800111 0.803390 0.806649
                          PC73     PC74     PC75     PC76     PC77     PC78
Eigenvalue            5.532546 5.512882 5.478654 5.449220 5.386962 5.340430
Proportion Explained  0.003234 0.003222 0.003202 0.003185 0.003148 0.003121
Cumulative Proportion 0.809883 0.813105 0.816307 0.819492 0.822640 0.825761
                          PC79    PC80     PC81     PC82     PC83     PC84
Eigenvalue            5.266293 5.23627 5.194200 5.169219 5.105011 5.079674
Proportion Explained  0.003078 0.00306 0.003036 0.003021 0.002984 0.002969
Cumulative Proportion 0.828839 0.83190 0.834935 0.837957 0.840940 0.843909
                         PC85     PC86     PC87     PC88     PC89   PC90
Eigenvalue            5.04673 5.000646 4.953136 4.850773 4.834124 4.7902
Proportion Explained  0.00295 0.002923 0.002895 0.002835 0.002825 0.0028
Cumulative Proportion 0.84686 0.849781 0.852676 0.855511 0.858337 0.8611
                          PC91     PC92     PC93     PC94     PC95     PC96
Eigenvalue            4.747070 4.646481 4.610741 4.601588 4.544317 4.489188
Proportion Explained  0.002774 0.002716 0.002695 0.002689 0.002656 0.002624
Cumulative Proportion 0.863911 0.866626 0.869321 0.872011 0.874667 0.877290
                          PC97     PC98    PC99    PC100    PC101    PC102
Eigenvalue            4.428100 4.402479 4.34579 4.317983 4.289615 4.251389
Proportion Explained  0.002588 0.002573 0.00254 0.002524 0.002507 0.002485
Cumulative Proportion 0.879878 0.882451 0.88499 0.887515 0.890022 0.892507
                         PC103    PC104    PC105   PC106    PC107    PC108
Eigenvalue            4.206402 4.101497 4.083153 4.03833 4.012932 3.972010
Proportion Explained  0.002458 0.002397 0.002386 0.00236 0.002345 0.002321
Cumulative Proportion 0.894965 0.897363 0.899749 0.90211 0.904455 0.906776
                         PC109    PC110    PC111    PC112    PC113    PC114
Eigenvalue            3.898906 3.864706 3.804274 3.779060 3.715445 3.709790
Proportion Explained  0.002279 0.002259 0.002223 0.002209 0.002172 0.002168
Cumulative Proportion 0.909055 0.911314 0.913537 0.915746 0.917917 0.920086
                         PC115   PC116    PC117    PC118    PC119    PC120
Eigenvalue            3.650424 3.57625 3.545824 3.504036 3.482633 3.434867
Proportion Explained  0.002134 0.00209 0.002072 0.002048 0.002035 0.002008
Cumulative Proportion 0.922219 0.92431 0.926382 0.928430 0.930465 0.932473
                         PC121    PC122    PC123    PC124    PC125    PC126
Eigenvalue            3.414401 3.359793 3.346152 3.338509 3.275925 3.235444
Proportion Explained  0.001996 0.001964 0.001956 0.001951 0.001915 0.001891
Cumulative Proportion 0.934468 0.936432 0.938387 0.940339 0.942253 0.944144
                         PC127    PC128    PC129    PC130   PC131    PC132
Eigenvalue            3.166798 3.158380 3.145361 3.100945 3.06349 3.054615
Proportion Explained  0.001851 0.001846 0.001838 0.001812 0.00179 0.001785
Cumulative Proportion 0.945995 0.947841 0.949679 0.951492 0.95328 0.955068
                         PC133    PC134    PC135    PC136    PC137    PC138
Eigenvalue            2.973743 2.957896 2.913109 2.906164 2.849495 2.818794
Proportion Explained  0.001738 0.001729 0.001703 0.001699 0.001665 0.001647
Cumulative Proportion 0.956806 0.958534 0.960237 0.961936 0.963601 0.965248
                         PC139    PC140   PC141    PC142    PC143    PC144
Eigenvalue            2.789842 2.701798 2.68542 2.670389 2.625179 2.557346
Proportion Explained  0.001631 0.001579 0.00157 0.001561 0.001534 0.001495
Cumulative Proportion 0.966879 0.968458 0.97003 0.971588 0.973123 0.974617
                         PC145    PC146    PC147    PC148    PC149    PC150
Eigenvalue            2.523665 2.465497 2.442256 2.422028 2.356660 2.103322
Proportion Explained  0.001475 0.001441 0.001427 0.001416 0.001377 0.001229
Cumulative Proportion 0.976092 0.977533 0.978961 0.980376 0.981754 0.982983
                         PC151    PC152  PC153     PC154     PC155     PC156
Eigenvalue            2.029944 1.995878 1.8819 1.6212677 1.5948046 1.4626046
Proportion Explained  0.001186 0.001167 0.0011 0.0009476 0.0009321 0.0008548
Cumulative Proportion 0.984169 0.985336 0.9864 0.9873832 0.9883153 0.9891701
                          PC157     PC158    PC159     PC160     PC161
Eigenvalue            1.2496250 1.1636752 1.091644 1.0389432 0.9805213
Proportion Explained  0.0007304 0.0006801 0.000638 0.0006072 0.0005731
Cumulative Proportion 0.9899005 0.9905806 0.991219 0.9918259 0.9923989
                          PC162     PC163     PC164     PC165     PC166
Eigenvalue            0.9388623 0.9182372 0.9096842 0.8980849 0.8556909
Proportion Explained  0.0005487 0.0005367 0.0005317 0.0005249 0.0005001
Cumulative Proportion 0.9929477 0.9934843 0.9940160 0.9945409 0.9950410
                          PC167     PC168     PC169     PC170     PC171
Eigenvalue            0.8333362 0.8063662 0.7818261 0.7770762 0.7558523
Proportion Explained  0.0004871 0.0004713 0.0004569 0.0004542 0.0004418
Cumulative Proportion 0.9955281 0.9959994 0.9964563 0.9969105 0.9973522
                          PC172     PC173     PC174     PC175     PC176
Eigenvalue            0.7269738 0.6904036 0.6720597 0.6390393 0.6235288
Proportion Explained  0.0004249 0.0004035 0.0003928 0.0003735 0.0003644
Cumulative Proportion 0.9977771 0.9981806 0.9985734 0.9989469 0.9993113
                          PC177     PC178
Eigenvalue            0.5967689 0.5815230
Proportion Explained  0.0003488 0.0003399
Cumulative Proportion 0.9996601 1.0000000

Accumulated constrained eigenvalues
Importance of components:
                          RDA1    RDA2    RDA3     RDA4     RDA5
Eigenvalue            289.8859 72.9640 56.4012 41.64097 15.81440
Proportion Explained    0.6081  0.1531  0.1183  0.08735  0.03317
Cumulative Proportion   0.6081  0.7612  0.8795  0.96683  1.00000

[?25h[?25h
--- R-squared and Adjusted R-squared ---
[?25h$r.squared
[1] 0.2786151

$adj.r.squared
[1] 0.2583515

[?25h
--- Global ANOVA (Significance of the Model) ---
[?25h[?25hPermutation test for rda under reduced model
Permutation: free
Number of permutations: 999

Model: rda(formula = genetic_data_final ~ wc2.1_30s_bio_1 + wc2.1_30s_bio_4 + wc2.1_30s_bio_12 + wc2.1_30s_bio_14 + wc2.1_30s_bio_16, data = as.data.frame(bioclim_scaled))
          Df Variance      F Pr(>F)    
Model      5   476.71 13.749  0.001 ***
Residual 178  1234.28                  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
[?25h
--- ANOVA by Terms (Marginal Effects of Variables) ---
[?25h^C^X^
Execution halted
[?25h]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ [ARscript RDAanalysis_2poplabeling_biopca_final_1000_prueba.R [C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[3Pnano[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[43@mv RDAanalysis_2poplabeling_biopca_final_1000_prueba.R RDAanalysis_2poplabeling_biopca_final_1000[C[C[C
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ 
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ 
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ 
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ 
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ 
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ cat RDAanalysis_2poplabeling_biopca_final.R 
#---------------------------------------------------#
# Step 1: Install and Load Required Packages
#---------------------------------------------------#
# 'vegan' for the RDA, 'ggplot2' for advanced plotting,
# and 'data.table' for efficient data reading.
if (!requireNamespace("vegan", quietly = TRUE)) {
  install.packages("vegan", dependencies = TRUE)
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", dependencies = TRUE)
}

if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table", dependencies = TRUE)
}

library(vegan)
library(ggplot2)
library(data.table)

#---------------------------------------------------#
# Step 2: Load and Clean Data
#---------------------------------------------------#

# Load the bioclimatic data.
bioclim_data <- read.csv("final_df_for_analysis.csv")

# Load the genetic data using fread() for efficiency.
raw_data <- fread("admixture_final_ld_pruned_filtered_for_RDA.raw")

# Load the population data.
tryCatch({
  # Changed from read.csv with sep="\t" to read.table with sep=" "
  pop_data <- read.table("population_list.txt", sep=" ", header = TRUE)
  if (is.null(pop_data) || nrow(pop_data) == 0) {
    stop("Population data file is empty or invalid.")
  }
}, error = function(e) {
  stop("Error loading population data: ", e$message)
})

# Format the genetic data: remove non-genotype columns and set row names.
genetic_data_final <- as.data.frame(raw_data[, 7:ncol(raw_data)])
rownames(genetic_data_final) <- raw_data$IID

# --- Subsample and clean the genetic data ---
# We will check for and remove any SNPs (columns) with NA/NaN/Inf values.
invalid_snps <- which(apply(genetic_data_final, 2, function(x) any(!is.finite(x))))
if(length(invalid_snps) > 0) {
    warning("Removed ", length(invalid_snps), " SNPs with invalid data.")
    genetic_data_final <- genetic_data_final[, -invalid_snps]
}

# Subsample SNPs to a manageable number (e.g., 5000) AFTER cleaning
set.seed(42) # For reproducibility
if (ncol(genetic_data_final) > 5000) {
    selected_snps <- sample(1:ncol(genetic_data_final), 5000)
    genetic_data_final <- genetic_data_final[, selected_snps]
}
cat("Final genetic data dimensions:", dim(genetic_data_final), "\n")


#---------------------------------------------------#
# Step 3: Align and Filter Data by Individuals
#---------------------------------------------------#
# This step ensures both datasets are perfectly aligned.

# Find individuals with complete bioclimatic data.
valid_bioclim_individuals <- complete.cases(bioclim_data)

# Find individuals with complete genetic data.
valid_genetic_individuals <- complete.cases(genetic_data_final)

# Find common individuals with valid data in all three sets.
common_valid_individuals <- intersect(bioclim_data$IND[valid_bioclim_individuals], rownames(genetic_data_final)[valid_genetic_individuals])
common_valid_individuals <- intersect(common_valid_individuals, pop_data$IID)

# --- NEW CHECK: Stop if no common individuals are found ---
if (length(common_valid_individuals) == 0) {
  stop("Error: No common individuals found across all three datasets (bioclimatic, genetic, and population data). Please check your input files.")
}

# Filter all data frames to keep only these individuals.
bioclim_data_filtered <- bioclim_data[bioclim_data$IND %in% common_valid_individuals, ]
genetic_data_final <- genetic_data_final[common_valid_individuals, ]
# The key change is adding drop = FALSE here to prevent simplification to a vector
pop_data_filtered <- pop_data[pop_data$IID %in% common_valid_individuals, , drop = FALSE]

# Order all data frames to ensure perfect alignment.
bioclim_data_filtered <- bioclim_data_filtered[order(bioclim_data_filtered$IND), ]
genetic_data_final <- genetic_data_final[order(rownames(genetic_data_final)), ]
# The key change is adding drop = FALSE here to prevent simplification to a vector
pop_data_filtered <- pop_data_filtered[order(pop_data_filtered$IID), , drop = FALSE]

# Final check of alignment
stopifnot(all(rownames(genetic_data_final) == bioclim_data_filtered$IND))
stopifnot(all(rownames(genetic_data_final) == pop_data_filtered$IID))
cat("Final number of individuals for RDA:", nrow(genetic_data_final), "\n")


#---------------------------------------------------#
# Step 4: Run the RDA Analysis
#---------------------------------------------------#
# Separate the predictor variables (bioclimatic) from the full data frame.
bioclim_predictors <- bioclim_data_filtered[, c(
  "wc2.1_30s_bio_1", "wc2.1_30s_bio_4", "wc2.1_30s_bio_12",
  "wc2.1_30s_bio_14", "wc2.1_30s_bio_16"
)]

# Scale and center the predictor variables.
bioclim_scaled <- scale(bioclim_predictors)

# Run the RDA.
rda_result <- rda(genetic_data_final ~ ., data = as.data.frame(bioclim_scaled))

#---------------------------------------------------#
# Step 5: Interpret and Visualize the RDA Results with ggplot2
#---------------------------------------------------#
summary(rda_result)
anova.cca(rda_result, permutations = 999)
anova.cca(rda_result, by = "margin", permutations = 999)

# Get the scores for plotting
site_scores <- scores(rda_result, display = "sites", scaling = 2)
env_scores <- scores(rda_result, display = "bp", scaling = 2)

# Convert scores to data frames for use with ggplot2
site_scores_df <- as.data.frame(site_scores)
site_scores_df$IID <- rownames(site_scores_df)
env_scores_df <- as.data.frame(env_scores)
env_scores_df$variables <- rownames(env_scores_df)

# Merge site scores with population data
plot_data <- merge(site_scores_df, pop_data_filtered, by = "IID")

# Create the RDA plot
rda_plot <- ggplot() +
  # Add points colored by population
  geom_point(data = plot_data, aes(x = RDA1, y = RDA2, color = POP), size = 3) +
  # Add arrows for environmental variables
  geom_segment(data = env_scores_df, aes(x = 0, y = 0, xend = RDA1, yend = RDA2), 
               arrow = arrow(length = unit(0.2, "cm")), color = "red") +
  # Add labels for environmental variables
  geom_text(data = env_scores_df, aes(x = RDA1, y = RDA2, label = variables), 
            hjust = 0, nudge_x = 0.05, color = "red") +
  
  # --- CHANGE START: Custom Colors and Labels ---
  # Maps POP1 to C. borbonica (cyan) and POP2 to C. boryana (orange)
  scale_color_manual(
    values = c("POP1" = "cyan", "POP2" = "orange"),
    labels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")
  ) +
  # --- CHANGE END ---

  # Set plot title and labels
  labs(title = "",
       subtitle = "",
       x = "RDA1", y = "RDA2", color = "Species") +
  # Add a theme for a clean look
  theme_minimal() +
  # Center the title and customize legend
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

# Save the RDA plot to a PDF file
pdf("RDA_plot_pop.pdf", width = 8, height = 8)
print(rda_plot)
dev.off()

print("RDA plot saved as RDA_plot_pop.pdf")

#---------------------------------------------------#
# Step 6: Perform and Visualize PCA on Bioclimatic Data
#---------------------------------------------------#
print("Performing PCA on bioclimatic variables...")

# Fix: Set the row names of the bioclimatic data to the individual IDs
# to ensure the merge with population data works correctly.
rownames(bioclim_predictors) <- bioclim_data_filtered$IND

# Perform PCA on the scaled bioclimatic data.
pca_bioclim <- prcomp(bioclim_predictors, center = TRUE, scale. = TRUE)

# Get the individual scores for PC1 and PC2.
pca_scores <- as.data.frame(pca_bioclim$x)
pca_scores$IID <- rownames(pca_scores)

# Merge PCA scores with population data.
pca_plot_data <- merge(pca_scores, pop_data_filtered, by = "IID")

# Create the PCA plot.
pca_plot <- ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = POP)) +
  geom_point(size = 3) +
  
  # --- CHANGE START: Custom Colors and Labels ---
  scale_color_manual(
    values = c("POP1" = "cyan", "POP2" = "orange"),
    labels = c("POP1" = "C. borbonica", "POP2" = "C. boryana")
  ) +
  # --- CHANGE END ---

  labs(title = "",
       subtitle = "",
       x = "PC1", y = "PC2", color = "Populations") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

# Save the PCA plot to a PDF file.
pdf("PCA_plot_bioclim.pdf", width = 8, height = 8)
print(pca_plot)
dev.off()

print("PCA plot saved as PCA_plot_bioclim.pdf")
]0;aescudero@login3:~/wgg/RDA(Renv) [aescudero@nodo4143 RDA]$ cat RDAanalysis_2poplabeling_biopca_final.R [K[K[K_1000.R 
#---------------------------------------------------#
# Step 1: Install and Load Required Packages
#---------------------------------------------------#
if (!requireNamespace("vegan", quietly = TRUE)) install.packages("vegan", dependencies = TRUE)
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2", dependencies = TRUE)
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table", dependencies = TRUE)

library(vegan)
library(ggplot2)
library(data.table)

# --- CONFIGURATION ---
# Path verified from your Python script
FST_WINDOW_FILE <- "/home/aescudero/wgg/fst_results/windowed_stats/windowed_fst_1000bp.windowed.weir.fst"
# Top % of FST windows to keep (0.00 = Keep ALL windows provided in the file)
FST_QUANTILE <- 0.00
# Target number of SNPs for RDA
TARGET_SNP_COUNT <- 5000
# ---------------------

#---------------------------------------------------#
# Step 2: Load and Clean Data
#---------------------------------------------------#

# Load the bioclimatic data.
bioclim_data <- read.csv("final_df_for_analysis.csv")

# Load the genetic data using fread() for efficiency.
raw_data <- fread("admixture_final_ld_pruned_filtered_for_RDA.raw")

# Load the population data.
tryCatch({
  pop_data <- read.table("population_list.txt", sep=" ", header = TRUE)
  if (is.null(pop_data) || nrow(pop_data) == 0) stop("Population data file is empty.")
}, error = function(e) stop("Error loading population data: ", e$message))

# Format the genetic data: remove non-genotype columns and set row names.
# Columns 1-6 are metadata (FID, IID, PAT, MAT, SEX, PHENOTYPE)
genetic_data_final <- as.data.frame(raw_data[, 7:ncol(raw_data)])
rownames(genetic_data_final) <- raw_data$IID

# --- Clean Genetic Data (Strict Filtering - Remove SNPs with ANY NAs) ---
cat("Total SNPs before cleaning:", ncol(genetic_data_final), "\n")

# Identify SNPs with ANY missing data (NA or Inf)
invalid_snps <- which(apply(genetic_data_final, 2, function(x) any(!is.finite(x))))

if(length(invalid_snps) > 0) {
    warning("Removed ", length(invalid_snps), " SNPs with invalid data (NAs or Inf).")
    genetic_data_final <- genetic_data_final[, -invalid_snps]
}

cat("Total SNPs available after strict filtering:", ncol(genetic_data_final), "\n")

#---------------------------------------------------#
# Step 2b: Filter SNPs based on 1000bp FST Windows
#---------------------------------------------------#
cat("\n--- STARTING FST WINDOW FILTERING ---\n")

# 1. Load FST Window Data
if (!file.exists(FST_WINDOW_FILE)) stop("FST window file not found.")
fst_windows <- fread(FST_WINDOW_FILE)

# 2. Identify Outlier Windows
# Filter out NAs first
fst_windows <- fst_windows[!is.na(WEIGH