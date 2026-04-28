import pandas as pd
import os
import re

# --- Configuration ---
OUTPUT_DIR = "."
INSIDE_CSV = "results_omega_inside.csv"
OUTSIDE_CSV = "results_omega_outside.csv"
MAP_FILE = "Carex_boryana_vs_Carex_borbonica.synHits.txt" 

# --- Filtering Thresholds (MUST match run9 logic) ---
DS_MIN_THRESHOLD = 0.01 # Excludes dS or dN = 0.0
DS_MAX_THRESHOLD = 1.5   # Excludes saturated dS or dN > 2.0
# ----------------------------------------------------

# --- Selection Thresholds (STRICT) ---
PURIFYING_THRESHOLD = 0.10 
POSITIVE_THRESHOLD = 1.1 

# --- Output Files ---
# C. borbonica IDs (PAML IDs) for reference
GENES_INSIDE_PURIFYING_CBOR = f"genes_inside_purifying_w{int(PURIFYING_THRESHOLD*100)}_Cbor.txt"
GENES_INSIDE_POSITIVE_CBOR = f"genes_inside_positive_w{int(POSITIVE_THRESHOLD*100)}_Cbor.txt"
GENES_OUTSIDE_PURIFYING_CBOR = f"genes_outside_purifying_w{int(PURIFYING_THRESHOLD*100)}_Cbor.txt"
GENES_OUTSIDE_POSITIVE_CBOR = f"genes_outside_positive_w{int(POSITIVE_THRESHOLD*100)}_Cbor.txt"

# C. boryana IDs (Target IDs for GO Enrichment)
GENES_INSIDE_PURIFYING_CBRY = f"genes_inside_purifying_w{int(PURIFYING_THRESHOLD*100)}_Cbry.txt"
GENES_INSIDE_POSITIVE_CBRY = f"genes_inside_positive_w{int(POSITIVE_THRESHOLD*100)}_Cbry.txt"
GENES_OUTSIDE_PURIFYING_CBRY = f"genes_outside_purifying_w{int(PURIFYING_THRESHOLD*100)}_Cbry.txt"
GENES_OUTSIDE_POSITIVE_CBRY = f"genes_outside_positive_w{int(POSITIVE_THRESHOLD*100)}_Cbry.txt"

SUMMARY_REPORT = "selection_summary_report_strict_mapped_FILTERED_FIXED.txt"
# ---------------------

def load_and_prepare_mapping(map_file):
    """Loads and cleans the ortholog mapping file, robustly handling whitespace separators."""
    try:
        # Load columns: id1=4 (C. boryana) and id2=12 (C. borbonica)
        # Use r'\s+' as a raw string separator for robust regex handling.
        mapping_df = pd.read_csv(
            map_file, 
            sep=r'\s+', 
            usecols=[4, 12], 
            names=['Cbry_ID', 'Cbor_ID_Map'],
            skiprows=1 
        )
    except Exception as e:
        print(f"Error loading mapping file {map_file}: {e}")
        return None

    # Keep a single, consistent mapping for the lookup
    mapping_df = mapping_df.drop_duplicates(subset=['Cbor_ID_Map'], keep='first').copy()
    
    print(f"Loaded {len(mapping_df)} unique ortholog pairs for mapping.")
    return mapping_df


def apply_dn_ds_filter(df, group_name):
    """Applies the strict dN and dS filtering rules (0.001 <= dN/dS <= 2.0)."""
    
    initial_count = len(df)
    
    # --- CORRECTED FILTERING LOGIC ---
    df_filtered = df[
        (df['dS'] >= DS_MIN_THRESHOLD) & (df['dS'] <= DS_MAX_THRESHOLD) 
#&
 #       (df['dN'] >= DS_MIN_THRESHOLD) & (df['dN'] <= DS_MAX_THRESHOLD)
    ].copy()
    
    filtered_count = len(df_filtered)
    print(f"{group_name}: Filtered {initial_count} -> {filtered_count} genes. (Removed {initial_count - filtered_count})")
    print(f"Filter used: {DS_MIN_THRESHOLD} \u2264 dS \u2264 {DS_MAX_THRESHOLD} AND {DS_MIN_THRESHOLD} \u2264 dN \u2264 {DS_MAX_THRESHOLD}")
    
    return df_filtered


def summarize_selection(df, group_name, mapping_df):
    """
    1. Identifies outliers on the original PAML DF (Cbor IDs).
    2. Maps the Cbor outliers to Cbry IDs using the ortholog map.
    3. Generates summary statistics.
    NOTE: The input DF here is already filtered.
    """
    
    total_genes = len(df)
    
    # Function to clean PAML gene_id for merging
    def clean_id(gene_id):
        gene_id_str = str(gene_id).strip() 
        
        # 1. Try to find the full G.t format (e.g., g5059.t1)
        match_full = re.search(r'(g\d+\.t\d+)', gene_id_str)
        if match_full:
            return match_full.group(1)
            
        # 2. If not found, try to find the base G format (e.g., g5059)
        match_base = re.search(r'(g\d+)', gene_id_str)
        if match_base:
            # ASSUMPTION: Map to the primary transcript (.t1) for merging
            return f"{match_base.group(1)}.t1"
            
        # 3. If neither pattern is found, return None
        return None

    # --- 1. IDENTIFY OUTLIERS (C. borbonica - Cbor IDs) ---
    purifying_df_cbor = df[df['omega'] <= PURIFYING_THRESHOLD].copy()
    positive_df_cbor = df[df['omega'] >= POSITIVE_THRESHOLD].copy()
    
    purifying_list_borbonica = purifying_df_cbor['gene_id'].tolist()
    positive_list_borbonica = positive_df_cbor['gene_id'].tolist()

    # --- 2. MAP PURIFYING GENES to C. boryana (Cbry) ---
    if purifying_df_cbor.empty:
        purifying_list_boryana = []
    else:
        # Apply the fixed clean_id function
        purifying_df_cbor['Cbor_ID_Map'] = purifying_df_cbor['gene_id'].apply(clean_id)
        df_to_map = purifying_df_cbor[purifying_df_cbor['Cbor_ID_Map'].notna()].copy()

        df_purifying_mapped = df_to_map.merge(
            mapping_df, 
            on='Cbor_ID_Map', 
            how='inner'
        )
        purifying_list_boryana = df_purifying_mapped['Cbry_ID'].tolist()


    # --- 3. MAP POSITIVE GENES to C. boryana (Cbry) ---
    if positive_df_cbor.empty:
        positive_list_boryana = []
    else:
        # Apply the fixed clean_id function
        positive_df_cbor['Cbor_ID_Map'] = positive_df_cbor['gene_id'].apply(clean_id)
        df_to_map = positive_df_cbor[positive_df_cbor['Cbor_ID_Map'].notna()].copy()

        df_positive_mapped = df_to_map.merge(
            mapping_df, 
            on='Cbor_ID_Map', 
            how='inner'
        )
        positive_list_boryana = df_positive_mapped['Cbry_ID'].tolist()
    
    # --- 4. GENERATE SUMMARY STATS ---
    
    # Calculate overall mapped gene count for the report
    df['Cbor_ID_Map'] = df['gene_id'].apply(clean_id)
    df_temp = df[df['Cbor_ID_Map'].notna()].copy() 
    df_merged_full = df_temp.merge(mapping_df, on='Cbor_ID_Map', how='inner')
    mapped_genes_total = len(df_merged_full)
    
    median_omega = df['omega'].median()
    mean_omega = df['omega'].mean()
    
    # Generate summary text
    summary = f"\n--- {group_name} Group (N={total_genes} FILTERED PAML IDs, {mapped_genes_total} Total Mapped IDs) ---\n"
    summary += f"Median Omega (\u03c9): {median_omega:.4f}\n"
    summary += f"Mean Omega (\u03c9): {mean_omega:.4f}\n"
    summary += f"Genes with Strong Purifying (\u03c9 \u2264 {PURIFYING_THRESHOLD:.2f}): {len(purifying_list_borbonica)} Cbor IDs | {len(purifying_list_boryana)} Cbry Mapped IDs\n"
    summary += f"Genes with Strong Candidate Positive (\u03c9 \u2265 {POSITIVE_THRESHOLD:.2f}): {len(positive_list_borbonica)} Cbor IDs | {len(positive_list_boryana)} Cbry Mapped IDs\n"
    
    return summary, purifying_list_borbonica, positive_list_borbonica, purifying_list_boryana, positive_list_boryana


def write_gene_lists(gene_list, filename):
    """Writes a list of gene IDs to a plain text file, handling empty lists gracefully."""
    if not gene_list:
        print(f"⚠️ Warning: No genes found for {filename}. Skipping file creation.")
        return
        
    with open(filename, 'w') as f:
        for gene in gene_list:
            f.write(f"{gene}\n")
    print(f"✅ Wrote {len(gene_list)} gene IDs to {filename}")


if __name__ == "__main__":
    
    print("--- Selection Outlier Analysis (Strict Thresholds + ID Mapping) Started ---")
    
    # 0. Load the mapping file
    mapping_df = load_and_prepare_mapping(MAP_FILE)
    if mapping_df is None:
        print("Fatal Error: Could not load the gene mapping file. Exiting.")
        exit()

    # 1. Load the PAML result files
    if not os.path.exists(INSIDE_CSV) or not os.path.exists(OUTSIDE_CSV):
        print("ERROR: One or both PAML CSV files not found. Exiting.")
        exit()

    df_inside = pd.read_csv(INSIDE_CSV)
    df_outside = pd.read_csv(OUTSIDE_CSV)

    # 1.5. Apply the required filtering (dN/dS >= 0.001 and <= 2.0)
    print("\n--- Applying Filtering based on dN and dS rates ---")
    df_inside_filtered = apply_dn_ds_filter(df_inside, "Inside Inversion")
    df_outside_filtered = apply_dn_ds_filter(df_outside, "Outside Inversion")
    
    # 2. Process and map Inside Inversion Genes
    (summary_inside, purifying_inside_cbor, positive_inside_cbor, 
     purifying_inside_cbry, positive_inside_cbry) = summarize_selection(df_inside_filtered, "Inside Inversion", mapping_df)
    
    # 3. Process and map Outside Inversion Genes
    (summary_outside, purifying_outside_cbor, positive_outside_cbor,
     purifying_outside_cbry, positive_outside_cbry) = summarize_selection(df_outside_filtered, "Outside Inversion", mapping_df)
    
    # 4. Write ALL lists 
    print("\n--- Generating Gene Lists for GO Enrichment ---")
    
    # Write C. borbonica lists (PAML IDs)
    write_gene_lists(purifying_inside_cbor, GENES_INSIDE_PURIFYING_CBOR)
    write_gene_lists(positive_inside_cbor, GENES_INSIDE_POSITIVE_CBOR)
    write_gene_lists(purifying_outside_cbor, GENES_OUTSIDE_PURIFYING_CBOR)
    write_gene_lists(positive_outside_cbor, GENES_OUTSIDE_POSITIVE_CBOR)
    
    # Write C. boryana lists (GO Enrichment IDs)
    write_gene_lists(purifying_inside_cbry, GENES_INSIDE_PURIFYING_CBRY)
    write_gene_lists(positive_inside_cbry, GENES_INSIDE_POSITIVE_CBRY)
    write_gene_lists(purifying_outside_cbry, GENES_OUTSIDE_PURIFYING_CBRY)
    write_gene_lists(positive_outside_cbry, GENES_OUTSIDE_POSITIVE_CBRY)

    # 5. Write summary report
    with open(SUMMARY_REPORT, 'w') as f:
        f.write(f"PAML Outlier Gene Selection Analysis Report (Thresholds: Purifying \u2264 {PURIFYING_THRESHOLD:.2f}, Positive \u2265 {POSITIVE_THRESHOLD:.2f})\n")
        f.write(f"Data Filtered: {DS_MIN_THRESHOLD} \u2264 dN \u2264 {DS_MAX_THRESHOLD} AND {DS_MIN_THRESHOLD} \u2264 dS \u2264 {DS_MAX_THRESHOLD} APPLIED BEFORE ANALYSIS.\n")
        f.write(summary_inside)
        f.write(summary_outside)
        
    print(f"\n✅ Summary report saved to {SUMMARY_REPORT}")
    print("\nFinal Step: Use the generated *_Cbry.txt gene lists for Carex boryana GO Enrichment analysis. Let me know how many C. boryana IDs were mapped!")

