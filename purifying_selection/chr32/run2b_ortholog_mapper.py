import os

def strip_transcript_suffix(gene_id):
    """Removes the .tX (e.g., .t1, .t1972) transcript suffix for cleaner matching."""
    # Find the last occurrence of '.', if it contains 't' it's likely a transcript suffix
    if '.' in gene_id and 't' in gene_id.rsplit('.', 1)[-1].lower():
        return gene_id.rsplit('.', 1)[0]
    return gene_id

def map_orthologs(gene_list_path, synteny_hits_path, output_path):
    """
    Reads a list of target C. borbonica gene IDs and maps them to their 
    orthologs in C. boryana using the synteny hits file, stripping transcript 
    suffixes for successful matching.
    """
    
    # --- Configuration based on header analysis ---
    # Col 4: id1 (C. boryana Gene ID), Col 12: id2 (C. borbonica Gene ID)
    BORYANA_COL = 4
    BORBONICA_COL = 12
    MIN_COLUMNS = max(BORBONICA_COL, BORYANA_COL) + 1
    
    print(f"--- Starting Ortholog Mapping: Step 2 (V4 - Final Fix) ---")
    
    # 1. Load the target C. borbonica gene IDs
    try:
        # Load the target gene IDs (e.g., g4919)
        with open(gene_list_path, 'r') as f:
            target_borbonica_genes = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(target_borbonica_genes)} target C. borbonica genes.")
    except FileNotFoundError:
        print(f"ERROR: Target gene list not found at {gene_list_path}.")
        return

    # 2. Process the synteny hits file and map
    mapped_boryana_genes = set()
    mapped_count = 0
    processed_lines = 0
    
    try:
        with open(synteny_hits_path, 'r', encoding='utf-8') as f:
            print(f"Processing synteny hits file: {synteny_hits_path}...")
            
            # Skip the header line
            f.readline()
            
            for line_number, line in enumerate(f, 2):
                
                # Split by tab
                parts = line.strip().split('\t')
                
                if len(parts) >= MIN_COLUMNS:
                    
                    # Get the full IDs from the synteny file
                    gene_borbonica_full = parts[BORBONICA_COL]
                    gene_boryana_full = parts[BORYANA_COL]
                    
                    # CRITICAL: Strip the suffix for matching against the target list
                    gene_borbonica_stripped = strip_transcript_suffix(gene_borbonica_full)
                    
                    if gene_borbonica_stripped in target_borbonica_genes:
                        # If a match is found, strip the suffix from the ortholog ID 
                        # and add the clean ID to the result set.
                        gene_boryana_stripped = strip_transcript_suffix(gene_boryana_full)
                        mapped_boryana_genes.add(gene_boryana_stripped)
                        mapped_count += 1
                    
                    processed_lines += 1
                
    except FileNotFoundError:
        print(f"FATAL ERROR: Synteny hits file not found at {synteny_hits_path}.")
        return
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred while reading the synteny file on line {line_number}: {e}")
        return

    # 3. Save the final list of orthologs
    try:
        with open(output_path, 'w') as out_f:
            for gene in sorted(list(mapped_boryana_genes)):
                out_f.write(gene + '\n')
                
        # Report results
        print("-" * 40)
        print(f"Total C. borbonica genes targeted: {len(target_borbonica_genes)}")
        print(f"Total gene-pairs processed in synHits: {processed_lines}")
        print(f"Total gene-pairs matched to target: {mapped_count}")
        print(f"Successfully mapped {len(mapped_boryana_genes)} unique C. boryana orthologs.")
        print(f"Ortholog list saved to: {output_path}")
        print("-" * 40)
        print("NEXT STEP: Now you have the gene lists for both species. Step 3 is preparing the alignment files.")

    except IOError as e:
        print(f"ERROR: Could not write to output file {output_path}: {e}")
        
# --- Main execution block ---
if __name__ == "__main__":
    GENE_LIST = "target_borbonica_genes_OUTSIDE.txt"
    SYNTENY_HITS = "Carex_boryana_vs_Carex_borbonica.synHits.txt" 
    OUTPUT_FILE = "target_boryana_orthologs_OUTSIDE.txt"
    
    map_orthologs(
        gene_list_path=GENE_LIST,
        synteny_hits_path=SYNTENY_HITS,
        output_path=OUTPUT_FILE
    )

