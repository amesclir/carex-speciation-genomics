import pandas as pd
import os

def load_and_parse_annotation(annotation_file):
    """
    Loads the GFF3 annotation file and filters for gene features.
    
    Args:
        annotation_file (str): Path to the BRAKER GFF3 file.
    
    Returns:
        pd.DataFrame: DataFrame containing only gene features with extracted IDs.
    """
    print(f"Loading gene annotation data from {annotation_file}...")
    
    # Load the GFF3 file, skipping header lines marked with '#'
    df_genes = pd.read_csv(
        annotation_file,
        sep='\t',
        header=None,
        comment='#',
        names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    )
    
    # Filter for 'gene' features only
    df_genes = df_genes[df_genes['feature'] == 'gene'].copy()
    
    # --- CORRECTION APPLIED HERE ---
    # The new regex extracts the ID directly after 'ID=' until the next semicolon ';' or end of line.
    # e.g., for 'ID=g1;', it extracts 'g1'.
    df_genes['gene_id'] = df_genes['attribute'].str.extract(r'ID=([^;]+)')
    
    df_genes.dropna(subset=['gene_id'], inplace=True)
    
    # Convert chromosome names (e.g., 'scaffold_14') to their integer form (e.g., 14) 
    # for easy comparison with the SV list.
    df_genes['seqname_int'] = (df_genes['seqname'].astype(str)
                               .str.replace('scaffold_', '', regex=False)
                               .astype(int, errors='ignore'))

    # Drop rows where conversion failed (like non-scaffold entries, if any)
    df_genes.dropna(subset=['seqname_int'], inplace=True)
    
    print(f"Annotation loaded. Found {len(df_genes)} unique gene features with IDs.")
    return df_genes

def define_top_svs():
    """
    Hardcoded definition of the top 10 SVs based on the provided list.
    """
    # CHR numbers are integers corresponding to the scaffold_X naming convention
    sv_list = [
        {'id': 'SV1', 'chr': 14, 'start': 3040103, 'end': 5331878, 'type': 'INV'},
        {'id': 'SV2', 'chr': 18, 'start': 1681715, 'end': 2543617, 'type': 'INV'},
        {'id': 'SV3', 'chr': 20, 'start': 4207035, 'end': 4474484, 'type': 'INV'},
        {'id': 'SV4', 'chr': 3, 'start': 6538064, 'end': 6773584, 'type': 'INV'},
        {'id': 'SV5', 'chr': 2, 'start': 15863192, 'end': 16067646, 'type': 'DUP'},
        {'id': 'SV6', 'chr': 3, 'start': 235306, 'end': 373115, 'type': 'TRANS'},
        {'id': 'SV7', 'chr': 2, 'start': 15962711, 'end': 16092230, 'type': 'DUP'},
        {'id': 'SV8', 'chr': 32, 'start': 890484, 'end': 1017805, 'type': 'INV'},
        {'id': 'SV9', 'chr': 22, 'start': 8850975, 'end': 8974282, 'type': 'INV'},
        {'id': 'SV10', 'chr': 2, 'start': 16085119, 'end': 16205029, 'type': 'DUP'}
    ]
    return pd.DataFrame(sv_list)

def extract_genes_in_svs(df_genes, df_svs):
    """
    Iterates through SVs, finds overlapping genes, and saves the list.
    """
    OUTPUT_DIR = "sv_gene_lists"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\nStarting gene extraction for 10 SVs...")
    
    for index, sv in df_svs.iterrows():
        sv_id = sv['id']
        chr_int = sv['chr']
        sv_start = sv['start']
        sv_end = sv['end']
        sv_type = sv['type']

        # Ensure correct coordinate order for gene overlap check
        start_coord = min(sv_start, sv_end)
        end_coord = max(sv_start, sv_end)
        
        # Identify genes that overlap with the SV region
        genes_in_sv = df_genes[
            (df_genes['seqname_int'] == chr_int) &
            (df_genes['start'] <= end_coord) &
            (df_genes['end'] >= start_coord)
        ].copy()

        gene_list = genes_in_sv['gene_id'].unique()
        count = len(gene_list)
        
        output_filename = f"{sv_id}_{sv_type}_Chr{chr_int}.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        if count > 0:
            # We sort the gene list alphabetically before writing for cleaner output
            gene_list_sorted = sorted(gene_list)
            with open(output_path, 'w') as f:
                for gene in gene_list_sorted:
                    f.write(f"{gene}\n")
            print(f"[{sv_id} (Chr{chr_int})] Found {count} genes. Saved to {output_path}")
        else:
            print(f"[{sv_id} (Chr{chr_int})] Found 0 genes. No file generated.")


def main():
    # --- USER-DEFINED INPUT PATH ---
    ANNOTATION_FILE = "/home/aescudero/genome_assembly/C2_braker/braker_cbory/braker.gff3"
    
    try:
        df_svs = define_top_svs()
        df_genes = load_and_parse_annotation(ANNOTATION_FILE)
        
        extract_genes_in_svs(df_genes, df_svs)

        print("\n--- Gene Extraction Complete ---")
        print(f"All gene lists are saved in the 'sv_gene_lists/' directory.")
        print("You are now ready to perform Gene Ontology (GO) enrichment analysis on these lists.")

    except FileNotFoundError as e:
        print(f"\nError: {e}. Please ensure the annotation file path is correct: {ANNOTATION_FILE}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
