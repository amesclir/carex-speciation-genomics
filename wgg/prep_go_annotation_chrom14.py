import pandas as pd
import sys

# --- USER-DEFINED INPUTS ---
# Path to your SyRI output file (used to confirm inversion details if needed)
# Please confirm this path is correct if your SyRI output file is in a different location.
syri_file = "/home/aescudero/syri/syri.out"

# Path to your BRAKER gene annotation file
annotation_file = "/home/aescudero/genome_assembly/C2_braker/braker_cbory/braker.gff3"

# Coordinates of the specific inversion on chromosome 14
inversion_chrom = "scaffold_14"
inversion_start = 3040103
inversion_end = 5331878

# --- SCRIPT LOGIC ---
try:
    print(f"Loading gene annotation data from {annotation_file}...")
    
    # Load the GFF3 file
    # We specify names for the columns and use comment='#' to skip header lines
    df_genes = pd.read_csv(
        annotation_file,
        sep='\t',
        header=None,
        comment='#',
        names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    )
    
    # Filter for 'gene' features only, as GFF3 files can contain many other feature types
    df_genes = df_genes[df_genes['feature'] == 'gene'].copy()
    
    # Extract the gene ID from the 'attribute' column. This regex pattern
    # specifically looks for 'ID=gene:' followed by the gene name up to a semicolon.
    df_genes['gene_id'] = df_genes['attribute'].str.extract(r'ID=gene:(.*?)(?:;|$)')
    df_genes.dropna(subset=['gene_id'], inplace=True)
    df_genes['seqname'] = df_genes['seqname'].astype(str)
    
    # Identify genes that overlap with the specified inversion region
    genes_in_inversion = df_genes[
        (df_genes['seqname'] == inversion_chrom) &
        (df_genes['start'] <= inversion_end) &
        (df_genes['end'] >= inversion_start)
    ].copy()

    # Extract the unique gene IDs
    gene_list = genes_in_inversion['gene_id'].unique()

    if len(gene_list) == 0:
        print("\nNo genes found within the specified inversion region.")
    else:
        output_file = "genes_in_inversion_chr14.txt"
        with open(output_file, 'w') as f:
            for gene in gene_list:
                f.write(f"{gene}\n")
        
        print(f"\nSuccess! Found {len(gene_list)} genes.")
        print(f"A list of gene IDs has been saved to {output_file}.")
        print("You can now use this list for GO annotation tools like DAVID, PANTHER, or g:Profiler.")
        print("For instance, you could run a tool like g:Profiler with this command:")
        print("  gprofiler --output=go_enrichment_results.txt --organism=<your_organism_id> --input=genes_in_inversion_chr14.txt")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure both the SyRI and gene annotation files exist and the paths are correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
