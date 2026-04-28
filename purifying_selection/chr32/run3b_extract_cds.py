import os
import re

# --- Helper Functions ---

def read_fasta(fasta_path):
    """Reads a FASTA file and returns a dictionary of sequences (keys=ID, values=seq)."""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                # The ID is the first word after the '>'
                current_id = line.strip().split()[0].replace('>', '')
                current_seq = []
            else:
                current_seq.append(line.strip().upper())
        # Add the last sequence
        if current_id is not None:
            sequences[current_id] = "".join(current_seq)
            
    print(f"Loaded {len(sequences)} scaffolds/chromosomes from {os.path.basename(fasta_path)}.")
    return sequences

def reverse_complement(seq):
    """Computes the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return "".join(complement.get(base, base) for base in reversed(seq))

def extract_base_gene_id(attributes):
    """
    Extracts the base gene ID (e.g., 'g1234') from the GFF3 attributes string.
    This handles common formats like 'Parent=g1234.t1' or 'ID=cds-g1234.t1'.
    """
    if 'Parent=' in attributes:
        # Find the parent ID (which is usually the transcript ID)
        match = re.search(r'Parent=([^;]+)', attributes)
        if match:
            # Strip off the transcript suffix (e.g., .t1)
            transcript_id = match.group(1)
            gene_match = re.match(r'(g\d+)', transcript_id)
            if gene_match:
                return gene_match.group(1)
            return transcript_id
    
    # Fallback to general search if 'Parent=' is not found
    gene_match = re.search(r'(g\d+)', attributes)
    if gene_match:
        return gene_match.group(1)
        
    return None # Return None if no identifiable gene ID is found

# --- Main Extraction Function ---

def extract_cds_from_gff3(genome_path, gff_path, gene_list_path, output_fasta_path):
    """
    Extracts CDS sequences for target genes using GFF3 coordinates and a full genome FASTA.
    """
    print(f"--- Processing {os.path.basename(gff_path)} ---")
    
    # 1. Load Genome Sequences
    genome_sequences = read_fasta(genome_path)
    if not genome_sequences:
        print("ERROR: Genome sequences failed to load.")
        return

    # 2. Load Target Gene IDs
    try:
        with open(gene_list_path, 'r') as f:
            target_genes = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(target_genes)} target genes from {os.path.basename(gene_list_path)}.")
    except FileNotFoundError:
        print(f"ERROR: Gene list not found at {gene_list_path}. Skipping.")
        return

    # 3. Parse GFF3 and Collect CDS Fragments
    
    # Structure: {base_gene_id: {transcript_id: [(scaffold, start, end, strand)]}}
    cds_fragments_by_gene = {}
    
    try:
        with open(gff_path, 'r') as gff_in:
            for line in gff_in:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 8 or fields[2] != 'CDS':
                    continue

                scaffold = fields[0]
                feature_type = fields[2]
                # GFF3 coordinates are 1-based, inclusive.
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]
                attributes = fields[8]
                
                base_gene_id = extract_base_gene_id(attributes)

                if base_gene_id and base_gene_id in target_genes:
                    # Find the transcript ID, which is the direct parent of the CDS
                    transcript_match = re.search(r'(Parent=)([^;]+)', attributes)
                    if not transcript_match:
                        # If no standard parent is found, skip this fragment
                        continue
                    
                    transcript_id = transcript_match.group(2)
                    
                    # Store the fragment (using Python 0-based slicing convention)
                    # Coordinates stored are (scaffold, start_0_based_inclusive, end_0_based_exclusive, strand)
                    fragment = (scaffold, start - 1, end, strand)

                    # Initialize structure if necessary
                    if base_gene_id not in cds_fragments_by_gene:
                        cds_fragments_by_gene[base_gene_id] = {}
                        
                    if transcript_id not in cds_fragments_by_gene[base_gene_id]:
                        cds_fragments_by_gene[base_gene_id][transcript_id] = []
                        
                    cds_fragments_by_gene[base_gene_id][transcript_id].append(fragment)

    except FileNotFoundError:
        print(f"FATAL ERROR: GFF3 file not found at {gff_path}.")
        return

    # 4. Assemble and Write Sequences
    extracted_count = 0
    with open(output_fasta_path, 'w') as fasta_out:
        for gene_id, transcripts in cds_fragments_by_gene.items():
            # For simplicity, we typically take the first transcript (longest, or 't1')
            # You might want to sort this later if you have complex transcripts
            
            # Find the best transcript ID (e.g., the one that appears first alphabetically/numerically)
            best_transcript_id = sorted(transcripts.keys())[0]
            fragments = transcripts[best_transcript_id]

            # Sort fragments by start coordinate for assembly
            # GFF3 files *should* be sorted, but sorting here ensures correct assembly
            fragments.sort(key=lambda x: x[1]) # Sort by 0-based start coordinate

            full_cds_sequence = []
            
            # The strand is the same for all fragments of a single transcript
            strand = fragments[0][3]
            scaffold = fragments[0][0]
            
            # Check if the scaffold actually exists in the FASTA (sanity check)
            if scaffold not in genome_sequences:
                print(f"WARNING: Scaffold {scaffold} for gene {gene_id} not found in .fna file. Skipping.")
                continue

            # Assemble the CDS sequence
            genome_seq = genome_sequences[scaffold]
            for _, start_0, end_1, _ in fragments:
                # Use Python slicing: [start_0_based : end_0_based_exclusive]
                full_cds_sequence.append(genome_seq[start_0:end_1])
            
            final_sequence = "".join(full_cds_sequence)
            
            # Apply reverse complement if needed
            if strand == '-':
                final_sequence = reverse_complement(final_sequence)

            # Write the assembled sequence
            fasta_out.write(f">{gene_id}|{best_transcript_id}|{scaffold}|{strand}\n")
            # Write in blocks of 60 characters
            for i in range(0, len(final_sequence), 60):
                fasta_out.write(final_sequence[i:i+60] + '\n')
            
            extracted_count += 1

    print(f"Successfully extracted {extracted_count} complete CDS sequences.")
    print(f"Subset FASTA saved to: {output_fasta_path}\n")


# --- Main execution block ---
if __name__ == "__main__":
    
    # File paths (now using your exact filenames)
    BORBONICA_LIST = "target_borbonica_genes_OUTSIDE.txt"
    BORYANA_LIST = "target_boryana_orthologs_OUTSIDE.txt"
    
    BORBONICA_FASTA_IN = "Carex_borbonica.fna" # Genome Assembly
    BORYANA_FASTA_IN = "Carex_boryana.fna"     # Genome Assembly
    
    BORBONICA_GFF3 = "Carex_borbonica.gff3"    # Annotation
    BORYANA_GFF3 = "Carex_boryana.gff3"        # Annotation
    
    # Output files for the next step (Alignment)
    BORBONICA_FASTA_OUT = "borbonica_ortholog_cds_OUTSIDE.fasta"
    BORYANA_FASTA_OUT = "boryana_ortholog_cds_OUTSIDE.fasta"

    print("--- Starting Ortholog CDS Extraction (GFF3 Method): Step 3 ---")
    
    # Process C. borbonica target genes
    extract_cds_from_gff3(
        genome_path=BORBONICA_FASTA_IN,
        gff_path=BORBONICA_GFF3,
        gene_list_path=BORBONICA_LIST,
        output_fasta_path=BORBONICA_FASTA_OUT
    )
    
    # Process C. boryana orthologs
    extract_cds_from_gff3(
        genome_path=BORYANA_FASTA_IN,
        gff_path=BORYANA_GFF3,
        gene_list_path=BORYANA_LIST,
        output_fasta_path=BORYANA_FASTA_OUT
    )
    
    print("-" * 40)
    print("All sequences extracted (via GFF3). NEXT STEP: Aligning the paired sequences.")

