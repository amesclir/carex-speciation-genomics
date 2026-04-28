import os
from Bio import SeqIO

# --- Configuration ---
INPUT_DIR = "codon_aligned_dnds_OUTSIDE"
OUTPUT_DIR = "paml_inputs_OUTSIDE"
# Update this if the problematic file changes, though it often remains the same
EMPTY_FILE_TO_REMOVE = os.path.join(INPUT_DIR, "g5098.final.fasta") 

# Standard PAML tree for a two-sequence comparison (Borbonica and Boryana)
# CRITICAL: These names MUST be 10 characters or less and MUST match the forced names below.
TREE_FILE_CONTENT = "(Carex_bobo:0.1, Carex_bory:0.1);"
TREE_FILE_NAME = "twoseq.tree"

# ---------------------

def write_paml_phylip(records, output_path):
    """
    Manually writes a sequence alignment to a PAML-compatible PHYLIP sequential file.
    This strictly enforces 10-character names, a minimum of two spaces separation, 
    and block formatting for long sequences, which PAML requires.
    """
    if not records:
        print(f"Warning: No records to write for {output_path}")
        return

    num_seqs = len(records)
    aln_length = len(records[0].seq)
    
    # PAML requires sequence data to be broken into blocks, usually 10 bases wide
    block_size = 10 

    # 1. Write header line: Number of sequences and length
    with open(output_path, 'w') as f:
        # PAML often requires a space before the number of sequences
        f.write(f" {num_seqs} {aln_length}\n") 

        # 2. Write the first block of sequences (Names and initial sequence block)
        for i, record in enumerate(records):
            # Format the name to be exactly 10 characters wide (left-justified), e.g., "Carex_bobo  "
            formatted_name = f"{record.id:<10}" 
            
            # Get the first block of the sequence
            seq_block = str(record.seq[:block_size])
            
            # Write the name, TWO spaces, and the first sequence block
            f.write(f"{formatted_name}  {seq_block}") 
            
            # If this is not the last sequence, print the remaining sequence blocks for the first sequence
            # PAML reads the first line for the name, and subsequent blocks for the rest of the sequence
            for start_pos in range(block_size, aln_length, block_size):
                # Write a block of sequence, preceded by one space
                seq_block = str(record.seq[start_pos : start_pos + block_size])
                f.write(f" {seq_block}")
            
            # Move to the next line for the next sequence's name and sequence blocks
            f.write("\n")


def prepare_paml_input():
    """
    Cleans up the directory, converts FASTA alignments to custom PHYLIP format, 
    shortening sequence IDs to 10 characters, and generates gene-specific 
    codeml control files for PAML analysis.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Clean up the empty file
    if os.path.exists(EMPTY_FILE_TO_REMOVE):
        os.remove(EMPTY_FILE_TO_REMOVE)
        print(f"Removed empty file: {EMPTY_FILE_TO_REMOVE}")
    
    # Write the shared tree file
    tree_path = os.path.join(OUTPUT_DIR, TREE_FILE_NAME)
    with open(tree_path, 'w') as f:
        f.write(TREE_FILE_CONTENT)
    print(f"Created shared tree file: {tree_path}")

    success_count = 0
    
    # 2. Convert and Generate Control Files
    print("-" * 40)
    print("Converting alignments and generating control files...")

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".final.fasta"):
            gene_id = filename.replace(".final.fasta", "")
            fasta_input_path = os.path.join(INPUT_DIR, filename)
            phylip_output_path = os.path.join(OUTPUT_DIR, f"{gene_id}.phy")
            ctl_output_path = os.path.join(OUTPUT_DIR, f"{gene_id}.ctl")

            try:
                # Read FASTA records
                records = list(SeqIO.parse(fasta_input_path, "fasta"))

                # CRITICAL FIX: Force Short IDs for PAML/PHYLIP compatibility.
                # The sequences are *always* C. borbonica and C. boryana orthologs.
                if len(records) == 2:
                    # Assign the fixed, 10-character names to match the TREE_FILE_CONTENT
                    # We assume records[0] is borbonica and records[1] is boryana, 
                    # based on standard ortholog extraction pipeline output.
                    records[0].id = "Carex_bobo"
                    records[0].name = "Carex_bobo"
                    records[0].description = ""
                    
                    records[1].id = "Carex_bory"
                    records[1].name = "Carex_bory"
                    records[1].description = ""
                    
                    shortened_records = records
                else:
                    print(f"ERROR: Alignment file {filename} does not contain exactly two sequences ({len(records)} found). Skipping.")
                    continue

                # 3. Write custom PAML-compatible PHYLIP file
                # The custom function ensures strict 10-char name padding and sequence blocking.
                write_paml_phylip(shortened_records, phylip_output_path)

                # 4. Generate the gene-specific codeml control file (.ctl)
                generate_control_file(
                    ctl_output_path, 
                    phylip_output_path, 
                    tree_path, 
                    gene_id
                )
                success_count += 1
                
            except Exception as e:
                print(f"FAILED to process {gene_id}: {e}")

    print("-" * 40)
    print(f"Successfully prepared {success_count} gene inputs in the '{OUTPUT_DIR}/' directory.")

def generate_control_file(ctl_path, aln_path, tree_path, gene_id):
    """Generates the codeml control file for the simple two-sequence dN/dS model (Model 0)."""
    
    # We use the Model 0 (one omega ratio for all branches) as the standard first step 
    # for pairwise dN/dS calculation.
    ctl_content = f"""
seqfile = {os.path.basename(aln_path)}
outfile = {gene_id}.mlc
treefile = {os.path.basename(tree_path)}

# Model parameters
noisy = 3       # 0: no messages, 3: detailed output
verbose = 1     # 1: brief results
runmode = 0     # 0: user tree (fixed)

# General settings
seqtype = 1     # 1: codon sequences
CodonFreq = 2   # F3x4 model of codon frequencies
clock = 0       # 0: no molecular clock

# Selection model
model = 0       # 0: one dN/dS ratio for all branches (pairwise calculation)

# Starting values (estimates will be refined)
NSsites = 0     # Site-specific models are NOT used here (using only branch model 0)
icode = 0       # 0: Universal genetic code (Standard)
fix_kappa = 0   # Estimate transition/transversion ratio (kappa)
kappa = 2.0     # Initial kappa value
fix_omega = 0   # Estimate dN/dS ratio (omega)
omega = 0.5     # Initial omega value

# Optimization settings
Small_Diff = 0.5e-6
cleandata = 1   # Remove sites with alignment gaps or ambiguous characters
"""
    
    with open(ctl_path, 'w') as f:
        f.write(ctl_content.strip())


if __name__ == "__main__":
    prepare_paml_input()
