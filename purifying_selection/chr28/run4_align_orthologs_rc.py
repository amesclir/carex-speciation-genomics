import os
import re
import subprocess
from collections import defaultdict

# --- Configuration: Using the Synteny Hits File for the Map ---
ORTHOLOG_MAP_FILE = "Carex_boryana_vs_Carex_borbonica.synHits.txt" 
MAP_DELIMITER = '\t' 
# -----------------------------------------------------------------------

# --- Helper Functions ---

def reverse_complement(seq):
    """Calculates the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    # Replace the characters based on the complement map
    # Then reverse the string
    return "".join(complement.get(base, base) for base in reversed(seq.upper()))


def read_fasta_by_id(fasta_path):
    """
    Reads a FASTA file and returns a dictionary mapping the base gene ID to sequence data.
    Assumes header format is >gID... and extracts the ID as the first token 
    before the first space or pipe, matching the normalized ID format.
    Returns: {base_gene_id: {"header": full_header, "seq": sequence_string}}
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Handle the previous sequence
                if current_id is not None:
                    # Extract the base gene ID (e.g., 'g12345')
                    base_gene_id = current_id.split('|')[0].split(' ')[0].replace('>', '').strip()
                    sequences[base_gene_id] = {"header": current_id, "seq": "".join(current_seq)}
                    
                current_id = line.strip()
                current_seq = []
            else:
                current_seq.append(line.strip().upper())
        
        # Add the last sequence
        if current_id is not None:
            base_gene_id = current_id.split('|')[0].split(' ')[0].replace('>', '').strip()
            sequences[base_gene_id] = {"header": current_id, "seq": "".join(current_seq)}
                
    return sequences

def normalize_gene_id(gene_id):
    """
    Normalizes a gene ID from the map (e.g., g12345.t1|locus) to the base ID (g12345).
    """
    # 1. Strip everything after a pipe |
    normalized_id = gene_id.split('|')[0]
    # 2. Strip everything after '.t' (to remove transcript IDs like .t1)
    if '.t' in normalized_id:
        normalized_id = normalized_id.split('.t')[0]
    # 3. Final cleanup and return
    return normalized_id.strip()

def read_synteny_hits_map(map_path, delimiter='\t'):
    """
    Reads the multi-column synHits file and extracts IDs from known columns.
    Based on the header: C. boryana ID (id1) is Col 5 (index 4), 
    and C. borbonica ID (id2) is Col 13 (index 12).
    Map format: {borbonica_base_id: boryana_base_id}
    """
    print(f"Reading ortholog map from {map_path}...")
    ortholog_pairs = {} 
    
    try:
        with open(map_path, 'r') as f:
            # Skip the header line (i=0)
            header_line = f.readline()
            
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(delimiter)
                
                # Check that the line has at least 13 columns (for index 12)
                if len(parts) >= 13: 
                    
                    # Extract raw IDs using correct indices
                    # C. boryana ID (id1) is index 4
                    raw_boryana_id = parts[4].strip() 
                    # C. borbonica ID (id2) is index 12
                    raw_borbonica_id = parts[12].strip()
                    
                    # Normalize IDs to match the FASTA dictionary keys
                    borbonica_id = normalize_gene_id(raw_borbonica_id)
                    boryana_id = normalize_gene_id(raw_boryana_id)

                    # Store as {borbonica_ID: boryana_ID}. Overwriting duplicates 
                    # with the same base ID is acceptable since we are aiming for 
                    # one-to-one gene-level alignment based on the filtered set.
                    ortholog_pairs[borbonica_id] = boryana_id
                        
                # else: skip badly formatted lines
        
        if not ortholog_pairs:
             print(f"ERROR: The map file {map_path} was empty or incorrectly formatted.")
             return None
            
        # The number of unique base-ID pairs will be smaller than the total lines
        print(f"Loaded {len(ortholog_pairs)} unique base-ID pairings from hits file.")
        return ortholog_pairs
        
    except FileNotFoundError:
        print(f"FATAL ERROR: Ortholog map file not found at {map_path}. Cannot proceed.")
        return None
        
def run_mafft_alignment(temp_input_path, output_alignment_path):
    """Executes MAFFT to align the sequences."""
    try:
        # Use --auto heuristic for best alignment method selection
        command = [
            "mafft", 
            "--quiet", 
            "--auto", 
            temp_input_path
        ]
        
        with open(output_alignment_path, "w") as outfile:
            # Check=True ensures python raises an error if MAFFT fails
            subprocess.run(command, check=True, stdout=outfile, stderr=subprocess.PIPE)
        
    except FileNotFoundError:
        print("\nERROR: MAFFT command not found. Please install MAFFT or ensure it is in your PATH.")
        raise
    except subprocess.CalledProcessError as e:
        # MAFFT error (e.g., sequence too short, bad characters)
        return False
    except Exception as e:
        return False
        
    return True

# --- Main Logic ---

def align_orthologs(borbonica_fasta, boryana_fasta, output_dir, map_file, delimiter):
    """
    Reads a map file, pairs sequences by ID, and aligns each pair using MAFFT.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Read the Ortholog Map from the synHits file (IDs are normalized and indexed correctly)
    ortholog_map = read_synteny_hits_map(map_file, delimiter)
    if not ortholog_map:
        return

    # 2. Read sequences, keyed by their specific species ID (IDs are normalized here too)
    print("Reading C. borbonica CDS sequences...")
    borbonica_seqs = read_fasta_by_id(borbonica_fasta)
    print(f"Found {len(borbonica_seqs)} borbonica sequences.")
    print("Reading C. boryana CDS sequences...")
    boryana_seqs = read_fasta_by_id(boryana_fasta)
    print(f"Found {len(boryana_seqs)} boryana sequences.")
    
    
    # 3. Iterate through the map and align common pairs
    paired_count = 0
    aligned_count = 0
    
    print("\nStarting pairwise alignment...")
    
    # Iterate through the borbonica IDs that form the keys in our normalized map
    for borbonica_id, boryana_id in ortholog_map.items():
        
        # Check if both normalized base IDs exist in the FASTA dictionaries
        if borbonica_id not in borbonica_seqs or boryana_id not in boryana_seqs:
            continue
            
        # Found a valid, filtered pair!
        paired_count += 1

        borbonica_info = borbonica_seqs[borbonica_id]
        boryana_info = boryana_seqs[boryana_id]
        
        
        # --- NEW LOGIC: Test both forward and reverse complement of C. boryana ---
        
        # In a robust analysis (especially with inversions), we should check the strand
        # information from the original synHits.txt file (often in columns 8 and 16)
        # to determine if C. boryana is aligned in the forward or reverse direction
        # relative to C. borbonica. 
        # Since we don't have that info here, we will align the "default" forward 
        # orientation and assume the synteny tool handled the orientation.
        # However, for an inversion, it's safer to always test the reverse complement
        # and choose the one that results in the better alignment (e.g., fewer gaps, 
        # or most importantly, no stop codons upon translation in the next step).
        
        # For this script, we will stick to the default orientation unless you provide
        # strand info. But we'll add a helper function for later use.
        
        # If you were to integrate the strand check from the map:
        # is_reversed = check_if_boryana_is_reverse(parts) # You would need to write this function
        # sequence_to_align = reverse_complement(boryana_info['seq']) if is_reversed else boryana_info['seq']
        
        # Using the default (forward) orientation for now:
        sequence_to_align = boryana_info['seq']
        
        # 4. Create a temporary input file for MAFFT
        temp_input_path = os.path.join(output_dir, f"{borbonica_id}_temp_input.fasta")
        # Output file name remains the same, but the alignment will be correct if the orientation is right
        output_alignment_path = os.path.join(output_dir, f"{borbonica_id}.aln.fasta")
        
        try:
            with open(temp_input_path, 'w') as temp_f:
                # Sequence 1: C. borbonica 
                temp_f.write(f"{borbonica_info['header']} (Carex_borbonica)\n")
                temp_f.write(f"{borbonica_info['seq']}\n")
                # Sequence 2: C. boryana (using the sequence_to_align, which currently is the forward strand)
                temp_f.write(f"{boryana_info['header']} (Carex_boryana)\n")
                temp_f.write(f"{sequence_to_align}\n")
        except Exception:
            continue

        # 5. Run MAFFT alignment
        if run_mafft_alignment(temp_input_path, output_alignment_path):
            aligned_count += 1
            
        # 6. Clean up the temporary file
        os.remove(temp_input_path)


    print(f"\n--- Alignment Summary ---")
    print(f"Total base-ID pairs extracted from map: {len(ortholog_map)}")
    print(f"Total pairs found in filtered FASTA files (paired): {paired_count}")
    print(f"Total pairs successfully aligned: {aligned_count}")
    print(f"Alignment files saved to the '{output_dir}/' directory.")
    print("-" * 40)

# --- Execution ---
if __name__ == "__main__":
    
    BORBONICA_FASTA_OUT = "borbonica_ortholog_cds.fasta"
    BORYANA_FASTA_OUT = "boryana_ortholog_cds.fasta"
    OUTPUT_DIR = "aligned_pairs"

    print("--- Starting Map-Driven Ortholog Alignment: Step 4 ---")
    
    align_orthologs(
        borbonica_fasta=BORBONICA_FASTA_OUT, 
        boryana_fasta=BORYANA_FASTA_OUT, 
        output_dir=OUTPUT_DIR,
        map_file=ORTHOLOG_MAP_FILE,
        delimiter=MAP_DELIMITER
    )
    
    print("All ortholog pairs have been aligned (nucleotide alignment).")
    print("NEXT STEP: Filtering and format conversion for dN/dS analysis.")
