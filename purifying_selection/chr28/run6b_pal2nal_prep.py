import os
import subprocess
import time
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

# --- Configuration ---
PEP_FASTA_1 = "Carex_borbonica.faa"
PEP_FASTA_2 = "Carex_boryana.faa"
MAP_FILE = "Carex_boryana_vs_Carex_borbonica.synHits.txt"

# Directories from previous steps
FILTERED_NUCLEOTIDE_DIR = "filtered_alignments_OUTSIDE"
FINAL_ORTHOLOG_LIST = "final_ortholog_list_OUTSIDE.txt"

# New output directory for the final, clean alignments (ready for PAML/HyPhy)
OUTPUT_DIR = "codon_aligned_dnds_OUTSIDE"

# Temporary directories for intermediate files
TEMP_PEP_ALIGN_DIR = "temp_pep_alignments_OUTSIDE" 
# ---------------------

def get_base_gene_id(full_id):
    """
    Extracts the base gene ID (e.g., 'g4919') from a full ID string 
    by splitting on common separators like '|', '.', and space.
    """
    parts = full_id.split('|')[0].split('.')[0].split(' ')
    return parts[0].strip()

def load_fasta_to_dict(filepath):
    """Loads a FASTA file into a dictionary mapping ID to sequence record."""
    print(f"Loading sequences from {filepath}...")
    try:
        # Use the base gene ID as the key for easy lookup
        return {get_base_gene_id(rec.id): rec for rec in SeqIO.parse(filepath, "fasta")}
    except FileNotFoundError:
        print(f"FATAL ERROR: Input file '{filepath}' not found.")
        return {}

def load_ortholog_map(map_file):
    """
    Loads the ortholog map from the synHits file.
    Maps: Carex_borbonica_ID (Key) -> Carex_boryana_ID (Value).
    """
    mapping = {}
    try:
        with open(map_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                # Skip the header line based on the non-gene-ID content in the first part (e.g., 'ofID1')
                if parts[0].isalpha() and parts[0] == "ofID1":
                    continue
                
                # The synHits file is space/tab-separated.
                # Carex_boryana gene ID (id1) is at index 4 (5th column).
                # Carex_borbonica gene ID (id2) is at index 12 (13th column).
                if len(parts) >= 13:
                    # Key: Borbonica ID (the ID in the final_ortholog_list and the alignment file names)
                    id_borbonica = get_base_gene_id(parts[12]) 
                    # Value: Boryana ID (the partner sequence needed from the .faa file)
                    id_boryana = get_base_gene_id(parts[4])
                    
                    # Store the mapping
                    mapping[id_borbonica] = id_boryana
                else:
                    print(f"Warning: Skipping line in map file due to too few columns: {line[:50]}...")
                    
    except FileNotFoundError:
        print(f"FATAL ERROR: Ortholog map file '{map_file}' not found.")
        return {}
        
    print(f"Loaded {len(mapping)} ortholog pairs.")
    return mapping

def run_pal2nal_prep():
    """Performs protein alignment and PAL2NAL codon alignment for all filtered genes."""
    
    if not os.path.exists(FINAL_ORTHOLOG_LIST):
        print(f"FATAL ERROR: Required file '{FINAL_ORTHOLOG_LIST}' not found. Did the filtering step run correctly?")
        return
        
    if not os.path.exists(FILTERED_NUCLEOTIDE_DIR):
        print(f"FATAL ERROR: Required directory '{FILTERED_NUCLEOTIDE_DIR}' not found. Please check your path.")
        return

    # Load resources
    # pep_dict_1 = Borbonica (ID from final list)
    # pep_dict_2 = Boryana (Partner ID)
    pep_dict_1 = load_fasta_to_dict(PEP_FASTA_1)
    pep_dict_2 = load_fasta_to_dict(PEP_FASTA_2)
    ortholog_map = load_ortholog_map(MAP_FILE)
    
    # Get the list of IDs that passed filtering
    with open(FINAL_ORTHOLOG_LIST, 'r') as f:
        base_gene_ids = [get_base_gene_id(line.strip()) for line in f if line.strip()]

    # Setup directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_PEP_ALIGN_DIR, exist_ok=True)

    total_genes = len(base_gene_ids)
    success_count = 0
    
    print(f"Starting frame-check and alignment cleanup for {total_genes} genes...")
    print("-" * 30)

    for i, gene_id_1 in enumerate(base_gene_ids):
        
        # gene_id_1 is the Carex_borbonica ID from the list
        gene_id_2 = ortholog_map.get(gene_id_1) # This is the Carex_boryana partner ID
        
        # --- Check for missing resources ---
        if not gene_id_2:
            # This should now only happen if the ID in the final_ortholog_list.txt 
            # was not found in the Carex_borbonica column (id2) of the map file.
            print(f"[{i+1}/{total_genes}] Skipping {gene_id_1}: Partner not found in ortholog map.") 
            continue
            
        # Get protein sequences (using base IDs for lookup)
        pep_rec_1 = pep_dict_1.get(gene_id_1)
        pep_rec_2 = pep_dict_2.get(gene_id_2)

        if not pep_rec_1 or not pep_rec_2:
            print(f"[{i+1}/{total_genes}] Skipping {gene_id_1}: Could not find protein sequences for pair (IDs: {gene_id_1}, {gene_id_2}).")
            continue

        # Check for nucleotide alignment file (using base ID)
        nucleotide_input_path = os.path.join(FILTERED_NUCLEOTIDE_DIR, f"{gene_id_1}.aln.fasta")
        if not os.path.exists(nucleotide_input_path):
            print(f"[{i+1}/{total_genes}] SKIPPING {gene_id_1}: Nucleotide alignment NOT FOUND at '{nucleotide_input_path}'")
            continue

        # --- STEP 1: Run MAFFT on Protein Sequences ---
        pep_input_path = os.path.join(TEMP_PEP_ALIGN_DIR, f"{gene_id_1}.unaligned.pep.fasta")
        pep_output_path = os.path.join(TEMP_PEP_ALIGN_DIR, f"{gene_id_1}.pep.aln.fasta")

        try:
            # 1. Read full headers from the filtered nucleotide alignment (g4919.aln.fasta)
            # This ensures the headers match exactly for PAL2NAL
            nucleotide_records = list(SeqIO.parse(nucleotide_input_path, "fasta"))
            
            # 2. Rename the unaligned protein records to match the *full* nucleotide headers
            pep_rec_1.id = nucleotide_records[0].id
            pep_rec_2.id = nucleotide_records[1].id
            pep_rec_1.description = ""
            pep_rec_2.description = ""
            
            # 3. Write unaligned protein sequences for MAFFT
            SeqIO.write([pep_rec_1, pep_rec_2], pep_input_path, "fasta")

            # 4. Run MAFFT
            mafft_command = ["mafft", "--quiet", pep_input_path]
            with open(pep_output_path, "w") as outfile:
                subprocess.run(mafft_command, check=True, stdout=outfile, stderr=subprocess.PIPE)
        
        except subprocess.CalledProcessError as e:
            print(f"[{i+1}/{total_genes}] MAFFT FAILED for {gene_id_1}. Error: {e.stderr.decode().strip()}")
            continue
        except FileNotFoundError:
             print(f"FATAL ERROR: MAFFT command not found. Ensure 'mafft' is in your PATH.")
             return
        except IndexError:
             print(f"[{i+1}/{total_genes}] SKIPPING {gene_id_1}: Nucleotide alignment file '{nucleotide_input_path}' did not contain exactly two sequences.")
             continue


        # --- STEP 2: Run PAL2NAL for Codon Alignment ---
        final_output_path = os.path.join(OUTPUT_DIR, f"{gene_id_1}.final.fasta")

        try:
            pal2nal_command = [
                "pal2nal.pl", 
                pep_output_path,          # Protein alignment
                nucleotide_input_path,    # Nucleotide sequences (unaligned or aligned, but must match proteins)
                "-output", "fasta",
                "-nogap" 
            ]
            
            result = subprocess.run(pal2nal_command, check=True, capture_output=True, text=True)
            
            with open(final_output_path, 'w') as f:
                f.write(result.stdout)
                
            success_count += 1
            print(f"[{i+1}/{total_genes}] SUCCESS: Processed {gene_id_1}. Final file size: {os.path.getsize(final_output_path)} bytes.")

        except subprocess.CalledProcessError as e:
            # Handles PAL2NAL errors (e.g., reading frame issues, non-triplet gaps)
            print(f"[{i+1}/{total_genes}] PAL2NAL FAILED for {gene_id_1}. Error: {e.stderr.decode().strip()}")
        except FileNotFoundError:
            print(f"FATAL ERROR: pal2nal.pl command not found. Ensure 'pal2nal.pl' is in your PATH.")
            return
        except Exception as e:
            print(f"[{i+1}/{total_genes}] AN UNEXPECTED ERROR occurred for {gene_id_1}: {e}")

    # Final Cleanup and Summary
    try:
        if os.path.exists(TEMP_PEP_ALIGN_DIR) and not os.listdir(TEMP_PEP_ALIGN_DIR):
            os.rmdir(TEMP_PEP_ALIGN_DIR)
    except OSError:
        pass 
        
    print("\n" + "=" * 40)
    print(f"--- PAL2NAL Summary (ORTHOLOG MAP FIXED) ---")
    print(f"Total genes processed: {total_genes}")
    print(f"Final, codon-aligned genes saved: {success_count}")
    print(f"Output saved to the '{OUTPUT_DIR}/' directory.")
    print("=" * 40)


if __name__ == "__main__":
    run_pal2nal_prep()

