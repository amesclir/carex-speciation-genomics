import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# --- Configuration ---
INPUT_DIR = "aligned_pairs_INSIDE"
OUTPUT_DIR = "filtered_alignments_INSIDE"
FINAL_LIST_FILE = "final_ortholog_list_INSIDE.txt"

MIN_ALIGNMENT_LENGTH_BP = 300  # Minimum nucleotide length (100 codons)
MAX_GAP_PERCENTAGE = 0.10      # Maximum gap percentage allowed in either sequence (10%)
# ---------------------

def calculate_gap_percentage(sequence):
    """Calculates the percentage of gaps ('-' or '?') in a sequence."""
    total_length = len(sequence)
    if total_length == 0:
        return 1.0
    gap_count = sequence.count('-') + sequence.count('?')
    return gap_count / total_length

def filter_and_prepare_alignments():
    """Reads, filters, and writes passing alignments for dN/dS analysis."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".aln.fasta")]
    
    total_alignments = len(all_files)
    passed_count = 0
    final_ortholog_ids = []
    
    print(f"Checking {total_alignments} alignments from '{INPUT_DIR}/'...")

    for filename in all_files:
        input_path = os.path.join(INPUT_DIR, filename)
        
        # The gene ID is the base filename before the first '.'
        gene_id = filename.split('.')[0] 
        
        try:
            # SeqIO.parse returns an iterable; we expect exactly two sequences
            records = list(SeqIO.parse(input_path, "fasta"))
            
            if len(records) != 2:
                print(f"Skipping {gene_id}: Alignment does not contain exactly 2 sequences.")
                continue

            seq1 = str(records[0].seq).upper()
            seq2 = str(records[1].seq).upper()
            
            # --- 1. Length Check ---
            if len(seq1) < MIN_ALIGNMENT_LENGTH_BP:
                # Note: Because the sequences are aligned, len(seq1) == len(seq2)
                print(f"Skipping {gene_id}: Alignment too short ({len(seq1)} bp).")
                continue
            
            # --- 2. Gap Check ---
            gap_percent1 = calculate_gap_percentage(seq1)
            gap_percent2 = calculate_gap_percentage(seq2)
            
            if gap_percent1 > MAX_GAP_PERCENTAGE or gap_percent2 > MAX_GAP_PERCENTAGE:
                print(f"Skipping {gene_id}: Excessive gaps (Seq1: {gap_percent1:.2f}, Seq2: {gap_percent2:.2f}).")
                continue

            # --- 3. Write Filtered Alignment ---
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            # The sequences already contain headers from the alignment script, 
            # so we just need to write them back out.
            SeqIO.write(records, output_path, "fasta")
            
            final_ortholog_ids.append(gene_id)
            passed_count += 1

        except Exception as e:
            print(f"Error processing {gene_id}: {e}")

    # --- Summary and Final List Generation ---
    with open(FINAL_LIST_FILE, 'w') as f:
        f.write('\n'.join(final_ortholog_ids))

    print("\n--- Filtering Summary ---")
    print(f"Total input alignments: {total_alignments}")
    print(f"Alignments passed filtering: {passed_count}")
    print(f"Filtered alignments saved to '{OUTPUT_DIR}/'")
    print(f"List of final ortholog IDs saved to '{FINAL_LIST_FILE}'")
    print("-" * 30)


if __name__ == "__main__":
    print("--- Starting Alignment Filtering and Preparation: Step 5 ---")
    filter_and_prepare_alignments()
    
    print("Next step is to perform frame check using a tool like PAL2NAL or run dN/dS analysis directly.")

