# Save this entire content into a file named reorder_rename_fasta.py
from Bio import SeqIO
import sys
import os

def reorder_and_rename_fasta_based_on_correspondence(input_fasta_path, output_fasta_path="reordered_and_renamed_cborb_final.fasta"):
    """
    Reorders and renames sequences in input_fasta_path based on a specific
    correspondence list derived from genomcbory.fa's desired order and cborb's original IDs.
    """

    # This list defines the *desired order and new names* for the output file
    # (corresponds to the perfect numerical order of genomcbory.fa)
    desired_output_scaffolds = [
        "scaffold_1", "scaffold_2", "scaffold_3", "scaffold_4", "scaffold_5",
        "scaffold_6", "scaffold_7", "scaffold_8", "scaffold_9", "scaffold_10",
        "scaffold_11", "scaffold_12", "scaffold_13", "scaffold_14", "scaffold_15",
        "scaffold_16", "scaffold_17", "scaffold_18", "scaffold_19", "scaffold_20",
        "scaffold_21", "scaffold_22", "scaffold_23", "scaffold_24", "scaffold_25",
        "scaffold_26", "scaffold_27", "scaffold_28", "scaffold_29", "scaffold_30",
        "scaffold_31", "scaffold_32", "scaffold_33", "scaffold_34"
    ]

    # This list defines which *original ID from genomecborb_rev.fa*
    # should be placed at each position in the output, and subsequently renamed.
    # Its order directly corresponds to desired_output_scaffolds.
    # For example, desired_output_scaffolds[0] (scaffold_1) should get the sequence
    # that is currently named original_cborb_ids_in_correspondence[0] (scaffold_1)
    original_cborb_ids_in_correspondence = [
        "scaffold_1", "scaffold_3", "scaffold_2", "scaffold_6", "scaffold_4",
        "scaffold_7", "scaffold_9", "scaffold_5", "scaffold_8", "scaffold_10",
        "scaffold_11", "scaffold_12", "scaffold_13", "scaffold_15", "scaffold_14",
        "scaffold_18", "scaffold_17", "scaffold_16", "scaffold_20", "scaffold_19",
        "scaffold_21", "scaffold_23", "scaffold_22", "scaffold_26", "scaffold_25",
        "scaffold_24", "scaffold_27", "scaffold_29", "scaffold_28", "scaffold_31",
        "scaffold_32", "scaffold_34", "scaffold_33", "scaffold_30"
    ]

    if len(desired_output_scaffolds) != len(original_cborb_ids_in_correspondence):
        print("Error: The two correspondence lists must have the same number of elements.", file=sys.stderr)
        sys.exit(1)

    # Step 1: Read all sequences from genomecborb_rev.fa into a dictionary for quick lookup.
    # Keys will be the original IDs (e.g., "scaffold_2", "scaffold_5" etc.)
    cborb_rev_records = {}
    try:
        with open(input_fasta_path, "r") as infile:
            for record in SeqIO.parse(infile, "fasta"):
                cborb_rev_records[record.id] = record
        print(f"Read {len(cborb_rev_records)} records from '{input_fasta_path}'")
    except FileNotFoundError:
        print(f"Error: Input FASTA file '{input_fasta_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading input FASTA: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Build the list of records for the output file in the desired order and with new names.
    output_records = []
    missing_original_ids = []

    for i in range(len(desired_output_scaffolds)):
        new_name_for_output = desired_output_scaffolds[i]
        # Get the original ID from genomecborb_rev.fa that corresponds to this new name/position
        original_id_to_fetch = original_cborb_ids_in_correspondence[i]

        if original_id_to_fetch in cborb_rev_records:
            record_to_add = cborb_rev_records[original_id_to_fetch]
            
            # Crucially, rename the record's ID to the desired output name
            record_to_add.id = new_name_for_output
            # Clear description if you want clean headers like ">scaffold_1"
            record_to_add.description = ""
            
            output_records.append(record_to_add)
        else:
            missing_original_ids.append(original_id_to_fetch)
            print(f"Warning: Original ID '{original_id_to_fetch}' (which should map to '{new_name_for_output}') not found in '{input_fasta_path}'. This sequence will be skipped.", file=sys.stderr)

    # Step 3: Write the reordered and renamed sequences to the new FASTA file.
    if not output_records:
        print("No records to write. Check your input file and correspondence lists.", file=sys.stderr)
        return

    try:
        with open(output_fasta_path, "w") as outfile:
            SeqIO.write(output_records, outfile, "fasta")
        print(f"Successfully reordered and renamed {len(output_records)} sequences to '{output_fasta_path}'")
    except IOError as e:
        print(f"Error writing to output file '{output_fasta_path}': {e}", file=sys.sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during writing: {e}", file=sys.stderr)

if __name__ == "__main__":
    # Ensure Biopython is installed: `conda install biopython` or `pip install biopython`
    # Activate your relevant conda environment before running: `conda activate syri` (or similar)

    if len(sys.argv) < 2:
        print("Usage: python reorder_rename_script.py <input_genomecborb_rev.fa> [output_fasta_file]")
        sys.exit(1)

    input_fasta_file = sys.argv[1]
    output_fasta_file = sys.argv[2] if len(sys.argv) > 2 else "reordered_and_renamed_cborb_final.fasta"

    reorder_and_rename_fasta_based_on_correspondence(input_fasta_file, output_fasta_file)
