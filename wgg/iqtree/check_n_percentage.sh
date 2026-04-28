#!/bin/bash

# Define the fixed alignment file path
ALIGNMENT_FILE="/home/aescudero/wgg/bams/all_samples_polymorphic_snps_mac20.min4.phy.varsites.phy.fixed.phy"

echo "Calculating N and gap percentage for each sequence in: $ALIGNMENT_FILE"
echo "---------------------------------------------------------------------"

# Check if the alignment file exists
if [ ! -f "$ALIGNMENT_FILE" ]; then
    echo "ERROR: Alignment file not found: $ALIGNMENT_FILE. Exiting."
    exit 1
fi

# Get total number of sites/columns from the header (second number on the first line)
TOTAL_COLS=$(head -n 1 "$ALIGNMENT_FILE" | awk '{print $2}')

if [ "$TOTAL_COLS" -eq 0 ]; then
    echo "Error: Alignment has 0 columns. Cannot calculate percentages."
    exit 1
fi

echo "Total alignment length (columns): $TOTAL_COLS"
echo "---------------------------------------------------------------------"
echo "Sequence Name           | N/Gap Count | % N/Gap"
echo "------------------------|-------------|----------"

# Loop through each sequence (skip the header line using tail -n +2)
tail -n +2 "$ALIGNMENT_FILE" | awk -v total_cols="$TOTAL_COLS" '{
    name = $1;
    # Extract the sequence part (everything after the name and first space)
    sequence = substr($0, length(name) + 2);
    
    # Remove all spaces from the sequence string to get only bases/gaps/Ns
    gsub(" ", "", sequence);

    # Count occurrences of N or - (case-insensitive for N if needed, but PHYLIP is usually uppercase)
    n_gap_count = 0;
    for (i = 1; i <= length(sequence); i++) {
        char = substr(sequence, i, 1);
        if (char == "N" || char == "n" || char == "-") {
            n_gap_count++;
        }
    }
    
    # Calculate percentage
    percent = (n_gap_count / total_cols) * 100;

    # Print formatted output
    printf "%-23s | %11d | %8.2f%%\n", name, n_gap_count, percent;
}' | sort -k3,3nr # Sorts by percentage (3rd column) in reverse numerical order (highest first)
