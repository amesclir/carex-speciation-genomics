#!/bin/bash
# Checks the line counts and the structure of the critical files.

PRUNE_IN="/home/aescudero/wgg/admixture_data/admixture_ld_temp_pruned.prune.in"
CLEAN_EXTRACT="/home/aescudero/wgg/admixture_data/admixture_ld_temp_pruned_clean.extract"
INPUT_BIM="/home/aescudero/wgg/admixture_data/admixture_all_samples_mac20_no_outgroups_sorted.bim"

echo "--- Diagnostic Check (V6) ---"
date

# 1. Check the original LD-pruned list (.prune.in)
echo -e "\n--- 1. Checking ORIGINAL PRUNED LIST: ${PRUNE_IN} ---"
# Count the lines in the original file
echo "Line Count (wc -l): $(wc -l < "${PRUNE_IN}")"
echo "Byte Count (wc -c): $(wc -c < "${PRUNE_IN}")"
echo "--- Viewing first 5 lines (with hidden characters shown) ---"
# cat -vET shows hidden characters: $ for newline, ^I for tab, etc.
# This should tell us if all 827k IDs are on line 1.
cat -vET "${PRUNE_IN}" | head -n 5

# 2. Check the input source BIM file
echo -e "\n--- 2. Checking INPUT BIM FILE: ${INPUT_BIM} ---"
echo "Total Variants in Source BIM: $(wc -l < "${INPUT_BIM}")"
echo "--- First 3 IDs in Source BIM ---"
# Show the second column (SNP ID)
awk '{print $2}' "${INPUT_BIM}" | head -n 3

# 3. Check the intermediate 'clean' list created by V6
echo -e "\n--- 3. Checking CLEAN EXTRACT LIST: ${CLEAN_EXTRACT} ---"
echo "Clean List Count: $(wc -l < "${CLEAN_EXTRACT}")"
echo "--- Contents of the Clean List ---"
cat "${CLEAN_EXTRACT}"

echo -e "\n--- Diagnostics Complete ---"
