#!/bin/bash
#SBATCH --job-name=fst_analysis
#SBATCH --output=fst_analysis_%j.out
#SBATCH --error=fst_analysis_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=long

set -e

echo "--- Starting Fst Analysis Workflow ---"
date +"%Y-%m-%d %H:%M:%S"

# --- Configuration ---
CONDA_ENV_NAME="bcftools"
PLINK_INPUT_PREFIX="/home/aescudero/wgg/admixture_data/admixture_final_850k_pruned"
PLINK_CLEANED_PREFIX="/home/aescudero/wgg/admixture_data/admixture_final_ld_pruned_cleaned"
PLINK_FILTERED_PREFIX="/home/aescudero/wgg/admixture_data/admixture_final_ld_pruned_filtered"

# Original population file with a header and two columns (IID and POP)
POP_FILE_RAW="/home/aescudero/wgg/admixture_data/population_list.txt"

# New file to be created with the correct format for PLINK --keep and --within
POP_FILE_FORMATTED="/home/aescudero/wgg/admixture_data/population_list_formatted.txt"

UPDATE_IDS_FILE="/home/aescudero/wgg/admixture_data/update_ids.txt"
FST_OUTPUT_DIR="/home/aescudero/wgg/fst_results"

# --- Activate Conda environment ---
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source "/home/aescudero/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}" || { echo "Error: Failed to activate Conda environment '${CONDA_ENV_NAME}'. Exiting." >&2; exit 1; }
echo "Conda environment '${CONDA_ENV_NAME}' activated."

# --- Create Output Directories ---
echo "Creating output directories..."
mkdir -p "${FST_OUTPUT_DIR}"

# --- Pre-processing Step ---
## We need to format the population list correctly for PLINK.
## PLINK requires the --keep and --within files to be in a specific format.
## We will create a new file that has the FID, IID, and POP columns.
echo "Formatting population list for PLINK..."
awk 'NR>1 {print $1, $1, $2}' "${POP_FILE_RAW}" > "${POP_FILE_FORMATTED}"

# -------------------------------------------------------------------------------------------------------------------------

## Step 1: Create the ID update file to fix the long file paths and extensions
echo "Generating ID update file to clean sample IDs..."
cut -f1,2 "${PLINK_INPUT_PREFIX}.fam" | \
awk 'BEGIN{OFS="\t"}{
    cleaned_name = gensub(/.*\//, "", "g", $2);
    cleaned_name = gensub(/\.bam$/, "", "g", cleaned_name);
    print $1, $2, cleaned_name, cleaned_name;
}' > "${UPDATE_IDS_FILE}"

## Step 2: Update the IDs in the PLINK file set
echo "Updating sample IDs in the PLINK file set..."
plink --bfile "${PLINK_INPUT_PREFIX}" \
      --update-ids "${UPDATE_IDS_FILE}" \
      --make-bed \
      --out "${PLINK_CLEANED_PREFIX}" \
      --chr-set 34 no-xy

# -------------------------------------------------------------------------------------------------------------------------

## Step 3: Filter the cleaned dataset using the population list
echo "Filtering samples to exclude outgroups and low-coverage samples..."
plink --bfile "${PLINK_CLEANED_PREFIX}" \
      --keep "${POP_FILE_FORMATTED}" \
      --make-bed \
      --out "${PLINK_FILTERED_PREFIX}" \
      --chr-set 34 no-xy

# -------------------------------------------------------------------------------------------------------------------------

## Step 4: Run Fst Calculation on the final filtered dataset
echo "Running Fst calculation on the final filtered data..."
plink --bfile "${PLINK_FILTERED_PREFIX}" \
      --within "${POP_FILE_FORMATTED}" \
      --fst \
      --chr-set 34 no-xy \
      --out "${FST_OUTPUT_DIR}/admixture_final_ld_pruned_filtered"

# -------------------------------------------------------------------------------------------------------------------------

## Step 5: Sort the Fst output
echo "Sorting Fst output by value..."
awk 'NR>1 {print $0 | "sort -k5,5nr"}' "${FST_OUTPUT_DIR}/admixture_final_ld_pruned_filtered.fst" > "${FST_OUTPUT_DIR}/admixture_final_ld_pruned_filtered.fst.sorted"

echo "--- Fst Analysis Workflow Completed ---"
date +"%Y-%m-%d %H:%M:%S"

#-------------------------------------------------------------------------------------------------------------------------
## Step 6: Export filtered PLINK data to a numeric format suitable for RDA
echo "Exporting PLINK data to a numeric format..."

# This command recodes the data into a plain text file with one row per individual
# and one column per SNP. Genotypes are represented by 0, 1, or 2 (counts of the
# non-reference allele). The `--out-chr-geno-pos` and `--out-geno` arguments
# are essential for creating a matrix-like output.
plink --bfile "${PLINK_FILTERED_PREFIX}" \
      --recode A \
      --out "${PLINK_FILTERED_PREFIX}_for_RDA" \
      --chr-set 34 no-xy

# The output from the above command will be a file named
# "admixture_final_ld_pruned_filtered_for_RDA.raw".

echo "Genetic data exported to '${PLINK_FILTERED_PREFIX}_for_RDA.raw'."
