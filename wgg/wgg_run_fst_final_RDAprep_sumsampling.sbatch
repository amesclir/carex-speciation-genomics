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
## Step 5: Randomly Subsample SNPs for RDA
echo "Subsampling SNPs for RDA..."

# Set the number of SNPs you want to include in the RDA
# A few thousand is a good starting point for 184 individuals.
NUM_SNPS_TO_KEEP=5000

# Create a list of randomly selected SNPs to keep
# We use 'awk' to skip the header and 'shuf' to shuffle the lines and select the top N.
awk 'NR>1 {print $2}' "${PLINK_FILTERED_PREFIX}.bim" | shuf -n ${NUM_SNPS_TO_KEEP} > "${PLINK_FILTERED_PREFIX}_snps_to_keep.txt"

# Create a new filtered PLINK file set with only the selected SNPs
plink --bfile "${PLINK_FILTERED_PREFIX}" \
      --extract "${PLINK_FILTERED_PREFIX}_snps_to_keep.txt" \
      --make-bed \
      --out "${PLINK_FILTERED_PREFIX}_for_RDA_subset" \
      --chr-set 34 no-xy

#-------------------------------------------------------------------------------------------------------------------------
## Step 6 (new): Export the smaller PLINK data to a numeric format for R
echo "Exporting PLINK data to a numeric format..."

plink --bfile "${PLINK_FILTERED_PREFIX}_for_RDA_subset" \
      --recode A \
      --out "${PLINK_FILTERED_PREFIX}_for_RDA_subset" \
      --chr-set 34 no-xy

echo "Genetic data for RDA exported to '${PLINK_FILTERED_PREFIX}_for_RDA_subset.raw'."
