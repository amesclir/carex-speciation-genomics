#!/bin/bash
#SBATCH --job-name=admixture_inv_chr18
#SBATCH --output=admixture_inv_chr18_%j.out
#SBATCH --error=admixture_inv_chr18_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=long

# Set strict error checking: exit immediately if any command fails
set -e

echo "--- Starting Admixture Inversion Data Preparation Workflow for Chromosome 18 (V2: LD Pruning Fixed) ---"
date +"%Y-%m-%d %H:%M:%S"

# --- Configuration for your specific inversion ---
# Inversion on Chromosome 18: 1681715 - 2543617
TARGET_CHR="18"
START_POS="1681715"
END_POS="2543617"
# The BCFtools filter needs the full scaffold name
BCFTOOLS_FILTER_CHR="scaffold_18"

# --- General Configuration ---
CONDA_ENV_NAME="bcftools"
BAM_DIR="/home/aescudero/wgg/bams"
ADMIXTURE_OUT_DIR="/home/aescudero/wgg/admixture_data"
INPUT_FILTERED_VCF="${BAM_DIR}/all_samples_calls_filtered.recode.vcf.gz"
EXCLUDE_SAMPLES_FILE="${ADMIXTURE_OUT_DIR}/samples_to_exclude_for_admixture.txt"

# --- PLINK Prefixes ---
PLINK_INITIAL_PREFIX="${ADMIXTURE_OUT_DIR}/admixture_chr${TARGET_CHR}_inversion_initial"
PLINK_SORTED_PREFIX="${PLINK_INITIAL_PREFIX}_sorted"
# CRITICAL: New prefix for the dataset with fixed CHR:POS IDs (needed for successful LD pruning/extraction)
PLINK_FIXED_ID_PREFIX="${PLINK_INITIAL_PREFIX}_sorted_fixed_id"
# New prefix for the .prune.in file only (temporary file)
PLINK_LD_TEMP_PREFIX="${PLINK_INITIAL_PREFIX}_ld_temp"
# Final prefix for the LD-pruned dataset
PLINK_FINAL_PREFIX="${ADMIXTURE_OUT_DIR}/admixture_chr${TARGET_CHR}_inversion_final_ld_pruned"
TEMP_VCF="${PLINK_INITIAL_PREFIX}_temp.vcf.gz" # Temporary VCF file

# IMPORTANT: Determine the maximum chromosome number in your data.
MAX_CHROM_NUM=34

echo "Processing Inversion on Chromosome ${TARGET_CHR} from ${START_POS} to ${END_POS}"
echo "Final PLINK Prefix: ${PLINK_FINAL_PREFIX}"
echo "--------------------------------------------------"

# --- Load Conda environment ---
echo "Loading Conda environment: ${CONDA_ENV_NAME}"
source "/home/aescudero/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}" || { echo "Error: Failed to activate Conda environment '${CONDA_ENV_NAME}'. Exiting." >&2; exit 1; }
echo "Conda environment '${CONDA_ENV_NAME}' activated."

# Verify tools are in PATH
command -v plink || { echo "Error: PLINK not found in PATH. Exiting." >&2; exit 1; }
command -v bcftools || { echo "Error: BCFtools not found in PATH. Exiting." >&2; exit 1; }
command -v awk || { echo "Error: AWK not found in PATH. Exiting." >&2; exit 1; }
echo "PLINK, BCFtools, and AWK commands are now accessible."

# --- Create Output Directory ---
mkdir -p "${ADMIXTURE_OUT_DIR}"

# ----------------------------------------------------------------------
# --- Step 1: Filter VCF by region, filter, clean chrom names, and save to temp file ---
# ----------------------------------------------------------------------
echo "Step 1: Filtering VCF for specified region and saving to temporary VCF file: ${TEMP_VCF}..."
bcftools view --threads "${SLURM_CPUS_PER_TASK}" -v snps -m2 -M2 -i 'MAC>=20' \
-r "${BCFTOOLS_FILTER_CHR}:${START_POS}-${END_POS}" "${INPUT_FILTERED_VCF}" | \
awk 'BEGIN {OFS="\t"} /^#/ {print; next} {if(substr($1,1,9)=="scaffold_") $1=substr($1,10); print}' | \
bcftools view -Oz -o "${TEMP_VCF}" || { echo "Error: Step 1 failed (Cleaning VCF). Exiting." >&2; exit 1; }

echo "Temporary cleaned VCF created."

# ----------------------------------------------------------------------
# --- Step 2: Convert cleaned VCF to PLINK binary format and exclude samples ---
# ----------------------------------------------------------------------
echo "Step 2: Converting cleaned VCF to PLINK binary format and excluding samples..."
plink --vcf "${TEMP_VCF}" \
     --remove "${EXCLUDE_SAMPLES_FILE}" \
     --double-id \
     --make-bed \
     --out "${PLINK_INITIAL_PREFIX}" \
     --chr-set "${MAX_CHROM_NUM}" \
     --allow-extra-chr 0 || { echo "Error: Step 2 failed (PLINK conversion). Exiting." >&2; exit 1; }

echo "Initial PLINK dataset for inversion created: ${PLINK_INITIAL_PREFIX}.bed/.bim/.fam"

# ----------------------------------------------------------------------
# --- Step 3: Sort PLINK data (Crucial for LD pruning) ---
# ----------------------------------------------------------------------
echo "Step 3: Sorting PLINK dataset (${PLINK_INITIAL_PREFIX}) to ensure proper order for LD pruning..."
plink --bfile "${PLINK_INITIAL_PREFIX}" \
     --make-bed \
     --out "${PLINK_SORTED_PREFIX}" \
     --chr-set "${MAX_CHROM_NUM}" \
     --allow-extra-chr 0 || { echo "Error: Step 3 failed (PLINK sorting). Exiting." >&2; exit 1; }

echo "PLINK dataset sorted: ${PLINK_SORTED_PREFIX}.bed/.bim/.fam"

# ----------------------------------------------------------------------
# --- Step 4: Fix SNP IDs in the sorted dataset (CRITICAL FIX) ---
# We must fix '.' as the SNP ID to CHR:POS.
# ----------------------------------------------------------------------
echo "Step 4: Fixing SNP IDs in the sorted dataset by replacing '.' with CHR:POS..."

# 1. Create a new BIM file with CHR:POS as the ID (column 2)
awk 'BEGIN{OFS="\t"} {
    original_id=$2;
    new_id=$1":"$4;
    # Replace the ID column with CHR:POS
    print $1, new_id, $3, $4, $5, $6
}' "${PLINK_SORTED_PREFIX}.bim" > "${PLINK_FIXED_ID_PREFIX}.bim"

# 2. Copy the .bed and .fam files from the sorted prefix
cp "${PLINK_SORTED_PREFIX}.bed" "${PLINK_FIXED_ID_PREFIX}.bed"
cp "${PLINK_SORTED_PREFIX}.fam" "${PLINK_FIXED_ID_PREFIX}.fam"

echo "New PLINK dataset created with fixed IDs: ${PLINK_FIXED_ID_PREFIX}.bed/.bim/.fam"

# ----------------------------------------------------------------------
# --- Step 5: Perform Linkage Disequilibrium (LD) pruning ---
# This step now uses the fixed-ID dataset and will generate a .prune.in file with VALID CHR:POS IDs.
# ----------------------------------------------------------------------
echo "Step 5: Performing Linkage Disequilibrium (LD) pruning on fixed-ID data..."
plink --bfile "${PLINK_FIXED_ID_PREFIX}" \
     --indep-pairwise 50 5 0.5 \
     --out "${PLINK_LD_TEMP_PREFIX}" \
     --chr-set "${MAX_CHROM_NUM}" || { echo "Error: Step 5 failed (PLINK LD pruning). Exiting." >&2; exit 1; }

echo "LD pruning complete. Pruned SNPs listed in: ${PLINK_LD_TEMP_PREFIX}.prune.in (using CHR:POS IDs)"

# ----------------------------------------------------------------------
# --- Step 6: Filter the fixed-ID dataset to create the FINAL SNP set ---
# This step now correctly uses the valid CHR:POS IDs from the .prune.in file.
# ----------------------------------------------------------------------
echo "Step 6: Filtering fixed-ID PLINK dataset with LD-pruned variants to create final dataset: ${PLINK_FINAL_PREFIX}..."
plink --bfile "${PLINK_FIXED_ID_PREFIX}" \
     --extract "${PLINK_LD_TEMP_PREFIX}.prune.in" \
     --make-bed \
     --out "${PLINK_FINAL_PREFIX}" \
     --chr-set "${MAX_CHROM_NUM}" || { echo "Error: Step 6 failed (Filtering with pruned variants). Exiting." >&2; exit 1; }

echo "Final LD-pruned PLINK dataset for inversion created: ${PLINK_FINAL_PREFIX}.bed/.bim/.fam"
echo "Final SNP count verification:"
wc -l "${PLINK_FINAL_PREFIX}.bim"

# --- Clean up temporary VCF file ---
echo "Cleaning up temporary VCF file..."
rm "${TEMP_VCF}" || echo "Warning: Could not remove temporary file: ${TEMP_VCF}"

echo "--- Admixture Inversion Data Preparation Workflow Completed Successfully! ---"
date +"%Y-%m-%d %H:%M:%S"
