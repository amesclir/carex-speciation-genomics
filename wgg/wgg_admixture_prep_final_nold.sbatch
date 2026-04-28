#!/bin/bash
#SBATCH --job-name=admixture_prep
#SBATCH --output=admixture_prep_%j.out
#SBATCH --error=admixture_prep_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G # Adjust based on your dataset size
#SBATCH --time=06:00:00 # HH:MM:SS, Increased time just in case. Adjust as needed for larger datasets
#SBATCH --partition=long # Or your appropriate partition

# Set strict error checking: exit immediately if any command fails
set -e

echo "--- Starting Admixture Data Preparation Workflow ---"
date +"%Y-%m-%d %H:%M:%S" # Consistent date format

# --- Configuration ---
CONDA_ENV_NAME="bcftools"
BAM_DIR="/home/aescudero/wgg/bams"
ADMIXTURE_OUT_DIR="/home/aescudero/wgg/admixture_data"
INPUT_FILTERED_VCF="${BAM_DIR}/all_samples_calls_filtered.recode.vcf.gz"
EXCLUDE_SAMPLES_FILE="${ADMIXTURE_OUT_DIR}/samples_to_exclude_for_admixture.txt"
PLINK_INITIAL_PREFIX="${ADMIXTURE_OUT_DIR}/admixture_all_samples_mac20_no_outgroups"
PLINK_SORTED_PREFIX="${ADMIXTURE_OUT_DIR}/admixture_all_samples_mac20_no_outgroups_sorted"
PLINK_LD_PRUNED_PREFIX="${ADMIXTURE_OUT_DIR}/admixture_final_ld_pruned"
TEMP_CLEANED_VCF="${ADMIXTURE_OUT_DIR}/all_samples_calls_filtered_cleaned_chroms.vcf.gz" # New temporary file

# IMPORTANT: Determine the maximum chromosome number in your data.
# Based on your error, it goes at least up to 34. Adjust this value if your actual
# maximum chromosome number is different (e.g., if you have 'scaffold_100', set this to 100).
MAX_CHROM_NUM=34 

echo "Input VCF: ${INPUT_FILTERED_VCF}"
echo "Output Directory: ${ADMIXTURE_OUT_DIR}"
echo "Temporary Cleaned VCF: ${TEMP_CLEANED_VCF}"
echo "Maximum Chromosome Number: ${MAX_CHROM_NUM}"
echo "--------------------------------------------------"

# --- Load Conda environment ---
echo "Loading Conda environment: ${CONDA_ENV_NAME}"
source "/home/aescudero/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}" || { echo "Error: Failed to activate Conda environment '${CONDA_ENV_NAME}'. Exiting." >&2; exit 1; }
echo "Conda environment '${CONDA_ENV_NAME}' activated."

# Verify tools are in PATH
command -v plink || { echo "Error: PLINK not found in PATH. Exiting." >&2; exit 1; }
command -v bcftools || { echo "Error: BCFtools not found in PATH. Exiting." >&2; exit 1; }
command -v awk || { echo "Error: AWK not found in PATH. Exiting." >&2; exit 1; } # Check for awk
echo "PLINK, BCFtools, and AWK commands are now accessible."

# --- Create Output Directory ---
echo "Creating output directory: ${ADMIXTURE_OUT_DIR}"
mkdir -p "${ADMIXTURE_OUT_DIR}"

# --- Create Samples to Exclude File (if not already present or needs regeneration) ---
echo "Creating list of samples to exclude for ADMIXTURE..."
if [ ! -f "${EXCLUDE_SAMPLES_FILE}" ] || [ "$(wc -l < "${EXCLUDE_SAMPLES_FILE}")" -eq 0 ]; then
  echo "# Add sample IDs to exclude here, one per line." > "${EXCLUDE_SAMPLES_FILE}"
  echo "# e.g., SampleID_1" >> "${EXCLUDE_SAMPLES_FILE}"
  echo "# e.g., SampleID_2" >> "${EXCLUDE_SAMPLES_FILE}"
  # Add actual sample IDs here if you want them automatically inserted:
  # echo "YourOutgroupSample1" >> "${EXCLUDE_SAMPLES_FILE}"
  # echo "YourOutgroupSample2" >> "${EXCLUDE_SAMPLES_FILE}"
  echo "Samples to exclude (re)generated: ${EXCLUDE_SAMPLES_FILE}"
else
  echo "Samples to exclude file already exists: ${EXCLUDE_SAMPLES_FILE}"
fi

# --- Step 1: Filter VCF, Clean Chromosome Names, and save to a temporary file ---
echo "Filtering VCF, cleaning chromosome names (removing 'scaffold_'), and saving to a temporary VCF file: ${TEMP_CLEANED_VCF}..."
bcftools view --threads "${SLURM_CPUS_PER_TASK}" -v snps -m2 -M2 -i 'MAC>=20' "${INPUT_FILTERED_VCF}" | \
awk 'BEGIN {OFS="\t"} /^#/ {print; next} {if(substr($1,1,9)=="scaffold_") $1=substr($1,10); print}' | \
bcftools view -Oz -o "${TEMP_CLEANED_VCF}" || { echo "Error: Creation of temporary cleaned VCF failed. Exiting." >&2; exit 1; }

echo "Temporary cleaned VCF created: ${TEMP_CLEANED_VCF}"

# --- Step 2: Convert cleaned VCF to PLINK binary format and exclude samples ---
echo "Converting cleaned VCF to PLINK binary format and excluding samples..."
plink --vcf "${TEMP_CLEANED_VCF}" \
      --remove "${EXCLUDE_SAMPLES_FILE}" \
      --double-id \
      --make-bed \
      --out "${PLINK_INITIAL_PREFIX}" \
      --chr-set "${MAX_CHROM_NUM}" \
      --allow-extra-chr 0 || { echo "Error: PLINK conversion from cleaned VCF failed. Exiting." >&2; exit 1; }

echo "Initial PLINK dataset created: ${PLINK_INITIAL_PREFIX}.bed/.bim/.fam"
echo "Number of individuals in initial PLINK dataset:"
wc -l "${PLINK_INITIAL_PREFIX}.fam"

# --- Step 3: Sort PLINK data (Crucial for LD pruning) ---
echo "Sorting PLINK dataset (${PLINK_INITIAL_PREFIX}) to ensure proper order for LD pruning..."
plink --bfile "${PLINK_INITIAL_PREFIX}" \
      --make-bed \
      --out "${PLINK_SORTED_PREFIX}" \
      --chr-set "${MAX_CHROM_NUM}" \
      --allow-extra-chr 0 || { echo "Error: PLINK sorting failed. Exiting." >&2; exit 1; }

echo "PLINK dataset sorted: ${PLINK_SORTED_PREFIX}.bed/.bim/.fam"

# --- Step 4: Perform Linkage Disequilibrium (LD) pruning ---
echo "Performing Linkage Disequilibrium (LD) pruning on sorted data..."
# --indep-pairwise <window size> <step size> <r^2 threshold>
plink --bfile "${PLINK_SORTED_PREFIX}" \
#      --indep-pairwise 50 5 0.5 \
      --out "${PLINK_LD_PRUNED_PREFIX}" \
      --chr-set "${MAX_CHROM_NUM}" || { echo "Error: PLINK LD pruning failed. Exiting." >&2; exit 1; }

echo "LD pruning complete. Pruned SNPs listed in: ${PLINK_LD_PRUNED_PREFIX}.prune.in and .prune.out"

# --- Step 5: Filter the sorted PLINK dataset to keep only independent SNPs ---
echo "Filtering sorted PLINK dataset with LD-pruned variants..."
plink --bfile "${PLINK_SORTED_PREFIX}" \
      --extract "${PLINK_LD_PRUNED_PREFIX}.prune.in" \
      --make-bed \
      --out "${PLINK_LD_PRUNED_PREFIX}" \
      --chr-set "${MAX_CHROM_NUM}" || { echo "Error: Filtering with pruned variants failed. Exiting." >&2; exit 1; }

echo "Final LD-pruned PLINK dataset created: ${PLINK_LD_PRUNED_PREFIX}.bed/.bim/.fam"

# --- Clean up temporary VCF file ---
echo "Cleaning up temporary VCF file: ${TEMP_CLEANED_VCF}"
rm "${TEMP_CLEANED_VCF}" || echo "Warning: Could not remove temporary file ${TEMP_CLEANED_VCF}"

echo "--- Admixture Data Preparation Workflow Completed Successfully! ---"
date +"%Y-%m-%d %H:%M:%S"
