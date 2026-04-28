#!/bin/bash
#SBATCH --job-name=windowed_pi_fst_analysis
#SBATCH --output=windowed_pi_fst_analysis_%j.out
#SBATCH --error=windowed_pi_fst_analysis_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=long

set -e

echo "--- Starting Windowed pi/Fst Analysis Workflow ---"
date +"%Y-%m-%d %H:%M:%S"

# --- Configuration ---
CONDA_ENV_NAME="bcftools"
PLINK_FILTERED_PREFIX="/home/aescudero/wgg/RDA/admixture_final_ld_pruned_filtered"
POP_FILE_RAW="/home/aescudero/wgg/RDA/population_list_pop1_2.txt"
UPDATE_IDS_FILE="/home/aescudero/wgg/admixture_data/update_ids.txt"
VCF_OUTPUT_BCF_DIRTY="${PLINK_FILTERED_PREFIX}.bcf" # Temporary BCF with dirty IDs
VCF_OUTPUT_BCF_CLEAN="${PLINK_FILTERED_PREFIX}_clean.bcf" # Final BCF with clean IDs
RESULTS_DIR="/home/aescudero/wgg/fst_results/windowed_stats"

WINDOW_SIZE=1000
CHR_SET=34

# --- Activate Conda environment ---
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source "/home/aescudero/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}" || { echo "Error: Failed to activate Conda environment '${CONDA_ENV_NAME}'. Exiting." >&2; exit 1; }
echo "Conda environment '${CONDA_ENV_NAME}' activated."

# --- Create Output Directories ---
echo "Creating output directories..."
mkdir -p "${RESULTS_DIR}"

# --- 1. CONVERT PLINK to VCF/BCF (Outputting VCF with dirty IDs) ---
echo "1. Converting PLINK to BCF format..."
plink --bfile "${PLINK_FILTERED_PREFIX}" \
      --recode vcf \
      --out "${PLINK_FILTERED_PREFIX}" \
      --chr-set ${CHR_SET} no-xy

bcftools view -O b -o "${VCF_OUTPUT_BCF_DIRTY}" "${PLINK_FILTERED_PREFIX}.vcf"

# --- 2. CRITICAL STEP: CLEAN SAMPLE IDs IN THE BCF HEADER ---
echo "2.1 Reheadering BCF to clean sample IDs (matching 184 samples)..."
NEW_SAMPLE_LIST="${RESULTS_DIR}/new_sample_names_184.txt"
DIRTY_ORDER_LIST="${RESULTS_DIR}/dirty_vcf_order.txt"
ID_MAP="${RESULTS_DIR}/id_map.txt"

# A. Extract the 184 dirty IDs currently in the BCF, in their precise order.
echo "A. Extracting 184 sample IDs from the BCF header..."
bcftools query -l "${VCF_OUTPUT_BCF_DIRTY}" > "${DIRTY_ORDER_LIST}"
DIRTY_COUNT=$(wc -l < "${DIRTY_ORDER_LIST}")
echo "BCF contains ${DIRTY_COUNT} samples."

# B. Create the map file (Dirty_IID \t Clean_IID) from update_ids.txt
# Assumes Col 2 (Dirty IID) maps to Col 4 (Clean IID).
echo "B. Creating map file from update_ids.txt (Dirty_IID from Col 2 -> Clean_IID from Col 4)..."
awk '{print $2 "\t" $4}' "${UPDATE_IDS_FILE}" > "${ID_MAP}"

# C. Generate list of clean IDs, correcting for mangled names using sequential lookup.
echo "C. Generating list of clean IDs in the correct BCF order (using sequential lookup for mangled IDs)..."
> "${NEW_SAMPLE_LIST}" # Clear the output list

# Process the dirty IDs one by one and look up the clean ID from the map file
while read DIRTY_VCF_ID; do
    
    CLEAN_ID=""
    
    # 1. Try a direct lookup first
    CLEAN_ID=$(grep -w "${DIRTY_VCF_ID}" "${ID_MAP}" | awk '{print $2}')
    
    if [[ -z "${CLEAN_ID}" ]]; then
        
        # 2. Sequential attempt for mangled IDs (e.g., C12ME24_1_C12ME24_1 -> C12ME24_1)
        
        # Method A: Strip the shortest suffix matching '_*' (e.g., C12ME24_1_C12ME24_1 -> C12ME24_1)
        BASE_DIRTY_ID_A="${DIRTY_VCF_ID%_*}"
        CLEAN_ID=$(grep -w "${BASE_DIRTY_ID_A}" "${ID_MAP}" | awk '{print $2}')
        
        # If still not found, try Method B: Force split on the first two underscore fields.
        if [[ -z "${CLEAN_ID}" ]]; then
            # This handles the case where the ID has two parts separated by an underscore (ID_P1_ID_P1)
            BASE_DIRTY_ID_B=$(echo "${DIRTY_VCF_ID}" | awk -F'_' '{print $1"_"$2}')
            CLEAN_ID=$(grep -w "${BASE_DIRTY_ID_B}" "${ID_MAP}" | awk '{print $2}')
        fi
    fi

    # Final result check
    if [[ -n "${CLEAN_ID}" ]]; then
        echo "${CLEAN_ID}" >> "${NEW_SAMPLE_LIST}"
    else
        echo "Error: Could not find clean ID for dirty ID: ${DIRTY_VCF_ID}. Exiting." >&2
        exit 1
    fi
    
done < "${DIRTY_ORDER_LIST}"

# D. Final check and Reheader the BCF
NEW_SAMPLE_COUNT=$(wc -l < "${NEW_SAMPLE_LIST}")
if [ "${NEW_SAMPLE_COUNT}" -ne "${DIRTY_COUNT}" ]; then
    echo "CRITICAL ERROR: Mismatch! New sample list count is ${NEW_SAMPLE_COUNT}, but BCF has ${DIRTY_COUNT}. Exiting." >&2
    exit 1
fi
echo "New sample list of ${NEW_SAMPLE_COUNT} samples generated successfully."

bcftools reheader -s "${NEW_SAMPLE_LIST}" -o "${VCF_OUTPUT_BCF_CLEAN}" "${VCF_OUTPUT_BCF_DIRTY}"
echo "BCF reheadered successfully to ${VCF_OUTPUT_BCF_CLEAN}."

# Set the main BCF variable to the newly cleaned file for subsequent steps
VCF_OUTPUT_BCF="${VCF_OUTPUT_BCF_CLEAN}"

# --- 2.2 GENERATE VCFtools Population Lists (using the clean IDs that now match the BCF) ---
echo "2.2 Generating VCFtools population files from the clean population_list.txt."
tail -n +2 "${POP_FILE_RAW}" | awk '$2 == "POP1" {print $1}' > "${RESULTS_DIR}/pop1_samples_cleaned.txt"
tail -n +2 "${POP_FILE_RAW}" | awk '$2 == "POP2" {print $1}' > "${RESULTS_DIR}/pop2_samples_cleaned.txt"

POP1_LIST="${RESULTS_DIR}/pop1_samples_cleaned.txt"
POP2_LIST="${RESULTS_DIR}/pop2_samples_cleaned.txt"

# --- 3. CALCULATE Fst (Weir & Cockerham) in Windows ---
echo "3. Calculating Fst in ${WINDOW_SIZE} bp windows..."
vcftools --bcf "${VCF_OUTPUT_BCF}" \
         --weir-fst-pop "${POP1_LIST}" \
         --weir-fst-pop "${POP2_LIST}" \
         --fst-window-size ${WINDOW_SIZE} \
         --fst-window-step ${WINDOW_SIZE} \
         --out "${RESULTS_DIR}/windowed_fst_${WINDOW_SIZE}bp"

# --- 4. CALCULATE PI (Nucleotide Diversity) in Windows ---
echo "4. Calculating Pi in ${WINDOW_SIZE} bp windows for each population..."

# POP1 Pi - Removed the unsupported --window-step flag.
vcftools --bcf "${VCF_OUTPUT_BCF}" \
         --keep "${POP1_LIST}" \
         --window-pi ${WINDOW_SIZE} \
         --out "${RESULTS_DIR}/windowed_pi_pop1_${WINDOW_SIZE}bp"

# POP2 Pi - Removed the unsupported --window-step flag.
vcftools --bcf "${VCF_OUTPUT_BCF}" \
         --keep "${POP2_LIST}" \
         --window-pi ${WINDOW_SIZE} \
         --out "${RESULTS_DIR}/windowed_pi_pop2_${WINDOW_SIZE}bp"

echo "--- Windowed pi/Fst Analysis Workflow Completed ---"
date +"%Y-%m-%d %H:%M:%S"
