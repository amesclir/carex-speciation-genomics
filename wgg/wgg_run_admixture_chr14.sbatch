#!/bin/bash
#SBATCH --job-name=admixture_chr14_inv
#SBATCH --output=admixture_chr14_inv_K2_%j.out
#SBATCH --error=admixture_chr14_inv_K2_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00

# Set strict error checking: exit immediately if any command fails
set -e

echo "--- Starting ADMIXTURE Analysis for Inversion on Chr14 ---"
date +"%Y-%m-%d %H:%M:%S"

# --- Configuration ---
CONDA_ENV_NAME="admixture"
# Path to the LD-pruned PLINK dataset for the chr14 inversion region
PLINK_INPUT_PATH="/home/aescudero/wgg/admixture_data/admixture_chr14_inversion_final_ld_pruned.bed"
ADMIXTURE_OUTPUT_DIR="/home/aescudero/wgg/admixture_data/admixture_results"

# --- Activate Conda environment ---
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source "/home/aescudero/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}" || { echo "Error: Failed to activate Conda environment '${CONDA_ENV_NAME}'. Exiting." >&2; exit 1; }
echo "Conda environment '${CONDA_ENV_NAME}' activated."

# --- Create Output Directory and Change to It ---
echo "Creating output directory: ${ADMIXTURE_OUTPUT_DIR}"
mkdir -p "${ADMIXTURE_OUTPUT_DIR}"
echo "Changing to output directory: ${ADMIXTURE_OUTPUT_DIR}"
cd "${ADMIXTURE_OUTPUT_DIR}" || { echo "Error: Could not change to output directory. Exiting." >&2; exit 1; }

# --- Run ADMIXTURE for the specified K values ---
for K in {1..8}; do
    echo "Running ADMIXTURE for K=${K}..."

    # The ADMIXTURE command will produce a .K.Q file and a .K.P file
    admixture --cv=10 "${PLINK_INPUT_PATH}" "${K}"

    # Check if ADMIXTURE run was successful
    if [ $? -eq 0 ]; then
        echo "ADMIXTURE for K=${K} finished successfully."
    else
        echo "Error: ADMIXTURE for K=${K} failed. Check log files." >&2
        exit 1
    fi
done  # <--- THIS WAS THE MISSING 'done'

echo "--- ADMIXTURE Analysis Workflow Completed for K=9 to K=15 ---"
date +"%Y-%m-%d %H:%M:%S"
