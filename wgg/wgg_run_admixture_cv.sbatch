#!/bin/bash
#SBATCH --job-name=admixture_cv
#SBATCH --output=admixture_cv_%j.out
#SBATCH --error=admixture_cv_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=long

set -e

echo "--- Starting ADMIXTURE Cross-Validation Workflow ---"
date +"%Y-%m-%d %H:%M:%S"

# --- Configuration ---
CONDA_ENV_NAME="admixture"
# Use the file that was successfully created by the prep script.
PLINK_FILE_PREFIX="/home/aescudero/wgg/admixture_data/admixture_final_850k_pruned"
OUTPUT_DIR="/home/aescudero/wgg/admixture_data/admixture_results_cv"

# --- Activate Conda environment ---
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source "/home/aescudero/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}" || { echo "Error: Failed to activate Conda environment '${CONDA_ENV_NAME}'. Exiting." >&2; exit 1; }
echo "Conda environment '${CONDA_ENV_NAME}' activated."

# --- Create Output Directory and Change to It ---
echo "Creating output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
echo "Changing to output directory: ${OUTPUT_DIR}"
cd "${OUTPUT_DIR}" || { echo "Error: Could not change to output directory. Exiting." >&2; exit 1; }

# --- Run ADMIXTURE with Cross-Validation for K=1 to K=8 ---
for K in {9..15}; do
  echo "Running ADMIXTURE for K=${K} with 5-fold cross-validation..."
  admixture --cv=5 "${PLINK_FILE_PREFIX}.bed" "${K}"
done

echo "--- ADMIXTURE Cross-Validation Workflow Completed ---"
date +"%Y-%m-%d %H:%M:%S"
