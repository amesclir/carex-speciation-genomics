#!/bin/bash
#SBATCH --job-name=admixture_run_allK
#SBATCH --output=admixture_run_allK_%j.out
#SBATCH --error=admixture_run_allK_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # Requesting 1 CPU per job as ADMIXTURE will run single-threaded.
#                         # This may be more efficient for the scheduler to manage multiple single-core jobs.
#SBATCH --mem=16G
#SBATCH --time=24:00:00 # Increased time since it will be single-threaded for many K values.
#SBATCH --partition=long

# Set strict error checking: exit immediately if any command fails
set -e

echo "--- Starting ADMIXTURE Analysis ---"
date +"%Y-%m-%d %H:%M:%S"

# --- Configuration ---
CONDA_ENV_NAME="admixture"
PLINK_INPUT_PATH="/home/aescudero/wgg/admixture_data/admixture_final_850k_pruned.bed"
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

# --- Run ADMIXTURE for a range of K values ---
K_START=1
K_END=10

for K in $(seq "${K_START}" "${K_END}"); do
    echo "Running ADMIXTURE for K=${K}..."
    
    # ADMIXTURE command without parallel flags. It is known to work now.
    admixture "${PLINK_INPUT_PATH}" "${K}"

    # Check if ADMIXTURE run for this K was successful
    if [ $? -eq 0 ]; then
        echo "ADMIXTURE for K=${K} finished successfully."
    else
        echo "Error: ADMIXTURE for K=${K} failed. Check log files." >&2
        # It's good to exit the script if one K fails.
        exit 1
    fi
done

echo "--- ADMIXTURE Analysis Workflow Completed ---"
date +"%Y-%m-%d %H:%M:%S"
