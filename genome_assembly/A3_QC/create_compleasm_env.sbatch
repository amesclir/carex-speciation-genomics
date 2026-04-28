#!/bin/bash
#SBATCH --job-name=Create_Compleasm_Env
#SBATCH --output=create_compleasm_env_output.txt
#SBATCH --error=create_compleasm_env_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G         # Sufficient memory for environment creation and install
#SBATCH --time=00:30:00  # Ample time for full creation and installation

# Define the environment name
ENV_NAME="compleasm"

echo "--- Starting Compleasm Environment Creation and Installation Script ---"
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo ""

# --- Source Conda ---
echo "Sourcing Conda initialization script..."
source ~/anaconda3/etc/profile.d/conda.sh

# --- Remove Existing Environment (if any) ---
echo "Attempting to remove existing '${ENV_NAME}' Conda environment..."
conda env remove --name "${ENV_NAME}" --yes

if [ $? -eq 0 ]; then
    echo "Existing '${ENV_NAME}' environment successfully removed (or did not exist)."
else
    echo "WARNING: Failed to remove existing '${ENV_NAME}' environment. Check error file."
    echo "Proceeding with environment creation, but a previous issue might persist."
fi
echo ""

# --- Create New Environment ---
echo "Creating a new '${ENV_NAME}' Conda environment..."
conda create --name "${ENV_NAME}" --yes

if [ $? -eq 0 ]; then
    echo "New '${ENV_NAME}' environment created successfully."
else
    echo "ERROR: Failed to create new '${ENV_NAME}' environment. Aborting installation."
    exit 1
fi
echo ""

# --- Activate New Environment ---
echo "Activating the newly created '${ENV_NAME}' environment..."
conda activate "${ENV_NAME}"

if [ $? -eq 0 ]; then
    echo "New '${ENV_NAME}' environment activated successfully."
    echo "Current Conda environment: $CONDA_DEFAULT_ENV"
else
    echo "ERROR: Failed to activate new '${ENV_NAME}' environment. Aborting installation."
    exit 1
fi
echo ""

# --- Install compleasm ---
echo "Installing compleasm into '${ENV_NAME}' environment..."
conda install --yes -c bioconda compleasm

if [ $? -eq 0 ]; then
    echo "Compleasm installed successfully."
else
    echo "ERROR: Conda installation failed for compleasm. Check error file for details."
    echo "This could be due to network issues, dependency conflicts, or insufficient disk space."
    exit 1
fi
echo ""

# --- Verify Installation ---
echo "Verifying compleasm installation:"
if which compleasm >/dev/null 2>&1; then
    echo "Compleasm FOUND at $(which compleasm)"
    echo "Attempting to get compleasm version (output will vary by compleasm version):"
    compleasm --version 2>&1 || echo "Could not retrieve compleasm version."
else
    echo "ERROR: Compleasm NOT FOUND after installation. Something went wrong."
fi

echo ""
echo "--- Compleasm Environment Creation and Installation Complete ---"
echo "Job finished on: $(date)"
