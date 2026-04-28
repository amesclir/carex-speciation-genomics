#!/bin/bash
#SBATCH --job-name=Recreate_BTK_Install_Blobtools
#SBATCH --output=recreate_btk_env_and_install_blobtools_output.txt
#SBATCH --error=recreate_btk_env_and_install_blobtools_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G         # More memory for environment creation and installation
#SBATCH --time=00:45:00  # Ample time for full recreation and installation

# Define the environment name
ENV_NAME="btk"

echo "--- Starting Blobtools Environment Recreation and Installation Script ---"
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo ""

# --- Source Conda ---
echo "Sourcing Conda initialization script..."
source ~/anaconda3/etc/profile.d/conda.sh

# --- Remove Existing Environment ---
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
# It's good practice to specify channels directly when creating,
# especially if you know the package is in bioconda/conda-forge.
conda create --name "${ENV_NAME}" --channel bioconda --channel conda-forge --yes

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

# --- Install blobtools (the correct Conda package name) ---
echo "Installing blobtools into '${ENV_NAME}' environment..."
conda install --yes -c bioconda blobtools

if [ $? -eq 0 ]; then
    echo "Blobtools installed successfully."
else
    echo "ERROR: Conda installation failed for blobtools. Check error file for details."
    echo "This could be due to network issues, dependency conflicts, or insufficient disk space."
    exit 1
fi
echo ""

# --- Verify Installation ---
echo "Verifying blobtools installation:"
if which blobtools >/dev/null 2>&1; then
    BLOBTOOLS_PATH=$(which blobtools)
    echo "SUCCESS: 'blobtools' FOUND at: $BLOBTOOLS_PATH"
    echo "Attempting to get 'blobtools' version:"
    blobtools --version 2>&1 || echo "Could not retrieve blobtools version."
else
    echo "ERROR: 'blobtools' command NOT FOUND after installation. Something went wrong."
    echo "Please ensure 'blobtools' was installed correctly."
fi

echo ""
echo "--- Blobtools Environment Recreation and Installation Complete ---"
echo "Job finished on: $(date)"
