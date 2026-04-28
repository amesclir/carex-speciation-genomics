#!/bin/bash
#SBATCH --job-name=Install_BlobToolKit_v4
#SBATCH --output=install_blobtoolkit_v4_output.txt
#SBATCH --error=install_blobtoolkit_v4_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G         # Adjust memory if needed, 8G should be sufficient
#SBATCH --time=01:00:00  # Ample time for environment creation and pip installation

# Define the environment name
ENV_NAME="btk"

echo "--- Starting BlobToolKit v4 Environment Setup and Installation Script ---"
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

# --- Create New Environment with Python 3.9 and pip ---
echo "Creating a new '${ENV_NAME}' Conda environment with Python 3.9 and pip..."
conda create --name "${ENV_NAME}" python=3.9 pip -c conda-forge --yes

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

# --- Install blobtoolkit using pip ---
echo "Installing blobtoolkit (BlobToolKit v4) into '${ENV_NAME}' environment using pip..."
pip install "blobtoolkit[full]"

if [ $? -eq 0 ]; then
    echo "BlobToolKit installed successfully via pip."
else
    echo "ERROR: pip installation failed for blobtoolkit. Check error file for details."
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
    # Use blobtools -v or --version based on actual usage in doc, v4 uses -v
    blobtools -v 2>&1 || echo "Could not retrieve blobtools version."
else
    echo "ERROR: 'blobtools' command NOT FOUND after installation. Something went wrong."
    echo "Please ensure 'blobtoolkit' was installed correctly."
fi

echo ""
echo "--- BlobToolKit v4 Environment Setup and Installation Complete ---"
echo "Job finished on: $(date)"
