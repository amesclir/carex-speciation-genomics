#!/bin/bash
#SBATCH --job-name=Install_HiC_Tools
#SBATCH --output=install_hic_tools_output.txt
#SBATCH --error=install_hic_tools_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G         # Some memory needed for installations
#SBATCH --time=00:30:00  # Give it enough time for downloads and installs

echo "--- Starting software installation script ---"
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo ""

# --- Conda Environment Activation ---
echo "Attempting to source Conda and activate 'ref_map' environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ref_map

if [ $? -eq 0 ]; then
    echo "Conda environment 'ref_map' activated successfully."
    echo "Current Conda environment: $CONDA_DEFAULT_ENV"
else
    echo "ERROR: Failed to activate Conda environment 'ref_map'. Aborting installation."
    exit 1
fi
echo ""

# --- Install missing tools ---
echo "Installing bwa, fastp, picard, and ensuring samtools is up-to-date in 'ref_map' environment..."
# The -y flag answers 'yes' to all prompts, making it non-interactive
conda install -c bioconda bwa fastp picard samtools --yes

if [ $? -eq 0 ]; then
    echo "All specified tools installed successfully (or already present)."
else
    echo "ERROR: Conda installation failed. Check install_hic_tools_error.txt for details."
    echo "This could be due to network issues, dependency conflicts, or insufficient disk space."
    exit 1
fi

echo ""
echo "--- Software installation complete ---"
echo "Job finished on: $(date)"
