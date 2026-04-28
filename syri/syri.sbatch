#!/usr/bin/bash
#SBATCH --job-name=syri
#SBATCH --output=syri.out
#SBATCH --error=syri.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --partition=standard
#SBATCH --time=24:00:00


# Set the working directory where your input files (out.sam, refgenome, qrygenome) are located
cd /home/aescudero/syri/

# Define the full path to the SyRI executable
#PATH_TO_SYRI="/home/aescudero/miniforge3/envs/syri/bin"

# Load Conda environment for SyRI
echo "Activating syri conda environment..."
source ~/.bashrc # Or ~/.bash_profile, or path to conda.sh if not automatically in bashrc
conda activate syri
echo "Conda environment 'syri' activated."

echo "Starting SyRI analysis..."
echo "Input SAM: out.sam"
echo "Reference: refgenome"
echo "Query: qrygenome"
echo "SyRI command: python3 ${PATH_TO_SYRI} -c out.sam -r refgenome -q qrygenome -k -F S"

# Run SyRI command
syri -c out.sam -r refgenome -q qrygenome -k -F S

# Check if SyRI command was successful
if [ $? -eq 0 ]; then
    echo "SyRI analysis completed successfully."
else
    echo "Error: SyRI analysis failed. Check error logs for details."
    exit 1
fi

echo "Job finished."

# Deactivate conda environment when done (optional but good practice)
conda deactivate
