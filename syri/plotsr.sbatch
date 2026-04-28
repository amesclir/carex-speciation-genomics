#!/usr/bin/bash
#SBATCH --job-name=plotsr
#SBATCH --output=plotsr.out
#SBATCH --error=plotsr.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --partition=standard
#SBATCH --time=24:00:00

# Set the working directory where your input files (syri.out, refgenome, qrygenome) are located
cd /home/aescudero/syri/

# Load Conda environment for plotsr
echo "Activating plotsr conda environment..."
source ~/.bashrc # Or ~/.bash_profile, or path to conda.sh if not automatically in bashrc
conda activate plotsr
echo "Conda environment 'plotsr' activated."

echo "Starting plotsr visualization..."
echo "Input files..."
echo "Output image will be generated in the current directory."
echo "plotsr running..."

# Run plotsr command directly
# If 'plotsr' is installed via Conda/pip into your 'plotsr' environment,
# it should be directly available in your PATH after activating the env.
plotsr --sr syri_final2.out --genomes genomes.txt -o Carex_boryana_borbonica_def.png

# Check if plotsr command was successful
if [ $? -eq 0 ]; then
    echo "plotsr visualization completed successfully."
else
    echo "Error: plotsr visualization failed. Check error logs for details."
    exit 1
fi

echo "Job finished."

# Deactivate conda environment when done (optional but good practice)
conda deactivate
