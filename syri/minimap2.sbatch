#!/usr/bin/bash
#SBATCH --job-name=minimap2
#SBATCH --output=minimap2.out
#SBATCH --error=minimap2.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --partition=standard
#SBATCH --time=24:00:00

# Set the working directory where your symlinks and input files reside
cd /home/aescudero/syri/

# Load Conda environment for minimap2
echo "Activating minimap2 conda environment..."
source ~/.bashrc # or ~/.bash_profile, or path to conda.sh if it's not automatically in bashrc
conda activate minimap2
echo "Conda environment 'minimap2' activated."

# Define your output file name
OUTPUT_SAM="out.sam"

echo "Starting minimap2 mapping using symlinked files 'refgenome' and 'qrygenome'..."
echo "Output will be: ${OUTPUT_SAM}"

# Run minimap2 command
minimap2 -ax asm5 --eqx -t "$SLURM_CPUS_PER_TASK" refgenome qrygenome > "${OUTPUT_SAM}"

# Check if minimap2 command was successful
if [ $? -eq 0 ]; then
    echo "minimap2 mapping completed successfully."
else
    echo "Error: minimap2 mapping failed. Check error logs for details."
    exit 1
fi

echo "Job finished."

# Deactivate conda environment when done (optional but good practice)
conda deactivate
