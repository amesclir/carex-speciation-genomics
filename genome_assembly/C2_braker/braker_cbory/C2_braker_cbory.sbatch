#!/usr/bin/bash
#SBATCH --job-name=C2_braker_cbory
#SBATCH --output=C2_braker_output_cbory
#SBATCH --error=C2_braker_error_cbory
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --partition=standard
#SBATCH --time=160:00:00

echo "Starting BRAKER at $(date)"

# Source Conda initialization script
source /home/aescudero/miniforge3/etc/profile.d/conda.sh

# Activate the braker environment
conda activate braker

# Define the environment base path for clarity and correctness
BRAKER3_ENV_PATH="/home/aescudero/miniforge3/envs/braker"

# Set necessary environment variables for BRAKER and its components
export LD_LIBRARY_PATH="$BRAKER3_ENV_PATH/lib/:$LD_LIBRARY_PATH"
export COMPLEASM_PATH="$BRAKER3_ENV_PATH/bin/"
export AUGUSTUS_CONFIG_PATH="$BRAKER3_ENV_PATH/config/"
export AUGUSTUS_BIN_PATH="$BRAKER3_ENV_PATH/bin/"
export AUGUSTUS_SCRIPTS_PATH="$BRAKER3_ENV_PATH/bin/"
export PYTHON3_PATH="$BRAKER3_ENV_PATH/bin/"

# Your external GeneMark and ProtHint paths (these remain as before)
export GENEMARK_PATH=/home/aescudero/software/gmetp_linux_64/bin/gmes
export PROTHINT_PATH=/home/aescudero/software/ProtHint/bin

###Set output directories
OUTDIR="/home/aescudero/genome_assembly/C2_braker"
# Navigate to the main project directory where BRAKER runs
cd "$OUTDIR"

# Define the absolute path to the braker output subdirectory
BRAKER_OUT_DIR="$OUTDIR/braker"

# Run braker
echo "Starting BRAKER"
braker.pl --genome=genomecbory.fa --prot_seq=proteins.fa --threads=32 --species=carex_boryana --busco_lineage=poales_odb10 --workingdir="$BRAKER_OUT_DIR"

conda deactivate

echo "BRAKER job finished."
