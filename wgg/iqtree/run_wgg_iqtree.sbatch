#!/bin/bash
#SBATCH --job-name=wgg_iqtree_mac20     # Job name (changed to reflect your current analysis)
#SBATCH --output=wgg_iqtree_mac20.log   # Output log file (changed name)
#SBATCH --error=wgg_iqtree_mac20.err   # Error log file (changed name)
#SBATCH --ntasks=1                    # Number of tasks (should be 1 for IQ-TREE)
#SBATCH --cpus-per-task=48            # Number of CPU cores per task (as you have)
#SBATCH --mem=400G                     # Memory per node (as you have)
#SBATCH --time=150:00:00              # Time limit hrs:min:sec (generous, good for large data)
#SBATCH --partition=compute           # Partition (queue) name (as you have)

# --- Setup ---
echo "Starting IQ-TREE analysis setup at $(date)"

# Load Miniconda module or conda environment
module load Miniconda3/24.5.0           # Adjust if your cluster's module name/version changed
source activate iqtree-2.3.4            # Ensure this is the correct conda environment where IQ-TREE is installed

# Check if conda environment activated successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'iqtree-2.3.4'. Exiting."
    exit 1
fi

# Define the path to your alignment file (UPDATED PATH)
ALIGNMENT_FILE="/home/aescudero/wgg/bams/all_samples_polymorphic_snps_mac20.min4.phy.varsites.phy.fixed.phy.varsites.phy"

# Check if the alignment file exists
if [ ! -f "$ALIGNMENT_FILE" ]; then
    echo "ERROR: Alignment file not found: $ALIGNMENT_FILE. Exiting."
    exit 1
fi

echo "Input Alignment File: $ALIGNMENT_FILE"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE" # SLURM_MEM_PER_NODE often reflects what --mem requests

# --- Run IQ-TREE ---
echo "Running IQ-TREE command..."
# Run IQ-TREE with ModelFinder Plus, Ultrafast Bootstrap, SH-aLRT, and automatic thread detection
# -m MFP: ModelFinder Plus for comprehensive model selection (recommended over a fixed model like GTR+ASC)
# -bb 1000: Ultrafast Bootstrap (UFBoot) with 1000 replicates (faster and often more accurate than standard -b)
# -alrt 1000: SH-aLRT test with 1000 replicates (another type of branch support)
# -nt AUTO: IQ-TREE will automatically use all available CPUs as specified by --cpus-per-task
# -redo: Allows resuming an interrupted run or overwriting previous output
iqtree -s "$ALIGNMENT_FILE" -st DNA -m GTR+ASC -bb 1000 -alrt 1000 -nt AUTO -redo

# --- Post-run checks ---
if [ $? -eq 0 ]; then
    echo "IQ-TREE analysis completed successfully at $(date)"
    echo "Output tree file should be: ${ALIGNMENT_FILE}.treefile"
else
    echo "ERROR: IQ-TREE analysis failed or exited with an error at $(date)."
    exit 1
fi

# Deactivate the conda environment (optional, but good practice)
conda deactivate
