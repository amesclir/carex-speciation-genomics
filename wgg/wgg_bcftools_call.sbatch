#!/bin/bash
#SBATCH --job-name=wgg_bcftools_call
#SBATCH --output=/home/aescudero/wgg/logs/wgg_bcftools_call.log
#SBATCH --error=/home/aescudero/wgg/logs/wgg_bcftools_call.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16               # Using 16 threads as per your original command
#SBATCH --mem=128G                        # Increased memory for joint calling
#SBATCH --time=168:00:00                  # Increased time for large datasets

# Add debugging flags
set -euxo pipefail

# Load Conda environment
source /home/aescudero/miniforge3/etc/profile.d/conda.sh
conda activate bcftools

# --- Configuration ---
SAMPLES_LIST="/home/aescudero/wgg/samples.txt"
BAM_DIR="/home/aescudero/wgg/bams"
GENOME="/home/aescudero/wgg/genomes/genomecbory.fa"
OUTPUT_VCF_NAME="all_samples_calls.vcf.gz"

THREADS=$SLURM_CPUS_PER_TASK

# Ensure output directories exist
mkdir -p "/home/aescudero/wgg/logs"

echo "--- Starting `bcftools` variant calling for all samples ---"
echo "BAM Directory: $BAM_DIR"
echo "Reference Genome: $GENOME"
echo "Output VCF: ${BAM_DIR}/${OUTPUT_VCF_NAME}"
echo "Number of Threads: $THREADS"

# --- Step 1: Create a list of BAM files ---
BAMLIST="${BAM_DIR}/wgg_bamlist.txt"
echo "Creating BAM list file: $BAMLIST"

# Find all .bam files and check for their indexes.
find "$BAM_DIR" -type f -name "*.bam" | sort > "$BAMLIST"

if [ ! -s "$BAMLIST" ]; then
    echo "Error: No BAM files found in $BAM_DIR. Please ensure Phase 1 completed successfully."
    exit 1
fi

BAM_COUNT=$(wc -l < "$BAMLIST")
echo "Found $BAM_COUNT BAM files."

# --- Step 2: Run `bcftools mpileup` and `bcftools call` ---
echo "Running `bcftools mpileup` and `bcftools call`..."
bcftools mpileup \
    -Ou \
    -f "$GENOME" \
    -b "$BAMLIST" \
    --threads "$THREADS" \
    -q 20 \
    -a DP | \
bcftools call \
    --threads "$THREADS" \
    -m \
    -Oz \
    -o "${BAM_DIR}/${OUTPUT_VCF_NAME}"

CALL_STATUS=$?
if [ $CALL_STATUS -ne 0 ]; then
    echo "Error: `bcftools call` failed. Status: $CALL_STATUS"
    exit 1
fi

# --- Step 3: Index the final VCF ---
echo "Indexing the final VCF..."
bcftools index \
    "${BAM_DIR}/${OUTPUT_VCF_NAME}"
INDEX_STATUS=$?
if [ $INDEX_STATUS -ne 0 ]; then
    echo "Error: `bcftools index` failed. Status: $INDEX_STATUS"
    exit 1
fi

echo "Successfully completed variant calling for all samples."
echo "Final VCF and index are located at: ${BAM_DIR}/${OUTPUT_VCF_NAME}"
