#!/bin/bash
#SBATCH --job-name=wgg_vcf_filter
#SBATCH --output=/home/aescudero/wgg/logs/wgg_vcf_filter.log
#SBATCH --error=/home/aescudero/wgg/logs/wgg_vcf_filter.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1              # VCFtools filtering itself is largely single-threaded.
#SBATCH --mem=32G                      # Adjust based on your VCF size; can be memory intensive.
#SBATCH --time=12:00:00                # Set a generous time limit.

# Exit immediately if a command exits with a non-zero status.
# Print commands and their arguments as they are executed.
# Exit if any command in a pipeline fails.
set -euxo pipefail

# Load Conda environment where vcftools, bgzip, and tabix are installed.
# We assume they are in the 'bcftools' environment you created.
source /home/aescudero/miniforge3/etc/profile.d/conda.sh
conda activate bcftools

# --- Configuration ---
# Directory where the VCF from Phase 2 is located and where the filtered VCF will be saved.
BAM_DIR="/home/aescudero/wgg/bams"

# Input VCF from Phase 2 (the one produced by bcftools call).
INPUT_VCF="${BAM_DIR}/all_samples_calls.vcf.gz"

# Prefix for the output filtered VCF from vcftools
OUTPUT_FILTERED_VCF_PREFIX="${BAM_DIR}/all_samples_calls_filtered"
# The actual uncompressed file vcftools outputs (with the .recode.vcf suffix)
UNCOMPRESSED_VCF="${OUTPUT_FILTERED_VCF_PREFIX}.recode.vcf"
# The desired final gzipped and indexed VCF name
FINAL_FILTERED_VCF="${OUTPUT_FILTERED_VCF_PREFIX}.recode.vcf.gz"


echo "--- Starting VCF Filtering ---"
echo "Input VCF: $INPUT_VCF"
echo "Output Filtered VCF: $FINAL_FILTERED_VCF"

# --- Check if input VCF and its index exist ---
if [ ! -f "$INPUT_VCF" ]; then
    echo "Error: Input VCF file not found: $INPUT_VCF"
    exit 1
fi
# Check for either .tbi OR .csi index
if [ ! -f "${INPUT_VCF}.tbi" ] && [ ! -f "${INPUT_VCF}.csi" ]; then
    echo "Error: Input VCF index file (.tbi or .csi) not found for: $INPUT_VCF"
    echo "Please ensure `bcftools index` completed successfully in Phase 2 and created an index."
    exit 1
fi

# --- Run vcftools filtering ---
echo "Running `vcftools` with specified filters (including indels)..."
# Parameters:
# --gzvcf      : Input VCF is gzipped
# --minDP 6    : Minimum depth for a genotype call
# --maxDP 100  : Maximum depth to remove potential paralogs/repeats
# --minGQ 30   : Minimum genotype quality
# --max-missing 0.5 : Max 50% missing data allowed per site (50% completeness)
# --min-alleles 2 : Keep only sites with at least 2 alleles (i.e., polymorphic)
# --max-alleles 2 : Keep only biallelic sites (removes tri-allelic etc. but keeps indels)
# --maf 0.01   : Minimum Minor Allele Frequency of 1%
# --recode     : Recode the VCF to a new file
# --recode-INFO-all : Keep all INFO field annotations from the original VCF
# --out        : Output file prefix (vcftools adds .recode.vcf or .vcf)
vcftools --gzvcf "$INPUT_VCF" \
    --minDP 6 \
    --maxDP 100 \
    --minGQ 30 \
    --max-missing 0.5 \
    --min-alleles 2 \
    --max-alleles 2 \
    --maf 0.01 \
    --recode \
    --recode-INFO-all \
    --out "$OUTPUT_FILTERED_VCF_PREFIX"

FILTER_STATUS=$?
if [ $FILTER_STATUS -ne 0 ]; then
    echo "Error: `vcftools` filtering failed. Status: $FILTER_STATUS"
    # Clean up any partial output from vcftools
    rm -f "$UNCOMPRESSED_VCF"
    exit 1
fi

echo "Successfully filtered VCF. Raw output: $UNCOMPRESSED_VCF"

# --- Compress and Index the Filtered VCF ---
echo "Compressing filtered VCF with bgzip..."
bgzip -f "$UNCOMPRESSED_VCF" # Use the correct uncompressed filename
BGZIP_STATUS=$?
if [ $BGZIP_STATUS -ne 0 ]; then
    echo "Error: bgzip failed. Status: $BGZIP_STATUS"
    exit 1
fi

echo "Indexing compressed filtered VCF with tabix..."
tabix -p vcf "$FINAL_FILTERED_VCF" # Use the correct final gzipped filename
TABIX_STATUS=$?
if [ $TABIX_STATUS -ne 0 ]; then
    echo "Error: tabix failed. Status: $TABIX_STATUS"
    exit 1
fi

echo "Finished VCF filtering and indexing."
echo "Final filtered and indexed VCF: $FINAL_FILTERED_VCF"
