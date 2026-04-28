#!/bin/bash
#SBATCH --job-name=wgg_vcf2phylip_maf05 # Job name for MAF 0.05 filter using vcf2phylip
#SBATCH --output=/home/aescudero/wgg/logs/wgg_vcf2phylip_maf05.log # New log file
#SBATCH --error=/home/aescudero/wgg/logs/wgg_vcf2phylip_maf05.err # New error log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Exit immediately if a command exits with a non-zero status.
# Print commands and their arguments as they are executed.
# Exit if any command in a pipeline fails.
set -euxo pipefail

# Load Conda environment
source /home/aescudero/miniforge3/etc/profile.d/conda.sh
conda activate bcftools

# --- Configuration ---
BAM_DIR="/home/aescudero/wgg/bams"
INPUT_FILTERED_VCF="${BAM_DIR}/all_samples_calls_filtered.recode.vcf.gz"

# Intermediate SNP-only VCF (filtered for MAC >= 20)
POLYM_SNP_VCF="${BAM_DIR}/all_samples_calls_filtered.polymorphic_snps_mac20.vcf.gz"
POLYM_SNP_VCF_STATS="${BAM_DIR}/all_samples_calls_filtered.polymorphic_snps_mac20.vcf.stats"

# Output PHYLIP alignment file (from vcf2phylip)
PHYLIP_ALIGNMENT="${BAM_DIR}/all_samples_polymorphic_snps_mac20.phylip"

# Path to vcf2phylip.py
VCF2PHYLIP_SCRIPT="/home/aescudero/software/vcf2phylip/vcf2phylip.py"

echo "--- Starting VCF to PHYLIP Conversion (MAC >= 20 Filter with vcf2phylip) ---"
echo "Input Filtered VCF: $INPUT_FILTERED_VCF"
echo "Polymorphic SNP VCF (MAC >= 20): $POLYM_SNP_VCF"
echo "Output Polymorphic SNP VCF Stats: $POLYM_SNP_VCF_STATS"
echo "Output PHYLIP Alignment: $PHYLIP_ALIGNMENT"
echo "Using vcf2phylip.py from: $VCF2PHYLIP_SCRIPT"


# --- Main Workflow ---

# 1. Filter the VCF for biallelic SNPs with MAC >= 20
# We use -v snps -m2 -M2 to ensure only biallelic SNPs are passed to vcf2phylip
# The -i 'MAC>=20' filter is also applied here.
echo "Filtering VCF for biallelic SNPs with MAC >= 20 using bcftools..."
bcftools view --threads $SLURM_CPUS_PER_TASK -v snps -m2 -M2 -i 'MAC>=20' -Oz -o "$POLYM_SNP_VCF" "$INPUT_FILTERED_VCF"
bcftools index "$POLYM_SNP_VCF"

if [ ! -f "$POLYM_SNP_VCF" ] || ([ ! -f "${POLYM_SNP_VCF}.tbi" ] && [ ! -f "${POLYM_SNP_VCF}.csi" ]); then
    echo "Error: Polymorphic SNP VCF or its index not created."
    exit 1
fi
echo "Polymorphic SNP VCF (MAC >= 20) created and indexed: $POLYM_SNP_VCF"

# 2. Run bcftools stats on the polymorphic SNP VCF for debugging
echo "Running bcftools stats on the polymorphic SNP VCF (MAC >= 20): $POLYM_SNP_VCF"
bcftools stats "$POLYM_SNP_VCF" > "$POLYM_SNP_VCF_STATS"

if [ ! -f "$POLYM_SNP_VCF_STATS" ]; then
    echo "Error: bcftools stats output for MAC >= 20 VCF not created."
    exit 1
fi
echo "bcftools stats for polymorphic SNP VCF (MAC >= 20) written to: $POLYM_SNP_VCF_STATS"


# 3. Convert the polymorphic SNP VCF to PHYLIP format using vcf2phylip.py
echo "Converting polymorphic SNP VCF (MAC >= 20) to PHYLIP format with vcf2phylip.py..."
# CORRECTED: Changed -p to --output-prefix for the prefix argument
python "$VCF2PHYLIP_SCRIPT" -i "$POLYM_SNP_VCF" --output-folder "$(dirname "$PHYLIP_ALIGNMENT")" --output-prefix "$(basename "$PHYLIP_ALIGNMENT" .phylip)"

# Check if PHYLIP alignment was created. vcf2phylip.py adds '.min1.phy' by default.
# It should also create a names file.
if [ ! -f "${PHYLIP_ALIGNMENT}.min1.phy" ]; then
    echo "Error: PHYLIP alignment not created by vcf2phylip. Expected: ${PHYLIP_ALIGNMENT}.min1.phy"
    exit 1
fi
echo "PHYLIP alignment created: ${PHYLIP_ALIGNMENT}.min1.phy"

echo "--- VCF to PHYLIP Conversion Completed Successfully ---"
echo "Output PHYLIP alignment is located at: ${PHYLIP_ALIGNMENT}.min1.phy"
