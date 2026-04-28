#!/bin/bash
#SBATCH --job-name=dsuite_SV2
#SBATCH --output=/home/aescudero/wgg/dsuite/dsuite_SV2_%j.out
#SBATCH --error=/home/aescudero/wgg/dsuite/dsuite_SV2_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=standard

set -e

source /home/aescudero/miniforge3/etc/profile.d/conda.sh
conda activate bcftools

# Paths
DSUITE_EXEC="/home/aescudero/software/Dsuite/Build/Dsuite"
CLEAN_VCF="/home/aescudero/wgg/dsuite/all_samples_renamed.vcf.gz"
SETS="/home/aescudero/wgg/dsuite/SETS_boryana_flow.txt"
TRIOS="/home/aescudero/wgg/dsuite/test_trios.txt"
OUT_DIR="/home/aescudero/wgg/dsuite/resultsboryana_SV2"

# The specific inversion coordinates on scaffold_14
REGION="scaffold_18:1681715-2543617"

mkdir -p $OUT_DIR
cd $OUT_DIR

echo "--- STARTING SV2 BARRIER ANALYSIS ---"
date

# 1. Count the number of variants in the inversion region
# NUMLINES is required by Dsuite when using piped (stdin) input
echo "Counting variants in region $REGION..."
NUMLINES=$(bcftools view -H -r "$REGION" "$CLEAN_VCF" | wc -l)
echo "Found $NUMLINES variants."

# 2. Run Dtrios on SV2 specifically via pipe
# This will give the specific D-statistic for the inversion block
echo "Running Dtrios on SV2 via stdin..."
bcftools view -r "$REGION" "$CLEAN_VCF" | \
$DSUITE_EXEC Dtrios -l $NUMLINES stdin "$SETS" -o SV2_only

# 3. Run Dinvestigate for windowed resolution within the inversion
# Using a smaller window (25 SNPs) to capture variation inside the 2.3Mb block
echo "Running Dinvestigate for localized f_d..."
$DSUITE_EXEC Dinvestigate -w 25,10 "$CLEAN_VCF" "$SETS" "$TRIOS" -n SV2_windows

echo "--- FINISHED ---"
date
