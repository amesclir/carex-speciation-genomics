#!/bin/bash
#SBATCH --job-name=dsuite_fix
#SBATCH --output=/home/aescudero/wgg/dsuite/dsuite_%j.out
#SBATCH --error=/home/aescudero/wgg/dsuite/dsuite_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=standard

set -e

source /home/aescudero/miniforge3/etc/profile.d/conda.sh
conda activate bcftools

# Paths
DSUITE_EXEC="/home/aescudero/software/Dsuite/Build/Dsuite"
ORIG_VCF="/home/aescudero/wgg/bams/all_samples_calls_filtered.recode.vcf.gz"
CLEAN_VCF="/home/aescudero/wgg/dsuite/all_samples_renamed.vcf.gz"
RENAME_MAP="/home/aescudero/wgg/dsuite/sample_rename_map.txt"
SETS="/home/aescudero/wgg/dsuite/SETS_borbonica_flow.txt"
TRIOS="/home/aescudero/wgg/dsuite/test_trios.txt"
OUT_DIR="/home/aescudero/wgg/dsuite/resultsborbonica"

mkdir -p $OUT_DIR

# --- STEP 0: Fix VCF Names ---
# Only run this if the clean VCF doesn't exist yet to save time
if [ ! -f "$CLEAN_VCF" ]; then
    echo "Renaming samples in VCF..."
    bcftools reheader -s "$RENAME_MAP" "$ORIG_VCF" -o "$CLEAN_VCF"
    bcftools index -t "$CLEAN_VCF"
fi

echo "--- STARTING DSUITE ANALYSIS ---"
cd $OUT_DIR

# Step 1: Dtrios
echo "Running Dtrios..."
$DSUITE_EXEC Dtrios "$CLEAN_VCF" "$SETS" -o borbonica_flow

# Step 2: Dinvestigate
echo "Running Dinvestigate..."
$DSUITE_EXEC Dinvestigate -w 50,25 "$CLEAN_VCF" "$SETS" "$TRIOS" -n window_analysis

echo "--- FINISHED ---"
