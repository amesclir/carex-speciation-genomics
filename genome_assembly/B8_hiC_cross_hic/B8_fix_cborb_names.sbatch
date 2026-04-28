#!/bin/bash
#SBATCH --job-name=B8_fix_hic
#SBATCH --output=B8_fix_hic_%j.out
#SBATCH --error=B8_fix_hic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --partition=standard
#SBATCH --time=12:00:00

set -e

# --- Environment & Paths ---
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ref_map

export JAVA_HOME=~/software/jdk-21.0.4+7
export PATH=$JAVA_HOME/bin:$PATH

# Change this to the path of your NEW, reordered fasta file
NEW_REF=/home/aescudero/syri/my_new_genome_cborb.fasta

# Existing directories
SUB_DIR=/home/aescudero/genome_assembly/B8_hiC_cross_hic/C_boryana_on_genomecborb_rev
SOFTWAREDIR=/home/aescudero/software

echo "====================================="
echo "Fixing chromosome names for C_boryana_on_genomecborb_rev..."

# 1. Translate names and fix Juicer sort order
echo "1. Translating names in alignments_sorted.txt..."

awk -v OFS="\t" '
BEGIN {
    map["scaffold_1"]="scaffold_1";   map["scaffold_3"]="scaffold_2";   map["scaffold_2"]="scaffold_3";
    map["scaffold_6"]="scaffold_4";   map["scaffold_4"]="scaffold_5";   map["scaffold_7"]="scaffold_6";
    map["scaffold_9"]="scaffold_7";   map["scaffold_5"]="scaffold_8";   map["scaffold_8"]="scaffold_9";
    map["scaffold_10"]="scaffold_10"; map["scaffold_11"]="scaffold_11"; map["scaffold_12"]="scaffold_12";
    map["scaffold_13"]="scaffold_13"; map["scaffold_15"]="scaffold_14"; map["scaffold_14"]="scaffold_15";
    map["scaffold_18"]="scaffold_16"; map["scaffold_17"]="scaffold_17"; map["scaffold_16"]="scaffold_18";
    map["scaffold_20"]="scaffold_19"; map["scaffold_19"]="scaffold_20"; map["scaffold_21"]="scaffold_21";
    map["scaffold_23"]="scaffold_22"; map["scaffold_22"]="scaffold_23"; map["scaffold_26"]="scaffold_24";
    map["scaffold_25"]="scaffold_25"; map["scaffold_24"]="scaffold_26"; map["scaffold_27"]="scaffold_27";
    map["scaffold_29"]="scaffold_28"; map["scaffold_28"]="scaffold_29"; map["scaffold_31"]="scaffold_30";
    map["scaffold_32"]="scaffold_31"; map["scaffold_34"]="scaffold_32"; map["scaffold_33"]="scaffold_33";
    map["scaffold_30"]="scaffold_34";
}
{
    if ($3 in map) $3 = map[$3]
    if ($7 in map) $7 = map[$7]

    if ($3 > $7 || ($3 == $7 && $4 > $8)) {
        tmpStr=$2; $2=$6; $6=tmpStr;
        tmpChr=$3; $3=$7; $7=tmpChr;
        tmpPos=$4; $4=$8; $8=tmpPos;
        tmpMap=$10; $10=$11; $11=tmpMap;
    }
    print $0
}' $SUB_DIR/alignments_sorted.txt | LC_ALL=C sort -k3,3d -k7,7d -T $SUB_DIR --parallel=16 -S50G > $SUB_DIR/alignments_RENAMED_sorted.txt

# 2. Generate new chrom.sizes from your NEW fasta
echo "2. Creating chrom.sizes from new reference..."
if [ ! -f "${NEW_REF}.fai" ]; then
    samtools faidx $NEW_REF
fi
cut -f1-2 ${NEW_REF}.fai > $SUB_DIR/my_new_genome_cborb.chrom.sizes

# 3. Build the final .hic file
echo "3. Generating corrected .hic file..."
java -jar -Xmx80G $SOFTWAREDIR/juicer/CPU/common/juicer_tools.jar pre \
    $SUB_DIR/alignments_RENAMED_sorted.txt \
    $SUB_DIR/C_boryana_on_my_new_genome_cborb.hic \
    $SUB_DIR/my_new_genome_cborb.chrom.sizes

echo "--- FIX COMPLETE ---"
