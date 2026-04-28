#!/usr/bin/bash
#SBATCH --job-name=A1_hifiasm_hifihic
#SBATCH --output=A1_hifiasm_hifihic_output
#SBATCH --error=A1_hifiasm_hifihic_error
#SBATCH --nodes=1                  
#SBATCH --ntasks=32
#SBATCH --mem=40G
#SBATCH --partition=standard
#SBATCH --time=48:00:00


# Define variables
INDIR_hifi="/home/aescudero/genome_assembly/raw_data/PacBio"
INDIR_hic="/home/aescudero/genome_assembly/raw_data/hiC"
OUTDIR="/home/aescudero/genome_assembly/A1_hifihiC_map/"
SAMPLE_DIR="/home/aescudero/genome_assembly/raw_data/samples.txt"

#Run hifiasm
hifiasm_assembly() {
SAMPLE=$1
mkdir -p $OUTDIR/${SAMPLE}
# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hifiasm
hifiasm --primary -t 32 -D 10 -N 200 -o $OUTDIR/${SAMPLE}/${SAMPLE}.hifihic.asm --h1 $INDIR_hic/${SAMPLE}/${SAMPLE}_hiC_1.fastq --h2 $INDIR_hic/${SAMPLE}/${SAMPLE}_hiC_2.fastq $INDIR_hifi/${SAMPLE}/${SAMPLE}_hifi.fastq
echo "hifiasm ${SAMPLE} done"
conda deactivate
}

### Parallelize over samples
export INDIR_hifi INDIR_hic OUTDIR SAMPLE_DIR
export -f hifiasm_assembly
parallel 'hifiasm_assembly {1}' :::: $SAMPLE_DIR

###   VERSIONS
# hifiasm 0.19.9
