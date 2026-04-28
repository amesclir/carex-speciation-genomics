#!/bin/bash
#SBATCH --job-name=B2_juicer_bf_mc
#SBATCH --export=ALL
#SBATCH --output=B2_juicer_bf_mc_output
#SBATCH --error=B2_juicer_bf_mc_error
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=23:30:00

### Load directories
INDIR=/home/aescudero/genome_assembly/B1_hifihiC_map/B11_hiCmap_shallow_mod
OUTDIR=/home/aescudero/genome_assembly/B2_juicer_before_mc
SOFTWAREDIR=/home/aescudero/software
SAMPLE_DIR="/home/aescudero/genome_assembly/raw_data/samples.txt"

### Create function to parallelize over samples
juicer_bf_mc() {
SAMPLE=$1

## Establish working directories
mkdir -p $OUTDIR/${SAMPLE}

##Load conda source and activate environments
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ref_map

export JAVA_HOME=~/software/jdk-21.0.4+7
export PATH=$JAVA_HOME/bin:$PATH


(java -jar -Xmx256G $SOFTWAREDIR/juicer/CPU/common/juicer_tools.jar pre $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT.txt $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT.hic.part $OUTDIR/${SAMPLE}/
${SAMPLE}_JBAT.chrom.sizes) && (mv $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT.hic.part $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT.hic)

echo "hiC contact map compatible with Juicebox created for ${SAMPLE}"

conda deactivate
}

### Parallelize over individuals
export INDIR OUTDIR SOFTWAREDIR SAMPLE_DIR
export -f juicer_bf_mc
parallel 'juicer_bf_mc {1}' :::: $SAMPLE_DIR

### SOFTWARE VERSIONS
# samtools 1.20
# yahs 1.2a.2
# juicer tools 1.22.01
