#!/bin/bash
#SBATCH --job-name=B3_af_mc
#SBATCH --export=ALL
#SBATCH --output=B3_af_mc_output
#SBATCH --error=B3_af_mc_error
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=47:00:00

### Load directories
INDIR=/home/aescudero/genome_assembly/B2_juicer_before_mc
OUTDIR=/home/aescudero/genome_assembly/B3_juicer_after_mc
SOFTWAREDIR=/home/aescudero/software
SAMPLE_DIR="/home/aescudero/genome_assembly/raw_data/samples.txt"

### Create function to parallelize over samples
hic_af_mc() {
SAMPLE=$1

## Create output directories
mkdir -p $OUTDIR/${SAMPLE}
cd $OUTDIR/${SAMPLE}

##Load conda source
source ~/anaconda3/etc/profile.d/conda.sh

##Estalish path to reference genome
REF=/home/aescudero/genome_assembly/A2_purge_dups/${SAMPLE}/purged.fa

## Modify the file with the genome assembly according to the (agp)
## manually curated .assembly file.

conda activate ref_map

export JAVA_HOME=~/software/jdk-21.0.4+7
export PATH=$JAVA_HOME/bin:$PATH

juicer post -o $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT $INDIR/${SAMPLE}/${SAMPLE}_JBAT.review.assembly $INDIR/${SAMPLE}/${SAMPLE}_JBAT.liftover.agp ${REF}
echo "juicer post mc step done for ${SAMPLE}"

## Then, recreate a final hic contact map comptaible with Juicebox using the reviewed version of
## the assembly file.

juicer pre -q 0 -a -o $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT_final $INDIR/${SAMPLE}/${SAMPLE}_yahs_out.bin $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT.FINAL.agp ${REF}.fai 2> $OUTDIR/${SAMPLE}/tmp_juicer_pre_JBAT.log

cat $OUTDIR/${SAMPLE}/tmp_juicer_pre_JBAT.log | grep "PRE_C_SIZE" | cut -d' ' -f2- > $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT_final.chrom.sizes

(java -jar -Xmx120G $SOFTWAREDIR/juicer/CPU/common/juicer_tools.jar pre $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT_final.txt $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT_final.hic.part $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT_final.chrom.sizes) && (mv $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT_final.hic.part $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT_final.hic)

conda deactivate
}

### Parallelize over individuals
export INDIR OUTDIR SOFTWAREDIR SAMPLE_DIR
export -f hic_af_mc
parallel 'hic_af_mc {1}' :::: $SAMPLE_DIR

### SOFTWARE VERSIONS
# yahs 1.2a.2
# juicer tools 1.22.01
