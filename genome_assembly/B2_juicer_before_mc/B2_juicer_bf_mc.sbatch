#!/bin/bash
#SBATCH --job-name=B2_juicer_bf_mc
#SBATCH --export=ALL
#SBATCH --output=B2_juicer_bf_mc_output
#SBATCH --error=B2_juicer_bf_mc_error
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=23:30:00

### Load directories
INDIR=/home/aescudero/genome_assembly/B1_hiC_mapping
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


## Sort .BAM file (the one with hiC mappings) by read name
BAM_UNSORTED=$INDIR/${SAMPLE}/repdir/${SAMPLE}_final.bam
samtools sort $BAM_UNSORTED -o $INDIR/${SAMPLE}/repdir/${SAMPLE}_final_sorted.bam -O BAM -n -@ 16
# Create a variable for the sorted .BAM file
BAM=$INDIR/${SAMPLE}/repdir/${SAMPLE}_final_sorted.bam

echo "BAM sorting done for ${SAMPLE}"

## Procede with scaffolding with yahs. Always use -q 0 to set the minimum
## read mapping quality to 0
REF=/home/aescudero/genome_assembly/A2_purge_dups/${SAMPLE}/purged.fa
yahs $REF $BAM -o $OUTDIR/${SAMPLE}/${SAMPLE}_yahs_out -e GATC -e GANTC --no-mem-check -q 0
echo "scaffolding with yahs done for ${SAMPLE}"

### Generate HiC contact maps:
## First, convert the yahs output to a juicer input. Here as well, also use -q 0!.

#Check if you have the indexing (.fai) of your genome assembly. This will be used to sort the juicer output.
(juicer pre -q 0 $OUTDIR/${SAMPLE}/${SAMPLE}_yahs_out.bin $OUTDIR/${SAMPLE}/${SAMPLE}_yahs_out_scaffolds_final.agp $REF.fai | \
LC_ALL=C sort -k2,2d -k6,6d -T $OUTDIR/${SAMPLE} --parallel=16 -S256G | \
awk 'NF' > $OUTDIR/${SAMPLE}/alignments_sorted.txt.part) && \
(mv $OUTDIR/${SAMPLE}/alignments_sorted.txt.part $OUTDIR/${SAMPLE}/alignments_sorted.txt)
echo "transformed yahs to juicer for ${SAMPLE}"

## Then index the final scaffold file and prepare a chromosome (scaffold) size file, where we indicate the size of scaffolds

samtools faidx $OUTDIR/${SAMPLE}/${SAMPLE}_yahs_out_scaffolds_final.fa
cut -f1-2 $OUTDIR/${SAMPLE}/${SAMPLE}_yahs_out_scaffolds_final.fa.fai > $OUTDIR/${SAMPLE}/${SAMPLE}_scaffolds.chrom.sizes
echo "scaffold size file done for ${SAMPLE}"

 ## Now, the hiC contact matrix is created, considering the size of the scaffolds
 ## indicated in a chrom.sizes file

(java -jar -Xmx120G $SOFTWAREDIR/juicer/CPU/common/juicer_tools.jar pre $OUTDIR/${SAMPLE}/alignments_sorted.txt \
$OUTDIR/${SAMPLE}/out.hic.part $OUTDIR/${SAMPLE}/${SAMPLE}_scaffolds.chrom.sizes) && \
(mv $OUTDIR/${SAMPLE}/out.hic.part $OUTDIR/${SAMPLE}/${SAMPLE}_out.hic)
echo "hiC contact map created for ${SAMPLE}"

## Repeat the creation of hiC contact matrix but generating an adapted output format
## compatible with "Juicebox", specifying the mode (-a)

juicer pre -q 0 -a -o $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT $OUTDIR/${SAMPLE}/${SAMPLE}_yahs_out.bin \
$OUTDIR/${SAMPLE}/${SAMPLE}_yahs_out_scaffolds_final.agp ${REF}.fai 2> $OUTDIR/${SAMPLE}/tmp_juicer_pre_JBAT.log
cat $OUTDIR/${SAMPLE}/tmp_juicer_pre_JBAT.log | grep "PRE_C_SIZE" | cut -d' ' -f2- > $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT.chrom.sizes

(java -jar -Xmx256G $SOFTWAREDIR/juicer/CPU/common/juicer_tools.jar pre $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT.txt $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT.hic.part $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT.chrom.sizes) && (mv $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT.hic.part $OUTDIR/${SAMPLE}/${SAMPLE}_JBAT.hic)

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
