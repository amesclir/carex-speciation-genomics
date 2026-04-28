#!/bin/bash
#SBATCH --job-name=B1_hiC_mapping
#SBATCH --output=B1_hiC_mapping_output
#SBATCH --error=B1_hiC_mapping_error
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=32
#SBATCH --partition=standard
#SBATCH --time=47:00:00

####Set directories
INDIR_hic=/home/aescudero/genome_assembly/raw_data/hiC
OUTDIR=/home/aescudero/genome_assembly/B1_hiC_mapping
SOFTWAREDIR=/home/aescudero/software
SAMPLE_DIR="/home/aescudero/genome_assembly/raw_data/samples.txt"

#### Create a function to parallelize the mapping of hiC reads
mapHiCreads() {
SAMPLE=$1
mkdir -p $OUTDIR/${SAMPLE}

##Set conda source and activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ref_map

export JAVA_HOME=~/software/jdk-21.0.4+7
export PATH=$JAVA_HOME/bin:$PATH

## Index ref, both with samtools and bwa
REF=/home/aescudero/genome_assembly/A2_purge_dups/${SAMPLE}/purged.fa
    bwa index -a bwtsw $REF
    samtools faidx $REF

## Trimming 5 bases out of 5' end out of read 1 and read 2,
##as they contain adapter sequencuences

awk '{ if(NR%2==0) {print substr($1,6)} else {print} }' $INDIR_hic/${SAMPLE}/${SAMPLE}_hiC_1.fastq | gzip > $INDIR_hic/${SAMPLE}/${SAMPLE}_R1_5trim.fastq.gz
awk '{ if(NR%2==0) {print substr($1,6)} else {print} }' $INDIR_hic/${SAMPLE}/${SAMPLE}_hiC_2.fastq | gzip > $INDIR_hic/${SAMPLE}/${SAMPLE}_R2_5trim.fastq.gz

echo "5 bases trim done for ${SAMPLE}"

##Trim poly-G tails from the ends of sequencing reads,as they are sequencing artifacts

fastp --trim_poly_g -l 120 -q 30 -w 16 \
--in1 $INDIR_hic/${SAMPLE}/${SAMPLE}_R1_5trim.fastq.gz \
--out1 $INDIR_hic/${SAMPLE}/${SAMPLE}_R1_trimmed.fastq.gz \
--in2 $INDIR_hic/${SAMPLE}/${SAMPLE}_R2_5trim.fastq.gz \
--out2 $INDIR_hic/${SAMPLE}/${SAMPLE}_R2_trimmed.fastq.gz

echo "poly-G tail trim done for ${SAMPLE}"

## Map HiC reads to ref using bwa mem
## First we need to map them as single reads, but we do it twice for both DNA strands

mkdir -p $OUTDIR/${SAMPLE}/rawbamdir

bwa mem -t 16 $REF $INDIR_hic/${SAMPLE}/${SAMPLE}_R1_trimmed.fastq.gz |
samtools view -@ 16 -Sb - > $OUTDIR/${SAMPLE}/rawbamdir/${SAMPLE}_1.bam

bwa mem -t 16 $REF $INDIR_hic/${SAMPLE}/${SAMPLE}_R2_trimmed.fastq.gz |
samtools view -@ 16 -Sb - > $OUTDIR/${SAMPLE}/rawbamdir/${SAMPLE}_2.bam

echo "single read mapping done for ${SAMPLE}"

## We will eliminate the 3' end as it is redundant with the 5' end of the other strand.

mkdir -p $OUTDIR/${SAMPLE}/filtbamdir

samtools view -h $OUTDIR/${SAMPLE}/rawbamdir/${SAMPLE}_1.bam | perl $SOFTWAREDIR/filter_five_end.pl | \
samtools view -Sb - > $OUTDIR/${SAMPLE}/filtbamdir/${SAMPLE}_1.bam

samtools view -h $OUTDIR/${SAMPLE}/rawbamdir/${SAMPLE}_2.bam | perl $SOFTWAREDIR/filter_five_end.pl | \
samtools view -Sb - > $OUTDIR/${SAMPLE}/filtbamdir/${SAMPLE}_2.bam

echo " 3' end eliminated from ${SAMPLE} "

## Now we will pair the two mapped DNA strands using  Map quality 0! is an important parameter which
##outputs a sorted, mapping quality filtered, paired-end BAM file. We then add read groups to this BAM
##file using Picard Tools.

    mkdir -p $OUTDIR/${SAMPLE}/tmpdir

    perl $SOFTWAREDIR/two_read_bam_combiner.pl $OUTDIR/${SAMPLE}/filtbamdir/${SAMPLE}_1.bam \
    $OUTDIR/${SAMPLE}/filtbamdir/${SAMPLE}_2.bam samtools 1 | \
    samtools view -bS -t ${REF}.fai - | samtools sort -@ 16 -o $OUTDIR/${SAMPLE}/tmpdir/${SAMPLE}.bam -

    mkdir -p $OUTDIR/${SAMPLE}/pairdir
    picard -Xmx100g AddOrReplaceReadGroups INPUT=$OUTDIR/${SAMPLE}/tmpdir/${SAMPLE}.bam OUTPUT=$OUTDIR/${SAMPLE}/pairdir/${SAMPLE}.bam ID=${SAMPLE} \
    LB=${SAMPLE} SM=scaffolding  PL=ILLUMINA PU=none

echo "read pairing done for ${SAMPLE}"

## Remove PCR duplicates from pair-end BAM file

    mkdir -p $OUTDIR/${SAMPLE}/repdir
    picard -Xmx100g MarkDuplicates \
    INPUT=$OUTDIR/${SAMPLE}/pairdir/${SAMPLE}.bam OUTPUT=$OUTDIR/${SAMPLE}/repdir/${SAMPLE}_final.bam \
    METRICS_FILE=$OUTDIR/${SAMPLE}/repdir/metrics.replicates.txt TMP_DIR=$OUTDIR/${SAMPLE}/tmpdir/ \
    ASSUME_SORTED=TRUE VALIDATION_STRINGENCY=LENIENT REMOVE_DUPLICATES=TRUE

echo "PCR duplicates removed from ${SAMPLE}"

##Finally, index the final file and calculate statistics to monitor the mapping

samtools index $OUTDIR/${SAMPLE}/repdir/${SAMPLE}_final.bam

perl $SOFTWAREDIR/get_stats.pl $OUTDIR/${SAMPLE}/repdir/${SAMPLE}_final.bam > $OUTDIR/${SAMPLE}/repdir/${SAMPLE}_final.bam.stats

## Not only calculate statistics of the final file but also from intermediate steps. Add them to flagstat file
    touch $OUTDIR/${SAMPLE}/flagstat.txt
    echo ">>> Stats raw bam R1" >> $OUTDIR/${SAMPLE}/flagstat.txt
    samtools flagstat $OUTDIR/${SAMPLE}/rawbamdir/${SAMPLE}_1.bam >> $OUTDIR/${SAMPLE}/flagstat.txt
    echo ">>> Stats raw bam R2" >> $OUTDIR/${SAMPLE}/flagstat.txt
    samtools flagstat $OUTDIR/${SAMPLE}/rawbamdir/${SAMPLE}_2.bam >> $OUTDIR/${SAMPLE}/flagstat.txt
    echo ">>> Stats bam chimeric reads filtered R1" >> $OUTDIR/${SAMPLE}/flagstat.txt
    samtools flagstat $OUTDIR/${SAMPLE}/filtbamdir/${SAMPLE}_1.bam >> $OUTDIR/${SAMPLE}/flagstat.txt
    echo ">>> Stats bam chimeric reads filtered R2" >> $OUTDIR/${SAMPLE}/flagstat.txt
    samtools flagstat $OUTDIR/${SAMPLE}/filtbamdir/${SAMPLE}_2.bam >> $OUTDIR/${SAMPLE}/flagstat.txt
    echo ">>> Stats bam paired and mapping quality filtered" >> $OUTDIR/${SAMPLE}/flagstat.txt
    samtools flagstat $OUTDIR/${SAMPLE}/tmpdir/${SAMPLE}.bam >> $OUTDIR/${SAMPLE}/flagstat.txt
    echo ">>> Stats final bam (duplicates filtered)" >> $OUTDIR/${SAMPLE}/flagstat.txt
    samtools flagstat $OUTDIR/${SAMPLE}/repdir/${SAMPLE}_final.bam >> $OUTDIR/${SAMPLE}/flagstat.txt

echo "index and stats done for ${SAMPLE}"

}

### Parallelize over individuals
export INDIR_hic OUTDIR SOFTWAREDIR SAMPLE_DIR
export -f mapHiCreads
parallel 'mapHiCreads {1}' :::: $SAMPLE_DIR

### SOFTWARE VERSIONS
# fastp 0.23.4
# bwa 0.7.15
# Picard 1.141
# Arima mapping pipeline V03
# Samtools 1.20
