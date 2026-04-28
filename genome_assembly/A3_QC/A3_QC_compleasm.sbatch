#!/usr/bin/bash
#SBATCH --job-name=A3_QC
#SBATCH --output=A3_QC_compleasm_output
#SBATCH --error=A3_QC_compleasm_error
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=30G
#SBATCH --partition=standard
#SBATCH --time=12:00:00

###Set directories
OUTDIR="/home/aescudero/genome_assembly/A3_QC"
INDIR_UNPURGED="/home/aescudero/genome_assembly/A1_hifihiC_map"
INDIR_PURGED="/home/aescudero/genome_assembly/A2_purge_dups"
SAMPLE_DIR="/home/aescudero/genome_assembly/raw_data/samples.txt"

###Prepare loop for quality control:
for SAMPLE in $(cat "$SAMPLE_DIR" | cut -f1)
do

## Load conda source:
source ~/anaconda3/etc/profile.d/conda.sh


##First, we will run the tests of the unpurged assembly

#Create directories
mkdir -p $OUTDIR/${SAMPLE}
mkdir -p $OUTDIR/${SAMPLE}/unpurged
cd $OUTDIR/${SAMPLE}/unpurged

#First test: We will test genome completedness with compleasm

conda activate compleasm
compleasm run -a $INDIR_UNPURGED/${SAMPLE}/${SAMPLE}.hifihic.asm.hic.p_ctg.fa  -o $OUTDIR/${SAMPLE}/unpurged/ -l poales_odb10 -t 16
conda deactivate

#Second test: We will test contiguity with Gfastats

#conda activate gfastats
#gfastats $INDIR_UNPURGED/${SAMPLE}/${SAMPLE}.hifihic.asm.hic.p_ctg.fa -j 16 > $OUTDIR/${SAMPLE}/unpurged/${SAMPLE}_gfastats.txt
#conda deactivate

# Third test: General contig lenth distribution with perl script.

# perl /home/aescudero/software/contig_length_fasta.pl $INDIR_UNPURGED/${SAMPLE}/${SAMPLE}.hifihic.asm.hic.p_ctg.fa > $OUTDIR/${SAMPLE}/unpurged/${SAMPLE}_contig_length.txt

 # Forth test:  Did we purge enough? with merqury

#conda activate merqury
#READS=/home/aescudero/genome_assembly/raw_data/PacBio/${SAMPLE}/${SAMPLE}_hifi.fastq
#meryl count k-mers in the sequenced reads, this step is only
#needed once per genome
#meryl k=21 count $READS output $OUTDIR/${SAMPLE}/${SAMPLE}.meryl
#merqury.sh $OUTDIR/${SAMPLE}/${SAMPLE}.meryl $INDIR_UNPURGED/${SAMPLE}/${SAMPLE}.hifihic.asm.hic.p_ctg.fa $INDIR_UNPURGED/${SAMPLE}/${SAMPLE}.hifihic.asm.hic.a_ctg.fa  merqury_${SAMPLE}_unpurged
#conda deactivate

##Then, we run them for the purged assembly.

mkdir -p $OUTDIR/${SAMPLE}/purged
cd $OUTDIR/${SAMPLE}/purged

conda activate compleasm
compleasm run -a $INDIR_PURGED/${SAMPLE}/purged.fa  -o $OUTDIR/${SAMPLE}/purged/ -l poales_odb10 -t 16
conda deactivate

#conda activate gfastats
#gfastats $INDIR_PURGED/${SAMPLE}/purged.fa -j 16 > $OUTDIR/${SAMPLE}/purged/${SAMPLE}_gfastats.txt
#conda deactivate

#perl /home/aescudero/software/contig_length_fasta.pl $INDIR_PURGED/${SAMPLE}/purged.fa > $OUTDIR/${SAMPLE}/purged/${SAMPLE}_contig_length.txt

#conda activate merqury
#merqury.sh $OUTDIR/${SAMPLE}/${SAMPLE}.meryl $INDIR_PURGED/${SAMPLE}/purged.fa $INDIR_PURGED/${SAMPLE}/alternate/purged_alt.fa merqury_${SAMPLE}_purged
#conda deactivate

done

### SOFTWARE VERSIONS
# compleasm 0.2.6
# contig_length_fasta.pl
# gfastat 1.3.6
# merqury 1.3
