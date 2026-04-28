#!/bin/bash
#SBATCH --job-name=B6_clean_assemblies
#SBATCH --export=ALL
#SBATCH --output=B6_clean_assemblies_output
#SBATCH --error=B6_clean_assemblies_error
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH --time=47:00:00

### Set directories
INDIR=~/genome_assembly/B4_decontaminate
OUTDIR=~/genome_assembly/B6_clean_assemblies
SAMPLE_DIR=~/genome_assembly/raw_data/samples.txt


for SAMPLE in $(cat $SAMPLE_DIR | cut -f1)
do


mkdir -p $OUTDIR/${SAMPLE}
cd $OUTDIR/${SAMPLE}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate seqkit

MINLEN=4000000 #maybe modify it according to contig length list, hiC and telomeres.
seqkit seq --threads 16 --min-len $MINLEN $INDIR/${SAMPLE}/eukarya_${SAMPLE}_JBAT.FINAL.fa > $OUTDIR/${SAMPLE}/${SAMPLE}.fasta
conda deactivate

echo "clean assemblies generated without small, umapped scaffolds for ${SAMPLE}"


perl ~/software/contig_length_fasta.pl ${SAMPLE}.fasta > clean_assembly_${SAMPLE}_scaffold_length.txt

conda activate gfastats
gfastats ${SAMPLE}.fasta -j 16 > clean_assembly_${SAMPLE}_gfastats.txt
conda deactivate

echo "trimming test done for ${SAMPLE}"

done
