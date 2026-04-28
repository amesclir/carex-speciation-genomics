#!/bin/bash
#SBATCH --job-name=earl_grey
#SBATCH --output=earl_grey_output
#SBATCH --error=earl_grey_error
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=360:00:00

SAMPLE_DIR="/home/aescudero/genome_assembly/raw_data/samples.txt"
INDIR=~/genome_assembly/B6_clean_assemblies
OUTDIR=~/genome_assembly/C3_earlgrey
OUTDIR_SOFT=~/genome_assembly/C4_earlgrey_softmasking
earlgrey_func() {
    SAMPLE=$1
    #Load conda source and earlgrey package
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate earlgrey
export PATH=/home/aescudero/anaconda3/envs/earlgrey/bin:$PATH
    #Load genome
    genome_pth=$INDIR/${SAMPLE}/${SAMPLE}.fasta

    #Establish output directory
    mkdir -p $OUTDIR/${SAMPLE}

    #Run earlgrey
    earlGrey -g $genome_pth -s ${SAMPLE} -o $OUTDIR/${SAMPLE} -t 16 -r 38820

    #Sofmasking using earlgrey
    #Establish output
    mkdir -p $OUTDIR_SOFT/${SAMPLE}

    bedtools maskfasta -soft -fi $genome_pth \
    -bed $OUTDIR/${SAMPLE}/${SAMPLE}_EarlGrey/${SAMPLE}_summaryFiles/${SAMPLE}.filteredRepeats.bed \
    -fo $OUTDIR_SOFT/${SAMPLE}/${SAMPLE}_earlgrey.softmasked.fasta

    echo "Soft masking complete"
    conda deactivate
}

### Parallelize over samples
export INDIR OUTDIR OUTDIR_SOFT SAMPLE_DIR
export -f earlgrey_func
parallel 'earlgrey_func {1}' :::: $SAMPLE_DIR
