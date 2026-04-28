#!/usr/bin/bash
#SBATCH --job-name=C1_repeatmask
#SBATCH --output=C1_repeatmask_output
#SBATCH --error=C1_repeatmask_error
#SBATCH --nodes=1                  # Using one node
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --partition=standard
#SBATCH --time=48:00:00

### Set directories
INDIR=/home/aescudero/genome_assembly/B6_clean_assemblies
OUTDIR=/home/aescudero/genome_assembly/C1_repeatmask
SAMPLE_DIR="/home/aescudero/genome_assembly/raw_data/samples.txt"

### Run repeatMasker in parallel

rmasker() {
    SAMPLE=$1
    REF_trim=$INDIR/${SAMPLE}/${SAMPLE}.fasta

    #set source for conda and activate environment
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate repeatmasker

    #establish working directories
    mkdir -p $OUTDIR/${SAMPLE}
    cd $OUTDIR/${SAMPLE}

    RepeatMasker -pa 4 -e rmblast -species "plants" -dir $OUTDIR/${SAMPLE} -xsmall -html -gff $REF_trim
    # -pa runs in parallel (pa 4 uses 16 cores in total in combination with rmblast)
    # -e specifies the search engine (rmblast = ncbi is the one used by EarlGrey)
    # -nolow does not mask low complexity regions
    # -s slow but more sensitive, -q fast but less sensitive
    # IMPORTANT if using RepeatMasker before gene prediction, I should not mask low complexity regions
    # because sometimes codons from genes would be masked
    # -xsmall for softmasking (repeats in lowercase) rather than changing them to Ns
    # -html and -gff create additional output files
        conda deactivate
}

### Parallelize over individuals
export INDIR OUTDIR SAMPLE_DIR
export -f rmasker
parallel 'rmasker {1}' :::: $SAMPLE_DIR

### SOFTWARE VERSIONS
# repeatmasker 4.1.5.
