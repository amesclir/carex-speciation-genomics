#!/usr/bin/bash
#SBATCH --job-name=smudgeplot
#SBATCH --output=smudgeplot_output
#SBATCH --error=smudgeplot_error
#SBATCH --partition=standard
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                
#SBATCH --mem=240G #This program requires a lot of memory     
#SBATCH --time=23:00:00

# Define directories and input files
SAMPLE_DIR="/home/aescudero/genome_assembly/raw_data/samples.txt"  # File with sample names (one per line)
INDIR="/home/aescudero/genome_assembly/raw_data/PacBio"            # Directory containing input .fastq files
OUTDIR="/home/aescudero/genome_assembly/raw_data/PacBio/qc"        # Directory where output files will be saved

### Function to run smudgeplot
smudgeplot_func(){
SAMPLE=$1

cd $OUTDIR/${SAMPLE}

# Load and activate the necessary conda environment
source ~/anaconda3/etc/profile.d/conda.sh


##Calculate unique kmer pairs. This is the most
##time and ressource consuming step
conda activate smudgeplot

##Then we do the plot the results.
smudgeplot.py plot ${SAMPLE}_kmer_pairs_coverages.tsv -o ${SAMPLE}_smudgeplot
}

###Parallelize the function
export INDIR OUTDIR SAMPLE_DIR
export -f smudgeplot_func
parallel 'smudgeplot_func {1}' :::: $SAMPLE_DIR

