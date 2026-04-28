#!/usr/bin/bash
#SBATCH --job-name=qc_hifi_reads
#SBATCH --output=qc_hifi_reads_output
#SBATCH --error=qc_hifi_reads_error
#SBATCH --partition=standard
#SBATCH --nodes=1                  
#SBATCH --ntasks=6                 
#SBATCH --cpus-per-task=4          
#SBATCH --time=6:00:00 

## Set the number of threads to use based on the allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


## Define variables
INDIR="/home/aescudero/genome_assembly/raw_data/PacBio"
OUTDIR="/home/aescudero/genome_assembly/raw_data/PacBio/qc"
SAMPLE_DIR="/home/aescudero/genome_assembly/raw_data/samples.txt"


### Create a function to run NanoPlot
nanoplot_func () {
SAMPLE=$1

#Create a directory for each sample
mkdir -p $OUTDIR/${SAMPLE}

#Run NanoPlot
## Load and activate the necessary conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nanoplot
NanoPlot -t $SLURM_CPUS_PER_TASK --fastq $INDIR/${SAMPLE}/${SAMPLE}_hifi.fastq -o $OUTDIR/${SAMPLE} -p ${SAMPLE}_hifi --loglength --plots dot -f jpeg
conda deactivate

echo "nanoplot ${SAMPLE} done"
}

### Parallelize over samples
export INDIR OUTDIR SAMPLE_DIR
export -f nanoplot_func
parallel 'nanoplot_func {1}' :::: $SAMPLE_DIR
