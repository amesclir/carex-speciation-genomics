#!/usr/bin/bash

# SLURM job submission settings:
#SBATCH --job-name=qc_jellyfish_raw      # Job name to identify the task
#SBATCH --output=qc_jellyfish_raw_output # File for standard output logs
#SBATCH --error=qc_jellyfish_raw_error   # File for error logs
#SBATCH --partition=standard             # Job partition/queue
#SBATCH --nodes=1                        # Number of nodes to use 
#SBATCH --ntasks=1                       # Number of tasks 
#SBATCH --cpus-per-task=16               # Number of CPU cores allocated to the task
#SBATCH --time=12:00:00                  # Maximum allowed runtime 

# Define directories and input files
SAMPLE_DIR="/home/aescudero/genome_assembly/raw_data/samples.txt"  # File with sample names (one per line)
INDIR="/home/aescudero/genome_assembly/raw_data/PacBio"            # Directory containing input .fastq files
OUTDIR="/home/aescudero/genome_assembly/raw_data/PacBio/qc"        # Directory where output files will be saved

# Set the number of threads to use based on the allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Function to run Jellyfish for k-mer counting and histogram creation
jellyfish_histo(){
  SAMPLE=$1  
  # Create outout directory
  mkdir -p $OUTDIR/${SAMPLE}
  # Load and activate the necessary conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kmers
  # Count k-mers in the input .fastq file following the parameters specified in GenomeScope
  # -C: count canonical k-mers (counts both strands)
  # -m 21: k-mer length of 21
  # -t: number of threads
  # -s 1000000000: hash size for the k-mer database
  # -o: output file for the counted k-mers
  jellyfish count -C -m 21 -t $SLURM_CPUS_PER_TASK $INDIR/${SAMPLE}/${SAMPLE}_hifi.fastq -s 1000000000 -o $OUTDIR/${SAMPLE}/${SAMPLE}.jf
  echo "k-mer counting for ${SAMPLE} completed"

  # Generate a histogram of k-mer frequencies
  # -t: number of threads
  # Output is redirected to a .histo file
  jellyfish histo -t $SLURM_CPUS_PER_TASK $OUTDIR/${SAMPLE}/${SAMPLE}.jf > $OUTDIR/${SAMPLE}/${SAMPLE}.histo
  echo "Histogram generation for ${SAMPLE} completed"
}

# Export necessary variables and function for use by GNU Parallel
export INDIR OUTDIR SAMPLE_DIR
export -f jellyfish_histo

# Run the jellyfish_histo function in parallel for each sample listed in SAMPLE_DIR
parallel 'jellyfish_histo {1}' :::: $SAMPLE_DIR

