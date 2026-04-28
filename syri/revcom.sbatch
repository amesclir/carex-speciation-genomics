#!/usr/bin/bash
#SBATCH --job-name=rev_complement
#SBATCH --output=rev_complement.out
#SBATCH --error=rev_complement.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --partition=standard
#SBATCH --time=24:00:00

source ~/miniforge3/etc/profile.d/conda.sh
conda activate seqkit

input_fasta=genomecborb.fa
output_fasta=genomecborb_rev.fasta

# Step 1: List of chromosomes to reverse
cat <<EOF > chrom_list.txt
scaffold_1
scaffold_3
scaffold_4
scaffold_7
scaffold_8
scaffold_9
scaffold_11
scaffold_14
scaffold_17
scaffold_21
scaffold_22
scaffold_25
scaffold_27
scaffold_29
scaffold_31
scaffold_32
EOF

# Step 2: Reverse complement only chromosomes selected
seqkit grep -n -f chrom_list.txt "$input_fasta" | seqkit seq -r -p -t DNA > reversed_chroms.fasta

# Step 3: Extract not modified chromosomes
seqkit grep -v -n -f chrom_list.txt "$input_fasta" > unchanged_chroms.fasta

# Step 4: Combine everything
cat unchanged_chroms.fasta reversed_chroms.fasta > "$output_fasta"
