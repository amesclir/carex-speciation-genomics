#!/usr/bin/bash
#SBATCH --job-name=A2_purge_dups_after_hifiasm
#SBATCH --output=A2_purge_dups_after_hifiasm_output
#SBATCH --error=A2_purge_dups_after_hifiasm_error
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --mem=96G
#SBATCH --partition=standard
#SBATCH --time=47:00:00



# Define variables
OUTDIR="/home/aescudero/genome_assembly/A2_purge_dups"
SAMPLE_DIR="/home/aescudero/genome_assembly/raw_data/samples.txt"

# Create the function with all the purge_dups commands
purge_dups_func() {
    SAMPLE=$1

    # Load conda source and activate conda environment
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate purge_dups

    #Load directories
    REF=/home/aescudero/genome_assembly/A1_hifihiC_map/${SAMPLE}/${SAMPLE}.hifihic.asm.hic.p_ctg.fa
    ALT=/home/aescudero/genome_assembly/A1_hifihiC_map/${SAMPLE}/${SAMPLE}.hifihic.asm.hic.a_ctg.fa
    READS=/home/aescudero/genome_assembly/raw_data/PacBio/${SAMPLE}/${SAMPLE}_hifi.fastq

    # Step1: Map reads to reference and calculate coverage
    mkdir -p $OUTDIR/${SAMPLE}
    cd $OUTDIR/${SAMPLE}
    minimap2 -xasm20 -t 12 $REF $READS | gzip -c - > ${SAMPLE}.paf.gz
    #Calculate coverage statistics
    pbcstat ${SAMPLE}.paf.gz
    #Calculate cutoff values based on coverage
    calcuts PB.stat > cutoffs 2> calcults.log
    echo "step 1 ${SAMPLE} done"

    # Intermediate step (optional): Create an histogram representing the coverage
    #and the cutoffs

    /home/aescudero/software/hist_plot.py -c cutoffs PB.stat cov_unpurged_hist.png

    # Step2: Split the assembly to do self-self alignment
    split_fa $REF > ${SAMPLE}.split
    minimap2 -xasm5 -t 12 -DP ${SAMPLE}.split ${SAMPLE}.split | gzip -c - > ${SAMPLE}.split.self.paf.gz
    echo "step 2 ${SAMPLE} done"

    # Step3: Purge haplotigs and overlaps
    purge_dups -2 -T cutoffs -c PB.base.cov ${SAMPLE}.split.self.paf.gz > dups.bed 2> purge_dups.log
    echo "step 3 ${SAMPLE} done"

    # Step4: Get purged primary and haplotigs sequences
    get_seqs -e dups.bed $REF
    echo "step 4 ${SAMPLE} done"

## We will do the same steps with the alternate assembly.
#Create directory for the alternate assembly
    mkdir -p alternate
    cd alternate

# When purging duplicates, more haplotig sequences were
#identified (hap) that should be added to the ones identified
#by hifiasm that are found in the alternate assembly. So we
#will merge both files
    cat $ALT ../hap.fa > ${SAMPLE}_alt_hap.fa
    ALTREF=${SAMPLE}_alt_hap.fa

#Step 1 again: Map reads into the reference genome
    minimap2 -xasm20 -t 12 $ALTREF $READS | gzip -c - > ${SAMPLE}.paf.gz
    pbcstat ${SAMPLE}.paf.gz
    calcuts PB.stat > cutoffs 2> calcults.log
    echo "step 1 part 2 ${SAMPLE} done"

#Step 2 again: Align the reference genome with it self
    split_fa $ALTREF > ${SAMPLE}.split
    minimap2 -xasm5 -t 12 -DP ${SAMPLE}.split ${SAMPLE}.split | gzip -c - > ${SAMPLE}.split.self.paf.gz
    echo "step 2 part 2 ${SAMPLE} done"

#Step 3 again:Purge duplicates
    purge_dups -2 -T cutoffs -c PB.base.cov ${SAMPLE}.split.self.paf.gz > dups.bed 2> purge_dups.log
    echo "step 3 part 2 ${SAMPLE} done"

#Step 4 again: Get the alternate assembly
    get_seqs -e dups.bed $ALTREF
    echo "step 4 part 2 ${SAMPLE} done"

conda deactivate
}

### Parallelize over samples
export OUTDIR SAMPLE_DIR
export -f purge_dups_func
parallel 'purge_dups_func {1}' :::: $SAMPLE_DIR


### SOFTWARE VERSIONS
# minimap2 2.28
# purge_dups 1.2.6
