#!/bin/bash
#SBATCH --job-name=B7_hiC_cross
#SBATCH --output=B7_hiC_cross_%j.out
#SBATCH --error=B7_hiC_cross_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --partition=standard
#SBATCH --time=47:00:00

set -e

# --- Environment & Paths ---
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ref_map

INDIR_hic=/home/aescudero/genome_assembly/raw_data/hiC
OUTDIR=/home/aescudero/genome_assembly/B7_hiC_cross_mapping
SOFTWAREDIR=/home/aescudero/software
REF_BORY=/home/aescudero/syri/genomecbory.fa
REF_BORB=/home/aescudero/syri/genomecborb_rev.fasta

mkdir -p $OUTDIR

# --- Mapping Function (Arima Style) ---
run_cross_map() {
    SAMPLE=$1
    REF=$2
    REF_ID=$3
    
    SUB_OUT=$OUTDIR/${SAMPLE}_on_${REF_ID}
    mkdir -p $SUB_OUT/rawbamdir $SUB_OUT/filtbamdir $SUB_OUT/tmpdir $SUB_OUT/repdir
    
    echo "Processing $SAMPLE on $REF_ID..."

    # 1. Indexing (if needed)
    if [ ! -f "${REF}.bwt" ]; then
        bwa index -a bwtsw $REF
        samtools faidx $REF
    fi

    # 2. Trim 5 bases (Arima standard)
    awk '{ if(NR%2==0) {print substr($1,6)} else {print} }' $INDIR_hic/${SAMPLE}/${SAMPLE}_hiC_1.fastq | gzip > $SUB_OUT/${SAMPLE}_R1_5trim.fastq.gz
    awk '{ if(NR%2==0) {print substr($1,6)} else {print} }' $INDIR_hic/${SAMPLE}/${SAMPLE}_hiC_2.fastq | gzip > $SUB_OUT/${SAMPLE}_R2_5trim.fastq.gz

    # 3. Map R1 and R2 separately
    bwa mem -t 32 $REF $SUB_OUT/${SAMPLE}_R1_5trim.fastq.gz | samtools view -@ 16 -Sb - > $SUB_OUT/rawbamdir/${SAMPLE}_1.bam
    bwa mem -t 32 $REF $SUB_OUT/${SAMPLE}_R2_5trim.fastq.gz | samtools view -@ 16 -Sb - > $SUB_OUT/rawbamdir/${SAMPLE}_2.bam

    # 4. Filter 5' ends (requires the perl script from your SOFTWAREDIR)
    samtools view -h $SUB_OUT/rawbamdir/${SAMPLE}_1.bam | perl $SOFTWAREDIR/filter_five_end.pl | samtools view -Sb - > $SUB_OUT/filtbamdir/${SAMPLE}_1.bam
    samtools view -h $SUB_OUT/rawbamdir/${SAMPLE}_2.bam | perl $SOFTWAREDIR/filter_five_end.pl | samtools view -Sb - > $SUB_OUT/filtbamdir/${SAMPLE}_2.bam

    # 5. Combine R1 and R2 (The "Combiner" step)
    perl $SOFTWAREDIR/two_read_bam_combiner.pl $SUB_OUT/filtbamdir/${SAMPLE}_1.bam $SUB_OUT/filtbamdir/${SAMPLE}_2.bam samtools 1 | \
    samtools view -bS -t ${REF}.fai - | samtools sort -@ 16 -o $SUB_OUT/tmpdir/${SAMPLE}_combined.sorted.bam -

    # 6. Add Read Groups (CRITICAL for Picard)
    picard -Xmx100g AddOrReplaceReadGroups \
        INPUT=$SUB_OUT/tmpdir/${SAMPLE}_combined.sorted.bam \
        OUTPUT=$SUB_OUT/tmpdir/${SAMPLE}_with_RG.bam \
        ID=${SAMPLE}_on_${REF_ID} LB=lib1 SM=${SAMPLE} PL=ILLUMINA PU=none

    # 7. Mark Duplicates (The step that was failing)
    # Note: Added ASSUME_SORTED=TRUE to fix the error you just got
    picard -Xmx100g MarkDuplicates \
        INPUT=$SUB_OUT/tmpdir/${SAMPLE}_with_RG.bam \
        OUTPUT=$SUB_OUT/repdir/${SAMPLE}_final.bam \
        METRICS_FILE=$SUB_OUT/repdir/metrics.txt \
        TMP_DIR=$SUB_OUT/tmpdir/ \
        ASSUME_SORTED=TRUE \
        VALIDATION_STRINGENCY=LENIENT \
        REMOVE_DUPLICATES=TRUE

    samtools index $SUB_OUT/repdir/${SAMPLE}_final.bam
    
    # Cleanup large intermediate files
    rm $SUB_OUT/*.fastq.gz
    echo "Finished $SAMPLE on $REF_ID"
}

# --- Main Workflow ---
run_cross_map "C_borbonica" "$REF_BORY" "genomecbory"
run_cross_map "C_boryana" "$REF_BORB" "genomecborb_rev"

echo "--- CROSS MAPPING COMPLETE ---"
