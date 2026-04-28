#!/bin/bash
#SBATCH --job-name=B8_hiC_cross_hic
#SBATCH --output=B8_hiC_cross_hic_%j.out
#SBATCH --error=B8_hiC_cross_hic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=standard
#SBATCH --time=24:00:00

set -e

# --- Environment & Paths ---
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ref_map

export JAVA_HOME=~/software/jdk-21.0.4+7
export PATH=$JAVA_HOME/bin:$PATH

INDIR=/home/aescudero/genome_assembly/B7_hiC_cross_mapping
OUTDIR=/home/aescudero/genome_assembly/B8_hiC_cross_hic
SOFTWAREDIR=/home/aescudero/software

mkdir -p $OUTDIR

# Define references
REF_BORY=/home/aescudero/syri/genomecbory.fa
REF_BORB=/home/aescudero/syri/genomecborb_rev.fasta

generate_cross_hic() {
    SAMPLE=$1
    REF_ID=$2
    REF_FASTA=$3

    PREFIX="${SAMPLE}_on_${REF_ID}"
    echo "====================================="
    echo "Processing $PREFIX..."
    
    SUB_OUT=$OUTDIR/$PREFIX
    mkdir -p $SUB_OUT

    # 1. Name-sort the BAM from B7 (required to match read pairs)
    BAM_IN=$INDIR/$PREFIX/repdir/${SAMPLE}_final.bam
    BAM_NSORT=$SUB_OUT/${PREFIX}_nsort.bam

    echo "1. Name sorting BAM..."
    samtools sort -n -@ 16 -o $BAM_NSORT $BAM_IN

    # 2. Convert to Juicer format with internal chromosome swapping
    echo "2. Converting to Juicer format & Sorting..."
    
    samtools view $BAM_NSORT | awk -v OFS="\t" '{
        # Drop unmapped, secondary, duplicates, and supplementary
        if (and($2, 3332)) next;
        
        if ($1 == prev_name) {
            str2 = (and($2, 16)) ? 1 : 0;
            chr2 = $3; pos2 = $4; mapq2 = $5;
            
            if (chr1 != "*" && chr2 != "*") {
                # FIX: Force chr1 to always be <= chr2 to prevent multi-block errors
                if (chr1 > chr2 || (chr1 == chr2 && pos1 > pos2)) {
                    # Swap Read 1 and Read 2
                    print prev_name, str2, chr2, pos2, 1, str1, chr1, pos1, 0, mapq2, mapq1;
                } else {
                    # Keep original order
                    print prev_name, str1, chr1, pos1, 0, str2, chr2, pos2, 1, mapq1, mapq2;
                }
            }
            prev_name = ""; # Reset for next pair
        } else {
            prev_name = $1;
            str1 = (and($2, 16)) ? 1 : 0;
            chr1 = $3; pos1 = $4; mapq1 = $5;
        }
    }' | LC_ALL=C sort -k3,3d -k7,7d -T $SUB_OUT --parallel=16 -S100G > $SUB_OUT/alignments_sorted.txt

    # 3. Create chrom.sizes file from the reference index
    echo "3. Creating chrom.sizes..."
    if [ ! -f "${REF_FASTA}.fai" ]; then
        samtools faidx $REF_FASTA
    fi
    cut -f1-2 ${REF_FASTA}.fai > $SUB_OUT/${REF_ID}.chrom.sizes

    # 4. Generate the final .hic map using juicer_tools
    echo "4. Generating .hic file..."
    java -jar -Xmx200G $SOFTWAREDIR/juicer/CPU/common/juicer_tools.jar pre \
        $SUB_OUT/alignments_sorted.txt \
        $SUB_OUT/${PREFIX}.hic \
        $SUB_OUT/${REF_ID}.chrom.sizes

    # Cleanup large intermediate name-sorted BAM
    rm $BAM_NSORT
    echo "Finished generating .hic for $PREFIX"
}

# --- Run for both Cross-Mapping combinations ---
generate_cross_hic "C_borbonica" "genomecbory" "$REF_BORY"
generate_cross_hic "C_boryana" "genomecborb_rev" "$REF_BORB"

echo "--- CROSS HIC GENERATION COMPLETE ---"
