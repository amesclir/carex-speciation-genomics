#!/bin/bash
#SBATCH --job-name=B4_decontaminate
#SBATCH --export=ALL
#SBATCH --output=B4_decontaminate_output
#SBATCH --error=B4_decontaminate_error
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH --time=47:00:00


##Load directories
INDIR=/home/aescudero/genome_assembly/B3_juicer_after_mc
OUTDIR=/home/aescudero/genome_assembly/B4_decontaminate
SOFTWAREDIR=/home/aescudero/software
SAMPLE_DIR="/home/aescudero/genome_assembly/raw_data/samples.txt"

export PATH=~/anaconda3/bin:$PATH
# Ensure anaconda3 is prioritized

## Run the Blobtools pipeline in parallel for each sample
blobs_func() {
source ~/anaconda3/etc/profile.d/conda.sh

SAMPLE=$1
REF_GEN=$INDIR/${SAMPLE}/${SAMPLE}_JBAT.FINAL.fa
READS=/home/aescudero/genome_assembly/raw_data/PacBio/${SAMPLE}/${SAMPLE}_hifi.fastq

##Create output folder for each sample
mkdir -p $OUTDIR/${SAMPLE}
cd $OUTDIR/${SAMPLE}

###  Create yaml file for the sample
#touch ${SAMPLE}.yaml
#echo -e "assembly:\n  alias: ${SAMPLE}\n  record_type: contig\ntaxon:\n  name: ${SAMPLE}" > ${SAMPLE}.yaml
#    echo "yaml file created"

## Create coverage file (adding directly the bam does not work for some reason)

#conda activate minimap_samtools
#minimap2 -ax map-pb -t 16 $REF_GEN $READS | samtools sort -@16 -O BAM -o ${SAMPLE}_coverage.bam
#samtools coverage ${SAMPLE}_coverage.bam > ${SAMPLE}_coverage.txt
#conda deactivate
#echo "coverage file created"

## Use tiara to quickly check if a contig is eukaryote, bacteria or organelle
#        conda activate tiara
#    tiara -i $REF_GEN -o ${SAMPLE}.tiara -t 16 --pr --tf all -m 1000
#        conda deactivate
#    echo "tiara run"

## Adding input files to blobtoolkit dataset
conda activate btk
mkdir -p $OUTDIR/${SAMPLE}/btk
blobtools create --fasta $REF_GEN --meta ${SAMPLE}.yaml --taxdump ../taxdump/ --replace ./btk
blobtools add --text ${SAMPLE}_coverage.txt --text-header --text-cols '#rname=identifier,meandepth=reads_cov' --replace ./btk
blobtools add --key plot.y=reads_cov --replace ./btk
blobtools add --text ${SAMPLE}.tiara --text-delimiter "\t" --text-header --text-cols "sequence_id=identifier,class_fst_stage=tiara" --replace ./btk
blobtools add --key plot.cat=tiara --replace ./btk
echo "bloobtools run"

## View the dataset on blobtool viewer:: call from python3

export RUST_BACKTRACE=full

# Create a summary table on the dataset
blobtools filter --table $OUTDIR/${SAMPLE}/${SAMPLE}_unfiltered_summary.tsv --table-fields gc,length,reads_cov,tiara ./btk

conda deactivate

        ### The decontaminated assembly is eukarya_${SAMPLE}_JBAT.FINAL.fa
        ### Keep mitochondrial genome and bacterial genomes separated
}
### Parallelize over individuals
export INDIR OUTDIR SOFTWAREDIR SAMPLE_DIR
export -f blobs_func
parallel 'blobs_func {1}' :::: $SAMPLE_DIR

### SOFTWARE VERSIONS
# samtools 1.20
# minimap2 2.28
# blobtoolkit v4.3.0
# blobtk 0.5.3
# tiara 1.0.3
