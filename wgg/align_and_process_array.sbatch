#!/bin/bash
#SBATCH --job-name=wgg_align
#SBATCH --output=/home/aescudero/wgg/logs/wgg_align_%A_%a.log
#SBATCH --error=/home/aescudero/wgg/logs/wgg_align_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # Number of threads for BWA, fastp, and Samtools
#SBATCH --mem=64G                        # Adjust based on genome size and read depth
#SBATCH --time=12:00:00                  # Adjust as needed (HH:MM:SS)
#SBATCH --array=0-191                    # For 192 samples (0 to 191)

# Add debugging flags at the beginning of the script
set -euxo pipefail

# Load Conda environment (assuming it's set up correctly for your user)
source /home/aescudero/miniforge3/etc/profile.d/conda.sh
conda activate fastp

# --- Configuration ---
SAMPLES_LIST="/home/aescudero/wgg/samples.txt"
RAW_DATA_DIR="/home/aescudero/wgg/rawdata"
BAM_DIR="/home/aescudero/wgg/bams"
GENOME="/home/aescudero/wgg/genomes/genomecbory.fa"
THREADS=$SLURM_CPUS_PER_TASK # This will be 8 based on SBATCH --cpus-per-task

# Ensure output directories exist
mkdir -p "$BAM_DIR"
mkdir -p "/home/aescudero/wgg/logs"

# --- Get Sample ID based on SLURM_ARRAY_TASK_ID ---
readarray -t SAMPLES < "$SAMPLES_LIST"

SAMPLE_ID="${SAMPLES[$SLURM_ARRAY_TASK_ID]}"

echo "Total samples in list: ${#SAMPLES[@]}. Array range should be 0-$((${#SAMPLES[@]} - 1))"
echo "--- Processing sample: $SAMPLE_ID (Array Task ID: $SLURM_ARRAY_TASK_ID) ---"
echo "Raw Data Directory: $RAW_DATA_DIR"
echo "BAM Output Directory: $BAM_DIR"
echo "Reference Genome: $GENOME"
echo "Number of Threads: $THREADS"

# --- Find FastQ files for the current sample ---
R1_FILES=()
R2_FILES=()

echo "DEBUG: Searching for files with pattern: ${SAMPLE_ID}_*.fq.gz in $RAW_DATA_DIR"

mapfile -d $'\0' -t ALL_FQ_FILES < <(find "$RAW_DATA_DIR" -type f -name "${SAMPLE_ID}_*.fq.gz" -print0)

[[ "${ALL_FQ_FILES[-1]}" == "" ]] && unset 'ALL_FQ_FILES[${#ALL_FQ_FILES[@]}-1]'

echo "DEBUG: mapfile captured ${#ALL_FQ_FILES[@]} files:"
for f in "${ALL_FQ_FILES[@]}"; do
    echo "  MAPFILE_FILE: $f"
done

for file in "${ALL_FQ_FILES[@]}"; do
    echo "DEBUG: Checking file in loop (from mapfile): $file"
    filename=$(basename "$file")
    echo "DEBUG: Extracted filename for comparison: $filename"

    if [[ "$filename" == *"1.fq.gz" ]]; then
        echo "DEBUG: Matched R1: $file"
        R1_FILES+=("$file")
    elif [[ "$filename" == *"2.fq.gz" ]]; then
        echo "DEBUG: Matched R2: $file"
        R2_FILES+=("$file")
    else
        echo "DEBUG: No R1/R2 match for: $file"
    fi
done

echo "DEBUG: R1_FILES array has ${#R1_FILES[@]} elements."
for f in "${R1_FILES[@]}"; do echo "  R1_FILE: $f"; done
echo "DEBUG: R2_FILES array has ${#R2_FILES[@]} elements."
for f in "${R2_FILES[@]}"; do echo "  R2_FILE: $f"; done

if [ ${#R1_FILES[@]} -eq 0 ] || [ ${#R2_FILES[@]} -eq 0 ]; then
    echo "Error: Could not find matching R1 or R2 FastQ files for sample $SAMPLE_ID in $RAW_DATA_DIR."
    exit 1
fi

IFS=$'\n' R1_FILES=($(sort <<<"${R1_FILES[*]}"))
IFS=$'\n' R2_FILES=($(sort <<<"${R2_FILES[*]}"))
unset IFS

# --- Determine input FastQ paths for fastp ---
FASTP_R1_INPUT=""
FASTP_R2_INPUT=""
TEMP_UNCOMPRESSED_R1=""
TEMP_UNCOMPRESSED_R2=""

# Process R1 files
if [ ${#R1_FILES[@]} -gt 1 ]; then
    echo "Multiple R1 files found. Concatenating to /dev/shm for fastp input..."
    TEMP_UNCOMPRESSED_R1="/dev/shm/${SAMPLE_ID}_merged_R1.fq"
    zcat "${R1_FILES[@]}" > "$TEMP_UNCOMPRESSED_R1"
    CONCAT_R1_STATUS=$?
    if [ $CONCAT_R1_STATUS -ne 0 ]; then
        echo "Error: Concatenating R1 files failed for sample $SAMPLE_ID. Status: $CONCAT_R1_STATUS"
        exit 1
    fi
    # Fastp can take uncompressed input, so no need to re-gzip for fastp.
    FASTP_R1_INPUT="$TEMP_UNCOMPRESSED_R1"
    echo "Using uncompressed merged R1: $FASTP_R1_INPUT"
else
    echo "Only one R1 file found. Using original path: ${R1_FILES[0]}"
    FASTP_R1_INPUT="${R1_FILES[0]}" # Use the original file directly
fi

# Process R2 files
if [ ${#R2_FILES[@]} -gt 1 ]; then
    echo "Multiple R2 files found. Concatenating to /dev/shm for fastp input..."
    TEMP_UNCOMPRESSED_R2="/dev/shm/${SAMPLE_ID}_merged_R2.fq"
    zcat "${R2_FILES[@]}" > "$TEMP_UNCOMPRESSED_R2"
    CONCAT_R2_STATUS=$?
    if [ $CONCAT_R2_STATUS -ne 0 ]; then
        echo "Error: Concatenating R2 files failed for sample $SAMPLE_ID. Status: $CONCAT_R2_STATUS"
        if [ -n "$TEMP_UNCOMPRESSED_R1" ]; then rm -f "$TEMP_UNCOMPRESSED_R1"; fi # Cleanup R1 if it was merged
        exit 1
    fi
    # Fastp can take uncompressed input, so no need to re-gzip for fastp.
    FASTP_R2_INPUT="$TEMP_UNCOMPRESSED_R2"
    echo "Using uncompressed merged R2: $FASTP_R2_INPUT"
else
    echo "Only one R2 file found. Using original path: ${R2_FILES[0]}"
    FASTP_R2_INPUT="${R2_FILES[0]}" # Use the original file directly
fi


# --- Run fastp, bwa, and samtools pipeline ---
echo "Running fastp, bwa, and samtools pipeline for $SAMPLE_ID with specified parameters..."

# The entire pipeline in one single pipe for efficiency
fastp \
    --stdout \
    --trim_poly_g \
    --qualified_quality_phred 28 \
    --unqualified_percent_limit 30 \
    -i "$FASTP_R1_INPUT" \
    -I "$FASTP_R2_INPUT" \
    --thread "$THREADS" \
    --json "${BAM_DIR}/${SAMPLE_ID}.fastp.json" \
    --html "${BAM_DIR}/${SAMPLE_ID}.fastp.html" | \
bwa mem -p -M -t "$THREADS" "$GENOME" - | \
samtools sort -n - | \
samtools fixmate -m - - | \
samtools sort - | \
samtools markdup -r - "${BAM_DIR}/${SAMPLE_ID}.bam"
ALIGN_STATUS=$?

if [ $ALIGN_STATUS -ne 0 ]; then
    echo "Error: Alignment and marking duplicates failed for sample $SAMPLE_ID. Status: $ALIGN_STATUS"
    # Cleanup only temporary merged FastQ files and partial BAM
    if [ -n "$TEMP_UNCOMPRESSED_R1" ]; then rm -f "$TEMP_UNCOMPRESSED_R1"; fi
    if [ -n "$TEMP_UNCOMPRESSED_R2" ]; then rm -f "$TEMP_UNCOMPRESSED_R2"; fi
    rm -f "${BAM_DIR}/${SAMPLE_ID}.bam" "${BAM_DIR}/${SAMPLE_ID}.fastp.json" "${BAM_DIR}/${SAMPLE_ID}.fastp.html"
    exit 1
fi

# --- Run samtools index ---
echo "Running samtools index for $SAMPLE_ID..."
samtools index -@ "$THREADS" "${BAM_DIR}/${SAMPLE_ID}.bam"
INDEX_STATUS=$?

if [ $INDEX_STATUS -ne 0 ]; then
    echo "Error: Samtools index failed for sample $SAMPLE_ID. Status: $INDEX_STATUS"
    # Cleanup merged FastQ, final BAM, and any partial index
    if [ -n "$TEMP_UNCOMPRESSED_R1" ]; then rm -f "$TEMP_UNCOMPRESSED_R1"; fi
    if [ -n "$TEMP_UNCOMPRESSED_R2" ]; then rm -f "$TEMP_UNCOMPRESSED_R2"; fi
    rm -f "${BAM_DIR}/${SAMPLE_ID}.bam" "${BAM_DIR}/${SAMPLE_ID}.bam.bai" "${BAM_DIR}/${SAMPLE_ID}.fastp.json" "${BAM_DIR}/${SAMPLE_ID}.fastp.html"
    exit 1
fi

echo "Successfully processed sample: $SAMPLE_ID"

# --- Cleanup temporary files ---
echo "Cleaning up temporary merged FastQ files for $SAMPLE_ID..."
if [ -n "$TEMP_UNCOMPRESSED_R1" ]; then rm -f "$TEMP_UNCOMPRESSED_R1"; fi
if [ -n "$TEMP_UNCOMPRESSED_R2" ]; then rm -f "$TEMP_UNCOMPRESSED_R2"; fi

echo "Finished processing sample: $SAMPLE_ID"
